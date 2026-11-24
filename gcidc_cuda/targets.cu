// Three-frame column geometry. No geometry leaves device memory.
// Input flows are in the unaugmented target grid, camera transforms in training orientation.
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct V {
    double x, y, z;
};
__device__ V vec(double x, double y, double z) { return {x, y, z}; }
__device__ V plus(V a, V b) { return vec(a.x + b.x, a.y + b.y, a.z + b.z); }
__device__ V mul(V a, double s) { return vec(a.x * s, a.y * s, a.z * s); }
__device__ double dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
__device__ double norm(V a) { return sqrt(dot(a, a)); }
struct Mat {
    double a[3][3];
};
__device__ void outer(Mat &m, V v, V w, double k) {
    double x[] = {v.x, v.y, v.z}, y[] = {w.x, w.y, w.z};
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            m.a[i][j] += k * x[i] * y[j];
}
__device__ V mv(Mat m, V v) {
    return vec(m.a[0][0] * v.x + m.a[0][1] * v.y + m.a[0][2] * v.z,
               m.a[1][0] * v.x + m.a[1][1] * v.y + m.a[1][2] * v.z,
               m.a[2][0] * v.x + m.a[2][1] * v.y + m.a[2][2] * v.z);
}
__device__ double infinity_norm(Mat m) {
    double a = 0;
    for (int i = 0; i < 3; i++)
        a = fmax(a, fabs(m.a[i][0]) + fabs(m.a[i][1]) + fabs(m.a[i][2]));
    return a;
}
// Symmetric 3x3 eigensystem, including the null-space projection for observability.
__device__ int inverse(Mat m, double tol, Mat &out, Mat &nullspace) {
    Mat q = {};
    for (int i = 0; i < 3; i++)
        q.a[i][i] = 1;
    for (int sweep = 0; sweep < 48; sweep++) {
        int p = 0, r = 1;
        for (int i = 0; i < 3; i++)
            for (int j = i + 1; j < 3; j++)
                if (fabs(m.a[i][j]) > fabs(m.a[p][r])) {
                    p = i;
                    r = j;
                }
        double v = m.a[p][r];
        if (fabs(v) <= DBL_EPSILON * fmax(infinity_norm(m), DBL_MIN))
            break;
        double angle = .5 * atan2(2 * v, m.a[r][r] - m.a[p][p]), c = cos(angle), s = sin(angle);
        for (int k = 0; k < 3; k++) {
            double l = m.a[k][p], h = m.a[k][r];
            m.a[k][p] = c * l - s * h;
            m.a[k][r] = s * l + c * h;
        }
        for (int k = 0; k < 3; k++) {
            double l = m.a[p][k], h = m.a[r][k];
            m.a[p][k] = c * l - s * h;
            m.a[r][k] = s * l + c * h;
            l = q.a[k][p];
            h = q.a[k][r];
            q.a[k][p] = c * l - s * h;
            q.a[k][r] = s * l + c * h;
        }
        m.a[p][r] = m.a[r][p] = 0;
    }
    int rank = 0;
    out = {};
    nullspace = {};
    for (int k = 0; k < 3; k++) {
        V v = vec(q.a[0][k], q.a[1][k], q.a[2][k]);
        if (m.a[k][k] > tol) {
            outer(out, v, v, 1 / m.a[k][k]);
            rank++;
        } else
            outer(nullspace, v, v, 1);
    }
    return rank;
}
__device__ void swapd(double &a, double &b) {
    double c = a;
    a = b;
    b = c;
}
__device__ double kth(double *v, int n, int k) {
    int l = 0, r = n - 1;
    while (l < r) {
        double p = v[(l + r) / 2];
        int i = l, j = r;
        while (i <= j) {
            while (v[i] < p)
                i++;
            while (v[j] > p)
                j--;
            if (i <= j) {
                swapd(v[i], v[j]);
                i++;
                j--;
            }
        }
        if (k <= j)
            r = j;
        else if (k >= i)
            l = i;
        else
            break;
    }
    return v[k];
}
__device__ double median(double *v, int n) {
    double a = kth(v, n, n / 2);
    return n % 2 ? a : .5 * (a + kth(v, n, n / 2 - 1));
}
struct Obs {
    double ax, ay, dx, dy, rz;
};
struct Rig {
    double fx, fy, cx, cy;
};
__device__ V ray(int x, double y, Rig k) { return vec((x - k.cx) / k.fx, (y - k.cy) / k.fy, 1); }
__device__ double sample(const float *p, int w, int h, double x, double y) {
    x = fmin(w - 1., fmax(0., rint(x * 32) / 32));
    y = fmin(h - 1., fmax(0., rint(y * 32) / 32));
    int l = (int)x, t = (int)y, r = min(l + 1, w - 1), b = min(t + 1, h - 1);
    return (1 - y + t) * ((1 - x + l) * p[t * w + l] + (x - l) * p[t * w + r]) +
           (y - t) * ((1 - x + l) * p[b * w + l] + (x - l) * p[b * w + r]);
}
__global__ void correspond(const float *flow, const double *ks, const double *poses,
                           const bool *flip, const int *need, Obs *obs, int B, int H, int W) {
    int n = H * W, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= B * 2 * n)
        return;
    int b = i / (2 * n), f = i / n % 2, p = i % n;
    Obs o = {};
    if (!need[b]) {
        obs[i] = o;
        return;
    }
    int x = p % W, y = p / W, nx = flip[b] ? W - 1 - x : x, u = y * W + nx;
    const float *v = flow + (b * 8 + f * 4) * n;
    float dx = v[u], dy = v[n + u], qx = nx + dx, qy = y + dy;
    if (qx >= 0 && qx <= W - 1 && qy >= 0 && qy <= H - 1) {
        double rx = sample(v + 2 * n, W, H, qx, qy), ry = sample(v + 3 * n, W, H, qx, qy);
        if ((dx + rx) * (dx + rx) + (dy + ry) * (dy + ry) <
            .01 * (dx * dx + dy * dy + rx * rx + ry * ry) + .5) {
            Rig k = *((const Rig *)ks + b);
            V a = ray(x, y, k);
            const double *t = poses + (b * 2 + f) * 16;
            o.rz = t[8] * a.x + t[9] * a.y + t[10];
            o.dx = ((flip[b] ? W - 1 - qx : qx) - k.cx) / k.fx;
            o.dy = (qy - k.cy) / k.fy;
            o.ax = (t[0] * a.x + t[1] * a.y + t[2] - o.dx * o.rz) * k.fx;
            o.ay = (t[4] * a.x + t[5] * a.y + t[6] - o.dy * o.rz) * k.fy;
        }
    }
    obs[i] = o;
}
struct Ground {
    V beta, surface;
    Mat cov;
    double sigma;
};
struct Equation {
    V r;
    double jx, jy, bx, by, rz;
    V tz;
};
__global__ void fit_ground(const Obs *obs, const bool *road, const float *depth, const double *ks,
                           const double *poses, const int *need, double *scratch, Ground *result,
                           int H, int W, int stride) {
    int b = blockIdx.x, kind = blockIdx.y;
    if (threadIdx.x || !need[b])
        return;
    const int n = H * W;
    Rig k = ((const Rig *)ks)[b];
    Ground &g = result[b];
    double *mem = scratch + (b * 2 + kind) * stride;
    int capacity = ((H + 3) / 4) * ((W + 3) / 4) * 2;
    Equation *eq = (Equation *)mem;
    double *res = (double *)(eq + capacity), *weight = res + capacity * 2;
    int size = 0;
    for (int y = 0; y < H; y += 4)
        for (int x = 0; x < W; x += 4) {
            int p = y * W + x;
            if (!road[b * n + p])
                continue;
            V r = ray(x, y, k);
            if (kind) {
                double d = depth[b * n + p];
                if (d > 0 && isfinite(d))
                    eq[size++] = {r, 1, 0, 1 / d, 0, 1, {}};
            } else
                for (int f = 0; f < 2; f++) {
                    Obs o = obs[(b * 2 + f) * n + p];
                    if (!o.rz)
                        continue;
                    const double *T = poses + (b * 2 + f) * 16;
                    eq[size++] = {r,
                                  (T[3] - o.dx * T[11]) * k.fx,
                                  (T[7] - o.dy * T[11]) * k.fy,
                                  -o.ax,
                                  -o.ay,
                                  o.rz,
                                  mul(r, T[11])};
                }
        }
    if (size < 1)
        return;
    for (int i = 0; i < size; i++)
        weight[i] = 1;
    V last = {}, beta = {};
    Mat cov = {}, nullspace = {};
    double scale = 0;
    for (int it = 0; it < 76; it++) {
        Mat normal = {};
        V rhs = {};
        for (int i = 0; i < size; i++) {
            Equation e = eq[i];
            double w = weight[i];
            outer(normal, e.r, e.r, w * (e.jx * e.jx + e.jy * e.jy));
            rhs = plus(rhs, mul(e.r, w * (e.jx * e.bx + e.jy * e.by)));
        }
        if (inverse(normal, 3 * DBL_EPSILON * infinity_norm(normal), cov, nullspace) != 3)
            return;
        if (it == 75)
            break;
        beta = mv(cov, rhs);
        int count = 0;
        for (int i = 0; i < size; i++) {
            Equation e = eq[i];
            double z = dot(e.r, beta), den = fmax(fabs(e.rz + dot(e.tz, beta)), sqrt(DBL_EPSILON));
            res[count++] = (e.jx * z - e.bx) / den;
            if (!kind)
                res[count++] = (e.jy * z - e.by) / den;
        }
        double med = median(res, count);
        for (int i = 0; i < count; i++)
            res[i] = fabs(res[i] - med);
        scale = fmax(1.4826022185 * median(res, count), sqrt(DBL_EPSILON));
        for (int i = 0; i < size; i++) {
            Equation e = eq[i];
            double z = dot(e.r, beta), den = e.rz + dot(e.tz, beta);
            double a = e.jx * z - e.bx, c = e.jy * z - e.by;
            weight[i] = 1 / (scale * scale * fmax(den * den, DBL_EPSILON) + a * a + c * c);
        }
        if (it && norm(plus(beta, mul(last, -1))) <= 1e-5 * fmax(norm(beta), DBL_EPSILON))
            it = 74;
        last = beta;
    }
    if (kind)
        g.surface = beta;
    else {
        g.beta = beta;
        g.cov = cov;
        g.sigma = scale;
    }
}
struct Column {
    double anchor, prec, road, old, init, linear, z, step, var, diag, g, rank, shape, fit;
    V cross;
};
struct Observation {
    int j, index;
    double weight;
};
struct Block {
    double x, y, w;
    int first, last;
};
// Pool adjacent violations, including all ties, then interpolate on the abscissa.
__device__ void isotonic(Column *c, int m, int *idx, Block *p, double *ys, bool anchor_map,
                         bool old_anchor) {
    int n = 0;
    for (int j = 0; j < m; j++) {
        double x = anchor_map ? c[j].shape : c[j].linear;
        if (!isfinite(x) || (!anchor_map && !isfinite(c[j].z)))
            continue;
        double a = old_anchor ? c[j].old : c[j].road;
        if (anchor_map && !(a > 0))
            continue;
        int l = n++;
        while (l > 0 && (anchor_map ? c[idx[l - 1]].shape : c[idx[l - 1]].linear) > x) {
            idx[l] = idx[l - 1];
            l--;
        }
        idx[l] = j;
    }
    if (!n)
        return;
    int nb = 0;
    for (int i = 0; i < n;) {
        int start = i;
        double sum = 0;
        double xx = anchor_map ? c[idx[i]].shape : c[idx[i]].linear;
        do {
            Column &v = c[idx[i++]];
            sum += anchor_map ? -log(old_anchor ? v.old : v.road) : log(v.z);
        } while (i < n && (anchor_map ? c[idx[i]].shape : c[idx[i]].linear) == xx);
        p[nb++] = {xx, sum, double(i - start), start, i - 1};
        while (nb > 1 && p[nb - 2].y / p[nb - 2].w >= p[nb - 1].y / p[nb - 1].w) {
            p[nb - 2].y += p[nb - 1].y;
            p[nb - 2].w += p[nb - 1].w;
            p[nb - 2].last = p[nb - 1].last;
            nb--;
        }
    }
    for (int t = 0; t < nb; t++)
        for (int i = p[t].first; i <= p[t].last; i++)
            ys[i] = p[t].y / p[t].w;
    if (!anchor_map) {
        for (int i = 0; i < n; i++)
            c[idx[i]].shape = ys[i];
        return;
    }
    for (int j = 0; j < m; j++) {
        double x = c[j].shape;
        if (!isfinite(x) || c[j].old > 0)
            continue;
        int l = 0, r = n;
        while (l < r) {
            int mid = (l + r) / 2;
            if (c[idx[mid]].shape < x)
                l = mid + 1;
            else
                r = mid;
        }
        int hi = min(l, n - 1), lo = max(l - 1, 0);
        double d = c[idx[hi]].shape - c[idx[lo]].shape;
        double t = d > 0 ? fmax(0., fmin(1., (x - c[idx[lo]].shape) / d)) : 0;
        c[j].fit = exp(-(ys[lo] + t * (ys[hi] - ys[lo])));
    }
}
// The translation block has three unknowns; eliminate the independent depths exactly.
__device__ V normal_step(Column *c, int m, Observation *rows, int count, const Obs *flow, int area,
                         Rig k, V tau, double sigma, int mode) {
    bool pixel = mode == 3;
    Mat mat = {};
    V h = {};
    double precision_sum = 0, maxdiag = 0;
    for (int j = 0; j < m; j++) {
        double w = c[j].prec;
        if (mode >= 2) {
            double e = c[j].z - c[j].anchor;
            w = w / (1 + w * e * e);
        }
        double deriv = pixel ? c[j].z : 1;
        c[j].diag = w * deriv * deriv;
        c[j].g = pixel ? w * deriv * (c[j].z - c[j].anchor) : -w * c[j].anchor;
        c[j].cross = {};
    }
    for (int i = 0; i < count; i++) {
        Observation &row = rows[i];
        Column &v = c[row.j];
        Obs o = flow[row.index];
        double s = (row.index / area % 2) ? 1 : -1;
        double den = v.z * o.rz + s * tau.z;
        double ex = o.ax * v.z + s * k.fx * (tau.x - o.dx * tau.z),
               ey = o.ay * v.z + s * k.fy * (tau.y - o.dy * tau.z);
        double ax = o.ax, ay = o.ay, bx = s * k.fx, by = s * k.fy, cx = -bx * o.dx, cy = -by * o.dy;
        double w = row.weight;
        if (pixel) {
            ex /= den;
            ey /= den;
            ax = (ax - ex * o.rz) * v.z / den;
            ay = (ay - ey * o.rz) * v.z / den;
            bx /= den;
            by /= den;
            cx = (cx - s * ex) / den;
            cy = (cy - s * ey) / den;
            precision_sum +=
                (bx * bx + by * by + cx * cx + cy * cy) / (sigma * sigma + ex * ex + ey * ey);
            w = 1 / (sigma * sigma + ex * ex + ey * ey);
        } else if (mode == 2)
            w = w / (1 + w * (ex * ex + ey * ey));
        V tx = vec(bx, 0, cx), ty = vec(0, by, cy);
        outer(mat, tx, tx, w);
        outer(mat, ty, ty, w);
        v.diag += w * (ax * ax + ay * ay);
        v.cross = plus(v.cross, mul(plus(mul(tx, ax), mul(ty, ay)), w));
        if (pixel) {
            v.g += w * (ax * ex + ay * ey);
            h = plus(h, mul(plus(mul(tx, ex), mul(ty, ey)), w));
        }
    }
    Mat schur = mat, ct = {}, nullspace = {};
    V rhs = mul(h, -1);
    for (int j = 0; j < m; j++)
        maxdiag = fmax(c[j].diag, maxdiag);
    double tol = pixel ? DBL_EPSILON * (m + 3) * fmax(maxdiag, precision_sum) : 0;
    for (int j = 0; j < m; j++) {
        Column &v = c[j];
        double di = v.diag > tol ? 1 / v.diag : 0;
        V cross = mul(v.cross, di);
        outer(schur, v.cross, cross, -1);
        rhs = plus(rhs, mul(cross, v.g));
        v.diag = di;
        v.cross = cross;
    }
    int rank =
        inverse(schur, DBL_EPSILON * (m + 3) * fmax(DBL_MIN, infinity_norm(mat)), ct, nullspace);
    V t = mv(ct, rhs);
    for (int j = 0; j < m; j++) {
        Column &v = c[j];
        v.step = -v.diag * v.g - dot(v.cross, t);
        v.var = v.diag + dot(v.cross, mv(ct, v.cross));
        if (v.diag <= 0 || (rank < 3 && norm(mv(nullspace, v.cross)) >
                                            sqrt(DBL_EPSILON) * fmax(1., norm(v.cross))))
            v.var = INFINITY;
    }
    return t;
}
__device__ double cost(Column *c, int m, Observation *rows, int count, const Obs *flow, int area,
                       Rig k, V tau, double sigma, double step) {
    double total = 0;
    for (int j = 0; j < m; j++) {
        c[j].rank = c[j].z * exp(c[j].step * step);
        if (!isfinite(c[j].rank))
            return INFINITY;
        double e = c[j].rank - c[j].anchor;
        if (c[j].prec > 0)
            total += log1p(c[j].prec * e * e);
    }
    for (int i = 0; i < count; i++) {
        Observation a = rows[i];
        Obs o = flow[a.index];
        double z = c[a.j].rank, s = a.index / area % 2 ? 1 : -1;
        double den = z * o.rz + s * tau.z;
        if (!(den > 0))
            return INFINITY;
        double x = (o.ax * z + s * k.fx * (tau.x - o.dx * tau.z)) / den,
               y = (o.ay * z + s * k.fy * (tau.y - o.dy * tau.z)) / den;
        total += log1p((x * x + y * y) / (sigma * sigma));
    }
    return total;
}
__device__ void profiles(Column *c, int m, Observation *rows, int count, const Obs *flow, int area,
                         Rig k, double sigma) {
    int r = 0, l = -1;
    while (r < m && !c[r].prec)
        r++;
    for (int j = 0; j < m; j++) {
        if (c[j].prec) {
            l = j;
            r = j + 1;
            while (r < m && !c[r].prec)
                r++;
        }
        c[j].init = l < 0    ? c[r].anchor
                    : r == m ? c[l].anchor
                             : c[l].anchor + (c[r].anchor - c[l].anchor) * (j - l) / (r - l);
        c[j].z = c[j].init;
    }
    if (!count) {
        for (int j = 0; j < m; j++)
            c[j].linear = c[j].z = c[j].prec ? c[j].anchor : NAN;
        return;
    }
    V t = {};
    for (int i = 0; i < count; i++) {
        Observation &a = rows[i];
        double v = sigma * c[a.j].init * flow[a.index].rz;
        a.weight = 1 / fmax(DBL_MIN, v * v);
    }
    t = normal_step(c, m, rows, count, flow, area, k, t, sigma, 0);
    for (int i = 0; i < count; i++) {
        Observation &a = rows[i];
        double rz = flow[a.index].rz;
        double den = c[a.j].step * rz + (a.index / area % 2 ? 1 : -1) * t.z;
        double v = sigma * (den > 0 && isfinite(den) ? den : c[a.j].init * rz);
        a.weight = 1 / fmax(DBL_MIN, v * v);
    }
    t = normal_step(c, m, rows, count, flow, area, k, t, sigma, 1);
    for (int j = 0; j < m; j++)
        c[j].z = c[j].step;
    for (int it = 0; it < 75; it++) {
        V nt = normal_step(c, m, rows, count, flow, area, k, t, sigma, 2);
        V d = plus(nt, mul(t, -1));
        double diff = fmax(fabs(d.x), fmax(fabs(d.y), fabs(d.z)));
        double level = fmax(fabs(nt.x), fmax(fabs(nt.y), fabs(nt.z)));
        t = nt;
        for (int j = 0; j < m; j++) {
            diff = fmax(diff, fabs(c[j].step - c[j].z));
            c[j].z = c[j].step;
            level = fmax(level, fabs(c[j].z));
        }
        if (diff <= 1e-5 * fmax(level, DBL_EPSILON))
            break;
    }
    for (int j = 0; j < m; j++)
        c[j].linear = c[j].z > 0 && isfinite(c[j].var) && c[j].var > 0 ? (float)c[j].z : NAN;
    Mat mat = {}, ct = {}, nullspace = {};
    V rhs = {};
    for (int i = 0; i < count; i++) {
        Observation a = rows[i];
        Obs o = flow[a.index];
        double s = a.index / area % 2 ? 1 : -1;
        if (c[a.j].z * o.rz + s * t.z <= 0)
            c[a.j].linear = NAN;
        double z = c[a.j].init, den = z * o.rz;
        V tx = vec(s * k.fx / den, 0, -s * k.fx * o.dx / den),
          ty = vec(0, s * k.fy / den, -s * k.fy * o.dy / den);
        outer(mat, tx, tx, 1);
        outer(mat, ty, ty, 1);
        rhs = plus(rhs, plus(mul(tx, -o.ax * z / den), mul(ty, -o.ay * z / den)));
    }
    inverse(mat, DBL_EPSILON * DBL_EPSILON * 4. * count * count * infinity_norm(mat), ct,
            nullspace);
    t = mv(ct, rhs);
    for (int j = 0; j < m; j++) {
        c[j].z = c[j].init;
        c[j].step = 0;
    }
    for (int it = 0; it < 1024; it++) {
        bool good = true;
        for (int i = 0; i < count; i++) {
            Observation a = rows[i];
            if (c[a.j].z * flow[a.index].rz + (a.index / area % 2 ? 1 : -1) * t.z <= 0) {
                good = false;
                break;
            }
        }
        if (good)
            break;
        t = mul(t, .5);
    }
    double loss = cost(c, m, rows, count, flow, area, k, t, sigma, 0);
    for (int it = 0; it < 75 && isfinite(loss); it++) {
        V dt = normal_step(c, m, rows, count, flow, area, k, t, sigma, 3);
        double step = 1, next = INFINITY;
        while (step > DBL_EPSILON) {
            next = cost(c, m, rows, count, flow, area, k, plus(t, mul(dt, step)), sigma, step);
            if (isfinite(next) && next <= loss + DBL_EPSILON * fmax(loss, 1.))
                break;
            step *= .5;
        }
        if (step <= DBL_EPSILON)
            break;
        for (int j = 0; j < m; j++)
            c[j].z = c[j].rank;
        t = plus(t, mul(dt, step));
        double diff = fabs(next - loss);
        loss = next;
        if (diff <= 1e-5 * fmax(1., loss))
            break;
    }
    normal_step(c, m, rows, count, flow, area, k, t, sigma, 3);
    for (int j = 0; j < m; j++)
        if (!(c[j].z > 0 && c[j].var > 0 && isfinite(c[j].var) && isfinite(loss)))
            c[j].z = NAN;
}
__global__ void complete(const int *frame, const int *required, const int64_t *cells,
                         const bool *valid, const int64_t *offsets, const int64_t *grid,
                         const bool *contact_road, const double *ks, const Ground *ground,
                         const Obs *flow, Column *cols, Observation *samples, int *ints,
                         Block *blocks, double *values, float *target, bool *mask, int H, int W,
                         int L) {
    int i = blockIdx.x;
    if (threadIdx.x || !required[i])
        return;
    int b = frame[i], area = H * W;
    Rig k = ((const Rig *)ks)[b];
    Ground g = ground[b];
    if (!(g.sigma > 0))
        return;
    Column *c = cols + i * W;
    int *map = ints + i * 2 * W, *chain = map + W;
    Observation *rows = samples + 2 * offsets[i];
    int m = 0, count = 0;
    for (int x = 0; x < W; x++)
        map[x] = -1;
    for (int a = 0; a < L; a++) {
        if (!valid[i * L + a])
            continue;
        int p = cells[i * L + a];
        map[p % W] = max(map[p % W], p / W);
    }
    int first_contact = -1, last_contact = -1;
    for (int x = 0; x < W; x++) {
        if (map[x] < 0)
            continue;
        int j = m++;
        c[j] = {};
        c[j].shape = NAN;
        int bottom = map[x];
        map[x] = j;

        // The road must begin at the silhouette. A connected road run can
        // provide flow when the boundary pixel itself is unreliable.
        for (int y = bottom + 1; y < H; y++) {
            int p = y * W + x, q = b * area + p;
            if (!contact_road[q] || grid[q] >= 0)
                break;
            if (!(flow[(b * 2) * area + p].rz || flow[(b * 2 + 1) * area + p].rz))
                continue;
            V a = ray(x, bottom + .5, k);
            double u = dot(a, g.beta);
            double v = dot(a, mv(g.cov, a)) + pow(g.beta.y / k.fy, 2) / 12;
            double surface = dot(a, g.surface);
            if (u > 0 && v > 0 && surface > 0) {
                c[j].anchor = 1 / u;
                c[j].prec = u * u * u * u / v;
                c[j].road = surface;
                if (first_contact < 0)
                    first_contact = j;
                last_contact = j;
            }
            break;
        }
    }
    if (first_contact < 0)
        return;
    for (int a = 0; a < L; a++)
        if (valid[i * L + a]) {
            int p = cells[i * L + a], j = map[p % W];
            if (mask[b * area + p])
                c[j].old = target[b * area + p];
            for (int f = 0; f < 2; f++) {
                int q = (b * 2 + f) * area + p;
                if (flow[q].rz)
                    rows[count++] = {j, q, 0};
            }
        }
    profiles(c, m, rows, count, flow, area, k, g.sigma);
    isotonic(c, m, chain, blocks + i * W, values + i * W, false, false);
    bool has_old = false;
    for (int j = 0; j < m; j++)
        if (isfinite(c[j].shape) && c[j].old > 0)
            has_old = true;
    isotonic(c, m, chain, blocks + i * W, values + i * W, true, has_old);
    for (int a = 0; a < L; a++)
        if (valid[i * L + a]) {
            int p = cells[i * L + a], q = b * area + p;
            int j = map[p % W];
            float u = c[j].fit;
            // Interpolate between supported columns without extending past them.
            if (!mask[q] && j >= first_contact && j <= last_contact && isfinite(u) && u > 0) {
                target[q] = u;
                mask[q] = true;
            }
        }
}
extern "C" int gcidc_complete(void **p, int B, int M, int H, int W, int L, int stride,
                              void *stream_ptr) {
    cudaStream_t s = (cudaStream_t)stream_ptr;
    int n = B * 2 * H * W;
    correspond<<<(n + 255) / 256, 256, 0, s>>>((float *)p[0], (double *)p[1], (double *)p[2],
                                               (bool *)p[3], (int *)p[4], (Obs *)p[5], B, H, W);
    fit_ground<<<dim3(B, 2), 1, 0, s>>>((Obs *)p[5], (bool *)p[6], (float *)p[7], (double *)p[1],
                                        (double *)p[2], (int *)p[4], (double *)p[8], (Ground *)p[9],
                                        H, W, stride);
    complete<<<M, 1, 0, s>>>((int *)p[10], (int *)p[11], (int64_t *)p[12], (bool *)p[13],
                             (int64_t *)p[14], (int64_t *)p[15], (bool *)p[23], (double *)p[1],
                             (Ground *)p[9], (Obs *)p[5], (Column *)p[16], (Observation *)p[17],
                             (int *)p[18], (Block *)p[19], (double *)p[20], (float *)p[21],
                             (bool *)p[22], H, W, L);
    return cudaGetLastError();
}
extern "C" void gcidc_sizes(int *s) {
    s[0] = sizeof(Obs);
    s[1] = sizeof(Equation);
    s[2] = sizeof(Ground);
    s[3] = sizeof(Column);
    s[4] = sizeof(Observation);
    s[5] = sizeof(Block);
}
