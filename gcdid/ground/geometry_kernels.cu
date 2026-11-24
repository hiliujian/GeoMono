// Column-wise lower-contour envelopes on CUDA.
#include <cuda_runtime.h>

__global__ void contours(const int *bottom, int width, int *stack, double *hull, int *facet) {
    int id = blockIdx.x;
    bottom += id * width;
    stack += id * width;
    hull += id * width;
    facet += id * width;
    int n = 0;
    for (int x = 0; x < width; x++)
        if (bottom[x] >= 0) {
            while (n > 1) {
                int a = stack[n - 2], b = stack[n - 1];
                if ((b - a) * (bottom[x] - bottom[a]) < (bottom[b] - bottom[a]) * (x - a))
                    break;
                n--;
            }
            stack[n++] = x;
        }
    for (int x = 0, j = 0; x < width; x++) {
        if (!n) {
            hull[x] = 0;
            facet[x] = 0;
            continue;
        }
        while (j + 1 < n - 1 && stack[j + 1] <= x)
            j++;
        int a = stack[j], b = stack[min(j + 1, n - 1)];
        // The exterior continuation of the lower silhouette is horizontal.
        int last = stack[n - 1];
        bool exterior = n > 1 && last < width - 1 && x >= last
                        && bottom[last] != bottom[stack[n - 2]];
        facet[x] = exterior ? n - 1 : j;
        hull[x] = (x <= a || a == b) ? bottom[a]
                  : x >= b           ? bottom[b]
                                     : bottom[a] + ((double)(bottom[b] - bottom[a]) / (b - a)) * (x - a);
    }
}
extern "C" int envelopes(const void *bottom, int n, int width, void *stack, void *hull, void *facet,
                         cudaStream_t stream) {
    contours<<<n, 1, 0, stream>>>((const int *)bottom, width, (int *)stack, (double *)hull, (int *)facet);
    return cudaGetLastError();
}
