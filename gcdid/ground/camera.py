import torch


NORMALIZED_K = ((0.58, 0, 0.5, 0),
                (0, 1.92, 0.5, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1))


def intrinsics(height, width, device):
    matrix = torch.tensor(NORMALIZED_K, device=device, dtype=torch.float32)[:3, :3]
    matrix[0] *= width
    matrix[1] *= height
    return matrix.double()
