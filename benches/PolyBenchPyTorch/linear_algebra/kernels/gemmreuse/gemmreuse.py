"""Two dependent GEMMs with different shapes, using the PolyBench interface.

This is an additional validation workload, not an upstream PolyBench kernel.
Both H = A @ B and Y = H @ W are returned so the intermediate is checked.
"""
import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir, get_dataset_dimensions as _dimensions, make_parser,
    resolve_dtype,
)


def get_dataset_dimensions(dataset: str) -> dict:
    return _dimensions("gemmreuse", dataset)


class GemmReuse(nn.Module):
    def __init__(self, ni: int, nj: int, nk: int, nl: int):
        super().__init__()
        self.ni, self.nj, self.nk, self.nl = ni, nj, nk, nl

    def forward(self, A, B, W):
        H = torch.matmul(A, B)
        Y = torch.matmul(H, W)
        return H, Y


def init_array(ni, nj, nk, nl, dtype=torch.float32):
    def matrix(rows, cols, seed):
        i = torch.arange(rows).reshape(-1, 1)
        j = torch.arange(cols).reshape(1, -1)
        return (((i * 7 + j * 3 + seed) % 17) - 8).to(dtype) / 32
    return matrix(ni, nk, 1), matrix(nk, nj, 5), matrix(nj, nl, 11)


def main():
    parser = make_parser("gemmreuse", "./output/gemmreuse.mlir")
    args = parser.parse_args()
    dims = get_dataset_dimensions(args.dataset)
    generate_mlir(GemmReuse(**dims),
                  init_array(**dims, dtype=resolve_dtype(args.dtype)),
                  args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
