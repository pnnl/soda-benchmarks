"""twomm without its elementwise ops: D = (A @ B) @ C.

Not an upstream PolyBench kernel: two matmuls in series, the case of one tile
per operator (docs/ESPTilePerOperatorPlan.md). Same dimensions and data as
twomm; --ni/--nj/--nk/--nl override them (all equal: both matmuls have one
shape).
"""
import torch
import torch.nn as nn

from benches.PolyBenchPyTorch.linear_algebra.kernels.twomm.twomm import (
    init_array as _twomm_init_array,
)
from benches.PolyBenchPyTorch.linear_algebra.utils import (
    generate_mlir, get_dataset_dimensions as _dimensions, make_parser,
    resolve_dtype,
)


def get_dataset_dimensions(dataset: str) -> dict:
    return _dimensions("twomm_mm", dataset)


class TwoMMOnlyMatmul(nn.Module):
    def __init__(self, ni: int, nj: int, nk: int, nl: int):
        super().__init__()
        self.ni, self.nj, self.nk, self.nl = ni, nj, nk, nl

    def forward(self, A, B, C):
        return torch.matmul(torch.matmul(A, B), C)


def init_array(ni, nj, nk, nl, dtype=torch.float32):
    """twomm's A (ni x nk), B (nk x nj) and C (nj x nl)."""
    return _twomm_init_array(ni, nj, nk, nl, dtype)[2:5]


def main():
    parser = make_parser("twomm_mm", "./output/twomm_mm.mlir")
    for d in ("ni", "nj", "nk", "nl"):
        parser.add_argument(f"--{d}", type=int)
    args = parser.parse_args()
    dims = get_dataset_dimensions(args.dataset)
    dims.update({d: getattr(args, d) for d in dims if getattr(args, d) is not None})
    generate_mlir(TwoMMOnlyMatmul(**dims),
                  init_array(**dims, dtype=resolve_dtype(args.dtype)),
                  args.out_mlir_path, args.dialect)


if __name__ == "__main__":
    main()
