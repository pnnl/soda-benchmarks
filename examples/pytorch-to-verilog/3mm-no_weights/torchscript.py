import torch
from torch_mlir import torchscript
import os
import argparse
import torch.nn.functional as F

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers/cache/'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to MLIR.")
    parser.add_argument("out_mlir_path", nargs="?", default="./output/01_tosa.mlir", help="Path to write the MLIR file to.")
    dialect_choices = ["tosa", "linalg-on-tensors", "torch", "raw", "mhlo"]
    parser.add_argument("--dialect", default="linalg-on-tensors", choices=dialect_choices, help="Dialect to use for lowering.")
    
    args = parser.parse_args()
    return args

class ThreeMM(torch.nn.Module):
    def __init__(self):
        super(ThreeMM, self).__init__()
        # Stateless model: no weights.
    
    def forward(self, input1, input2, input3, input4):
        # First multiplication: input1 @ input2
        out1 = torch.matmul(input1, input2)
        # Second multiplication: out1 @ input3
        out2 = torch.matmul(out1, input3)
        # Third multiplication: out2 @ input4
        out3 = torch.matmul(out2, input4)
        return out3

def main():
    args = parse_args()

    # Create a model
    model = ThreeMM()
    print(model)

    # Prepare directory and input data with four inputs.
    os.makedirs(os.path.dirname(args.out_mlir_path), exist_ok=True)
    # Define input dimensions for chain matrix multiplication:
    # input1: [bs, M, K], input2: [bs, K, L], input3: [bs, L, P], input4: [bs, P, N]
    input_dims = {
        'bs': 1,
        'M': 8,
        'K': 16,
        'L': 10,
        'P': 6,
        'N': 12
    }

    input1 = torch.randn(input_dims['bs'], input_dims['M'], input_dims['K'])
    input2 = torch.randn(input_dims['bs'], input_dims['K'], input_dims['L'])
    input3 = torch.randn(input_dims['bs'], input_dims['L'], input_dims['P'])
    input4 = torch.randn(input_dims['bs'], input_dims['P'], input_dims['N'])

    # Generate the MLIR module with four inputs wrapped as a tuple.
    module = torchscript.compile(model, (input1, input2, input3, input4), output_type=args.dialect, use_tracing=True)
    with open(args.out_mlir_path, "w", encoding="utf-8") as outf:
        outf.write(str(module))
    
if __name__ == "__main__":
    main()
