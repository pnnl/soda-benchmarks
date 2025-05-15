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

class SimpleMLP(torch.nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Removed weight layers for a stateless computation graph.
    
    def forward(self, input1, input2):
        # Matrix multiplication of two inputs
        mm = torch.matmul(input1, input2)

        # Reshape tensor to 4D for pooling: [N, C, H, W]
        mm_unsq = mm.unsqueeze(1)
        # Max pooling with 2x2 kernel
        pooled = F.max_pool2d(mm_unsq, kernel_size=2, stride=2)
        # Apply ReLU activation
        relu_out = torch.relu(pooled)
        # Restore original number of dimensions
        return relu_out.squeeze(1)


def main():
    args = parse_args()

    # Create a model
    model = SimpleMLP()
    print(model)

    # Prepare directory and input data with two inputs.
    os.makedirs(os.path.dirname(args.out_mlir_path), exist_ok=True)
    # Define input dimensions
    input_dims = {
        'bs': 1,
        'M': 8,
        'K': 16,
        'N': 12
    }

    # For valid matmul, input1: [bs, M, K] and input2: [bs, K, N]
    input1 = torch.randn(input_dims['bs'], input_dims['M'], input_dims['K'])
    input2 = torch.randn(input_dims['bs'], input_dims['K'], input_dims['N'])

    # Generate the MLIR module with two inputs wrapped as a tuple.
    module = torchscript.compile(model, (input1, input2), output_type=args.dialect, use_tracing=True)
    with open(args.out_mlir_path, "w", encoding="utf-8") as outf:
        outf.write(str(module))
    
if __name__ == "__main__":
    main()
