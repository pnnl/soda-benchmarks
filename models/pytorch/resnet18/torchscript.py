import torch
from torch_mlir import torchscript
import os
import argparse
import torchvision.models as models

os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers/cache/'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a PyTorch model to MLIR.")
    parser.add_argument("out_mlir_path", nargs="?", default="./output/01_tosa.mlir", help="Path to write the MLIR file to.")
    dialect_choices = ["tosa", "linalg-on-tensors", "torch", "raw", "mhlo"]
    parser.add_argument("--dialect", default="linalg-on-tensors", choices=dialect_choices, help="Dialect to use for lowering.")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Create a model (downloaded resnet18 from torchvision)
    model = models.resnet18(weights=models.resnet.ResNet18_Weights.DEFAULT)
    model.train(False)
    print(model)

    # Prepare directory and input data: input shape updated for resnet18 (3 channels, 224x224)
    os.makedirs(os.path.dirname(args.out_mlir_path), exist_ok=True)
    in_shape = {'bs': 4, 'c': 3, 'h': 224, 'w': 224}
    input_data = torch.randn(in_shape['bs'], in_shape['c'], in_shape['h'], in_shape['w'])

    # Generate the MLIR module
    module = torchscript.compile(model, input_data, output_type=args.dialect, use_tracing=True)
    with open(args.out_mlir_path, "w", encoding="utf-8") as outf:
        outf.write(str(module))
    
if __name__ == "__main__":
    main()
