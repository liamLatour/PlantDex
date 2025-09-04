import torch
import os
import sys
from torchvision.models import mobilenet_v3_small

def load_model(model, filename, use_gpu):
    if not os.path.exists(filename):
        raise FileNotFoundError

    device = 'cuda:0' if use_gpu else 'cpu'
    d = torch.load(filename, map_location=device, weights_only=False)
    # If loading a .pt file, it may be a state_dict or a full model
    if isinstance(d, dict) and 'model' in d:
        model.load_state_dict(d['model'])
    elif isinstance(d, dict):
        model.load_state_dict(d)
    else:
        model = d  # If it's a full model object
    return model

def main():
    # Usage: python convert_to_onnx.py [input_pt_filename] [output_onnx_filename]
    pt_filename = "model.pt"
    onnx_filename = "model.onnx"
    if len(sys.argv) > 1:
        pt_filename = sys.argv[1]
    if len(sys.argv) > 2:
        onnx_filename = sys.argv[2]

    use_gpu = False  # load weights on the gpu
    model = mobilenet_v3_small(num_classes=1081) # 1081 classes in Pl@ntNet-300K

    model = load_model(model, filename=pt_filename, use_gpu=use_gpu)
    model.eval()

    example_inputs = (torch.randn(1, 3, 224, 224),)

    torch.onnx.export(
        model,
        example_inputs,
        onnx_filename,
        opset_version=9,
        input_names=["x"],
        output_names=["output"],
    )
    print(f"ONNX model saved to {onnx_filename}")

if __name__ == "__main__":
    main()