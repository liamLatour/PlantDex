import torch

# Path to your .pt file
pt_file = "model.pt"

# Load the checkpoint
try:
    checkpoint = torch.load(pt_file, map_location="cpu")
except Exception as e:
    print("Failed to load the .pt file:", e)
    exit(1)

# If it's a dictionary, it might be a state_dict
if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
else:
    print("Loaded object is not a dict. It may be a full scripted/traced model.")
    state_dict = None

if state_dict:
    print(f"Total layers found: {len(state_dict)}\n")
    for k, v in state_dict.items():
        print(f"{k:60} | shape: {tuple(v.shape)}")

    print("\n=== Heuristic suggestions ===\n")

    # Check for Vision Transformer patterns
    vt_keys = [k for k in state_dict.keys() if "patch_embed" in k or "blocks" in k or "cls_token" in k or "pos_embed" in k]
    if vt_keys:
        print(f"Found {len(vt_keys)} keys that match ViT-style architecture:")
        for k in vt_keys:
            print("  ", k)
        print("\nSuggestion: This checkpoint likely comes from a Vision Transformer (ViT) or a derivative model like DeiT, Swin, or similar.\n")
    else:
        print("No clear ViT-style keys found. Could be a custom CNN or other architecture.")

else:
    print("This is likely a TorchScript model (already scripted/traced).")
    print("You can try loading it as:")
    print("import torch")
    print("model = torch.jit.load('model.pt')")
