
import onnxruntime as ort
from PIL import Image
import numpy as np
import sys




import numpy as np
from PIL import Image

# Normalization constants (same as in Java)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_SIZE = 518  # corresponds to the 'i' in Java code


def center_crop(image: Image.Image) -> Image.Image:
    """Center-crop a PIL image to a square."""
    width, height = image.size
    if width == height:
        return image
    if width > height:
        left = (width - height) // 2
        top = 0
        right = left + height
        bottom = height
    else:
        left = 0
        top = (height - width) // 2
        right = width
        bottom = top + width
    return image.crop((left, top, right, bottom))


def preprocess_image(image_path: str, target_size: int = TARGET_SIZE) -> np.ndarray:
    """
    Load an image, center-crop, resize, normalize, and convert to float32
    in CHW format (C, H, W) for ONNX Runtime.
    """
    img = Image.open(image_path).convert("RGB")
    img = center_crop(img)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0  # H x W x C
    # Normalize
    img_array = (img_array - MEAN) / STD
    # Convert to C x H x W
    img_array = np.transpose(img_array, (2, 0, 1))
    # Add batch dimension: 1 x C x H x W
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def main():
	if len(sys.argv) < 3:
		print("Usage: python test_onnx.py <onnx_model_path> <image_path>")
		return
	model_path = sys.argv[1]
	image_path = sys.argv[2]

	# Preprocess image
	input_tensor = preprocess_image(image_path)

	# Load ONNX model
	session = ort.InferenceSession(model_path)
	input_name = session.get_inputs()[0].name
	outputs = session.run(None, {input_name: input_tensor})
	print(f"Predicted class index: {outputs[1][0][0]}")

if __name__ == "__main__":
	main()
