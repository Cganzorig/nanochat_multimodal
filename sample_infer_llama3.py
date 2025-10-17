import torch
from PIL import Image
import matplotlib.pyplot as plt
from model_multimodal_infer import MultiModalLlamaInfer

# === Settings ===
image_path = "data/images/val2017/000000581317.jpg"
prompt = "A photo of"

# === Load model ===
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiModalLlamaInfer().to(device)
model.eval()

# === Load image ===
image = Image.open(image_path).convert("RGB")

# === Generate caption ===
print("🧠 Generating caption...")
caption = model.generate_caption(image, prompt=prompt)
print("\n📝 Caption:", caption)

# === Show ===
plt.figure(figsize=(6,6))
plt.imshow(image)
plt.axis("off")
plt.title(caption, fontsize=12, wrap=True)
plt.show()
