import torch
from PIL import Image
import matplotlib.pyplot as plt
from model_multimodal import MultiModalModel

# === Settings ===
ckpt_path = "out/checkpoints/final_model.pt"
image_path = "data/images/val2017/000000581317.jpg"
prompt = "A photo of"

# === Load model ===
print("🚀 Loading model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MultiModalModel("distilgpt2").to(device)
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

tokenizer = model.text_tokenizer
processor = model.image_processor

# === Load and preprocess image ===
print("📸 Loading image:", image_path)
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# === Generate caption ===
print("🧠 Generating caption...")
with torch.no_grad():
    vision_outputs = model.vision_encoder(pixel_values)
    vision_embeds = model.visual_proj(vision_outputs.last_hidden_state)
    text_embeds = model.text_model.transformer.wte(input_ids)
    inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

    img_mask = torch.ones(vision_embeds.size()[:-1], device=device)
    txt_mask = torch.ones(text_embeds.size()[:-1], device=device)
    attention_mask = torch.cat([img_mask, txt_mask], dim=1)

    outputs = model.text_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.8,
    )

caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📝 Generated caption:")
print(caption)

# === Display image and caption ===
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(caption, fontsize=12, wrap=True)
plt.tight_layout()
plt.show()
