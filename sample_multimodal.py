import torch
from PIL import Image
import matplotlib.pyplot as plt
import importlib
from transformers import AutoImageProcessor

# === Settings ===
ckpt_path = "out/checkpoints/final_model.pt"
image_path = "data/images/val2017/000000581317.jpg"
prompt = "A photo of"

# === Load model dynamically ===
print("🚀 Loading model checkpoint...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Try both model classes
try:
    model_module = importlib.import_module("model_multimodal_llama")
    model = model_module.MultiModalLlama().to(device)
    model_type = "llama"
except (ModuleNotFoundError, AttributeError):
    from model_multimodal import MultiModalModel
    model = MultiModalModel("distilgpt2").to(device)
    model_type = "gpt2"

# Load weights
state_dict = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state_dict, strict=False)
model.eval()

# === Tokenizer & processor setup ===
tokenizer = model.text_tokenizer
if hasattr(model, "image_processor"):
    processor = model.image_processor
else:
    # fallback for SigLIP or any model without explicit processor
    processor = AutoImageProcessor.from_pretrained(
        getattr(model, "vision_model_name", "google/siglip-so400m-patch14-384")
    )

# === Load and preprocess image ===
print(f"📸 Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to(device)
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

# === Build embeddings ===
print("🧠 Preparing multimodal embeddings...")
with torch.no_grad():
    # Vision
    vision_outputs = model.vision_encoder(pixel_values)
    vision_embeds = model.visual_proj(vision_outputs.last_hidden_state)

    # Text embeddings (switch by model type)
    if model_type == "llama":
        text_embeds = model.text_model.model.embed_tokens(input_ids)
    else:  # GPT-2 / DistilGPT-2
        text_embeds = model.text_model.transformer.wte(input_ids)

    # Concatenate embeddings
    inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

    # Attention masks
    img_mask = torch.ones(vision_embeds.size()[:-1], device=device)
    txt_mask = torch.ones(text_embeds.size()[:-1], device=device)
    attention_mask = torch.cat([img_mask, txt_mask], dim=1)

    # === Generate text ===
    print("💬 Generating caption...")
    outputs = model.text_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=64,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
    )

# === Decode and display ===
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n📝 Generated caption:")
print(caption)

plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.axis("off")
plt.title(caption, fontsize=12, wrap=True)
plt.tight_layout()
plt.show()
