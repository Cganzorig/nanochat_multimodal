import torch
import torch.nn as nn
from transformers import SiglipVisionModel, AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor

class MultiModalLlamaInfer(nn.Module):
    def __init__(self,
                 vision_model_name="google/siglip-so400m-patch14-384",
                 text_model_name="meta-llama/Meta-Llama-3-8B-Instruct"):
        super().__init__()

        # --- Vision encoder (SigLIP) ---
        print(f"📷 Loading vision encoder: {vision_model_name}")
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_name)
        self.image_processor = AutoImageProcessor.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size  # 1024

        # --- Text model (LLaMA 3) ---
        print(f"💬 Loading language model: {text_model_name}")
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_dim = self.text_model.config.hidden_size  # 4096

        # --- Projection ---
        self.visual_proj = nn.Linear(vision_dim, text_dim)
        print("✅ Model initialized for inference.")

    @torch.no_grad()
    def generate_caption(self, image, prompt="Describe this image:", max_new_tokens=64):
        device = next(self.parameters()).device
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        input_ids = self.text_tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        # Vision encoding
        vision_outputs = self.vision_encoder(pixel_values)
        vision_embeds = self.visual_proj(vision_outputs.last_hidden_state)

        # Text embedding
        text_embeds = self.text_model.model.embed_tokens(input_ids)

        # Fuse
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
        img_mask = torch.ones(vision_embeds.size()[:-1], device=device)
        txt_mask = torch.ones(text_embeds.size()[:-1], device=device)
        attention_mask = torch.cat([img_mask, txt_mask], dim=1)

        outputs = self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )

        caption = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
