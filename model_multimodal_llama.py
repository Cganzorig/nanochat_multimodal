import torch
import torch.nn as nn
from transformers import SiglipVisionModel, AutoModelForCausalLM, AutoTokenizer

class MultiModalLlama(nn.Module):
    def __init__(self,
                 vision_model_name="google/siglip-so400m-patch14-384",
                 text_model_name="meta-llama/Llama-2-7b-hf"):
        super().__init__()

        # --- Vision encoder (SigLIP) ---
        self.vision_encoder = SiglipVisionModel.from_pretrained(vision_model_name)
        vision_dim = self.vision_encoder.config.hidden_size  # 1024

        # --- Text model (LLaMA) ---
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        text_dim = self.text_model.config.hidden_size  # 4096

        # --- Projection ---
        self.visual_proj = nn.Linear(vision_dim, text_dim)

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        # Vision
        vision_outputs = self.vision_encoder(pixel_values)
        vision_embeds = self.visual_proj(vision_outputs.last_hidden_state)  # (B, ~730, 4096)

        # Text
        text_embeds = self.text_model.model.embed_tokens(input_ids)         # (B, L_t, 4096)

        # Fuse
        inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

        # Masks
        B, img_seq_len = vision_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones((B, text_embeds.size(1)), device=inputs_embeds.device)
        img_mask = torch.ones((B, img_seq_len), device=inputs_embeds.device)
        attention_mask = torch.cat([img_mask, attention_mask], dim=1)

        # Labels padding
        if labels is not None:
            pad_labels = torch.full((B, img_seq_len), -100, dtype=labels.dtype, device=labels.device)
            labels = torch.cat([pad_labels, labels], dim=1)

        return self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
