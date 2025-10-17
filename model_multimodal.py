import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer

class MultiModalModel(nn.Module):
    def __init__(self, text_model_name="distilgpt2"):
        super().__init__()
        # --- Text model ---
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model_name)

        # --- Vision encoder (frozen CLIP) ---
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # --- Projection layer ---
        self.visual_proj = nn.Linear(self.vision_encoder.config.hidden_size,
                                     self.text_model.config.hidden_size)
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None, labels=None):
        # --- multimodal case ---
        if pixel_values is not None:
            # Encode image
            vision_outputs = self.vision_encoder(pixel_values)
            vision_embeds = self.visual_proj(vision_outputs.last_hidden_state)

            # Encode text
            text_embeds = self.text_model.transformer.wte(input_ids)
            inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)

            # Build attention mask
            B = input_ids.size(0)
            img_seq_len = vision_embeds.size(1)
            txt_seq_len = text_embeds.size(1)
            if attention_mask is None:
                attention_mask = torch.ones((B, txt_seq_len), device=inputs_embeds.device)
            img_mask = torch.ones((B, img_seq_len), device=inputs_embeds.device)
            attention_mask = torch.cat([img_mask, attention_mask], dim=1)

            # 🩹 Ensure labels match total length
            if labels is not None:
                # pad labels for image tokens
                pad_labels = torch.full((B, img_seq_len), -100, dtype=labels.dtype, device=labels.device)
                labels = torch.cat([pad_labels, labels], dim=1)
                assert labels.size(1) == inputs_embeds.size(1), (
                    f"Label length {labels.size(1)} != input length {inputs_embeds.size(1)}"
                )

            outputs = self.text_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        # --- text-only fallback ---
        else:
            outputs = self.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        return outputs