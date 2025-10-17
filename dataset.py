import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor, AutoTokenizer

class COCOMultiModalDataset(Dataset):
    def __init__(self, annotations_file, image_dir, text_model_name="distilgpt2"):
        with open(annotations_file, 'r') as f:
            self.captions = json.load(f)['annotations']
        self.image_dir = image_dir
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def __getitem__(self, idx):
        caption = self.captions[idx]['caption']
        image_id = self.captions[idx]['image_id']
        image_path = f"{self.image_dir}/{image_id:012d}.jpg"
        image = Image.open(image_path).convert("RGB")

        text_inputs = self.tokenizer(caption, return_tensors='pt', truncation=True, padding='max_length', max_length=64)
        image_inputs = self.processor(images=image, return_tensors='pt')

        return {
            "input_ids": text_inputs['input_ids'].squeeze(0),
            "attention_mask": text_inputs['attention_mask'].squeeze(0),
            "pixel_values": image_inputs['pixel_values'].squeeze(0),
            "labels": text_inputs['input_ids'].squeeze(0)
        }

    def __len__(self):
        return len(self.captions)
