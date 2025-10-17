# ============================
# Config for SigLIP + LLaMA
# ============================

# Dataset
image_dir = "data/images/val2017"
annotations_file = "data/annotations/captions_val2017.json"

# Models
vision_model_name = "google/siglip-so400m-patch14-384"
text_model_name = "meta-llama/Llama-2-7b-hf"

# Training parameters
batch_size = 1
learning_rate = 1e-5
max_iters = 10000
save_interval = 200
precision = "bf16"

# Optional LoRA fine-tuning flag (if used later)
use_lora = False
