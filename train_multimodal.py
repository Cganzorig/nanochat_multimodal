import os
import sys
import importlib.util
import torch
from torch.utils.data import DataLoader
from model_multimodal import MultiModalModel
from dataset import COCOMultiModalDataset

# -------------------------------------------------
# Load config dynamically from path
# -------------------------------------------------
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config


# -------------------------------------------------
# Training loop
# -------------------------------------------------
def train(config_path="config/train_coco_subset.py"):
    # --- Load config ---
    config = load_config(config_path)
    os.makedirs("out/checkpoints", exist_ok=True)

    print(f"⚙️ Loaded config from: {config_path}")
    print(f"📦 Dataset: {config.image_dir}")
    print(f"🧠 Max iters: {getattr(config, 'max_iters', 10000)}")
    print(f"🚀 Batch size: {config.batch_size}")

    # --- Initialize dataset and dataloader ---
    dataset = COCOMultiModalDataset(
        annotations_file=config.annotations_file,
        image_dir=config.image_dir,
        text_model_name="distilgpt2",
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # --- Initialize model ---
    model = MultiModalModel("distilgpt2").cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    max_iters = getattr(config, "max_iters", 10000)
    save_interval = getattr(config, "save_interval", 500)
    step = 0

    model.train()
    print("🚀 Starting training...")

    for batch in dataloader:
        if step >= max_iters:
            print(f"✅ Reached max_iters = {max_iters}. Stopping training.")
            break

        # Move data to GPU
        for k in batch:
            batch[k] = batch[k].cuda()

        # Forward / backward
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Log progress
        if step % 50 == 0:
            print(f"Step {step:05d}: loss = {loss.item():.4f}")

        # Save checkpoint
        if step > 0 and step % save_interval == 0:
            save_path = f"out/checkpoints/step_{step}.pt"
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss.item(),
                },
                save_path,
            )
            print(f"💾 Saved checkpoint: {save_path}")

        step += 1

    # --- Save final model ---
    final_path = "out/checkpoints/final_model.pt"
    torch.save(model.state_dict(), final_path)
    print(f"✅ Training finished. Final model saved to {final_path}")


# -------------------------------------------------
# Entry point
# -------------------------------------------------
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/train_coco_subset.py"
    train(config_path)
