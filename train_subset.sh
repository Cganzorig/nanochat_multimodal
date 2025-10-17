#!/bin/bash
set -e

# === Train multimodal NanoChat on a small COCO subset ===
# (Assumes data is already available in data/coco_subset)

echo "⚙️ Generating config/train_coco_subset.py ..."
cat > config/train_coco_subset.py << 'EOF'
image_dir = 'data/images/val2017'
annotations_file = 'data/annotations/captions_val2017.json'
batch_size = 2
n_layer = 6
n_head = 6
n_embd = 384
max_iters = 10000
learning_rate = 1e-5
EOF

echo "🚀 Starting training on small subset..."
python train_multimodal.py config/train_coco_subset.py
