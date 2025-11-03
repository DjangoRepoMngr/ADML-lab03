
# ADML Lab 3 – Project Skeleton (Lab1+Lab2 code split)

This repo is a clean split of your Lab1 (visualization) and Lab2 (training) into
folders like in the slides.

## Layout
- `data/` – transforms, datasets, loaders
- `models/` – `CustomNet`
- `utils/` – visualization and optional W&B helper
- `checkpoints/` – saved models
- `dataset/` – put Tiny-ImageNet-200 here (kept out of git)

## Setup
```bash
pip install -r requirements.txt



Visualization (Lab1)
python utils/viz.py --data_dir dataset/tiny-imagenet-200


Training (Lab2)
python train.py --data_dir dataset/tiny-imagenet-200 \
  --batch_size 64 --epochs 10 --lr 0.001 --momentum 0.9 --num_workers 2
Train: python train.py --data_dir dataset/tiny-imagenet-200 --use_wandb (optional).


With Weights & Biases (optional)
python train.py --data_dir dataset/tiny-imagenet-200 --use_wandb

EVAL:
python eval.py --data_dir dataset/tiny-imagenet-200 --checkpoint checkpoints/best_model.pth