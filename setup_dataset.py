"""
Dataset Setup Helper
Generates a synthetic dataset of placeholder images to test the pipeline
before you have real fundus images. Replace with real data before training.
"""

import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
import random
import math

CLASSES = [
    "diabetic_retinopathy",
    "glaucoma",
    "hypertensive_retinopathy",
    "macular_degeneration",
    "normal"
]

COLORS = {
    "diabetic_retinopathy":     [(180, 40, 40), (220, 80, 60)],
    "glaucoma":                 [(40, 80, 180), (60, 120, 220)],
    "hypertensive_retinopathy": [(180, 100, 40), (220, 140, 60)],
    "macular_degeneration":     [(140, 100, 160), (180, 140, 200)],
    "normal":                   [(60, 160, 60), (80, 200, 80)],
}

IMG_SIZE   = 224
PER_CLASS  = 100


def make_fundus_placeholder(class_name: str, idx: int) -> Image.Image:
    """Synthetic circular fundus-like placeholder image."""
    img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (10, 10, 10))
    draw = ImageDraw.Draw(img)
    c1, c2 = COLORS[class_name]
    cx, cy = IMG_SIZE // 2, IMG_SIZE // 2
    r = 90

    # Retina disk
    for radius in range(r, 0, -1):
        ratio = radius / r
        col = tuple(int(c1[i] * ratio + c2[i] * (1 - ratio)) for i in range(3))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=col)

    # Optic disc
    draw.ellipse([cx - 14, cy - 18, cx + 14, cy + 18], fill=(255, 240, 200))

    # Vessels
    for _ in range(random.randint(6, 10)):
        angle = random.uniform(0, 2 * math.pi)
        length = random.randint(30, 70)
        x2 = int(cx + math.cos(angle) * length)
        y2 = int(cy + math.sin(angle) * length)
        draw.line([cx, cy, x2, y2], fill=(30, 0, 0), width=1)

    # Disease markers
    if class_name == "diabetic_retinopathy":
        for _ in range(random.randint(5, 12)):
            x = random.randint(cx - r + 10, cx + r - 10)
            y = random.randint(cy - r + 10, cy + r - 10)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 80, 80))
    elif class_name == "glaucoma":
        draw.ellipse([cx - 25, cy - 30, cx + 25, cy + 30], fill=(220, 200, 160))
    elif class_name == "macular_degeneration":
        for _ in range(random.randint(3, 8)):
            x = random.randint(cx - 25, cx + 25)
            y = random.randint(cy - 25, cy + 25)
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(200, 160, 80))

    img = img.filter(ImageFilter.GaussianBlur(0.5))
    return img


def create_dataset():
    root = Path("dataset")
    total = 0
    for cls in CLASSES:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        existing = len(list(cls_dir.glob("*.jpg")))
        if existing >= PER_CLASS:
            print(f"  ✓ {cls:30s} — {existing} images (already exists, skipping)")
            total += existing
            continue
        for i in range(PER_CLASS):
            img = make_fundus_placeholder(cls, i)
            img.save(cls_dir / f"{cls}_{i:03d}.jpg", quality=92)
        print(f"  ✓ {cls:30s} — {PER_CLASS} synthetic images created")
        total += PER_CLASS

    print(f"\n  Total: {total} images across {len(CLASSES)} classes")
    print("  Dataset is ready at ./dataset/\n")
    print("  ⚠  These are SYNTHETIC placeholders for pipeline testing.")
    print("     Replace with real fundus images before training for clinical use.\n")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Dataset Setup")
    print("="*55 + "\n")
    create_dataset()
