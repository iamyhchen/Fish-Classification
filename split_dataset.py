import os
import shutil
import random

source_dir = "dataset/Fish_Dataset"

target_base = "dataset"
splits = ["train", "test", "eval"]

# 可調整分割比例
split_ratios = {"train": 0.8, "test": 0.1, "eval": 0.1}

for split in splits:
    os.makedirs(os.path.join(target_base, split), exist_ok=True)

for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    images = [img for img in images if img.lower().endswith(".png")]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * split_ratios["train"])
    test_end = train_end + int(total * split_ratios["test"])

    split_data = {
        "train": images[:train_end],
        "test": images[train_end:test_end],
        "eval": images[test_end:],
    }

    for split in splits:
        split_dir = os.path.join(target_base, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img_name in split_data[split]:
            src_path = os.path.join(class_path, img_name)
            dst_path = os.path.join(split_dir, img_name)
            shutil.copy2(src_path, dst_path)

print("Dataset split completed successfully!")
