import os
import random
import shutil

# Paths
images_path = "images"
labels_path = "labels-YOLO"

output = "RoadDataset"

# Create folders
for folder in [
    "train/images",
    "train/labels",
    "valid/images",
    "valid/labels",
]:
    os.makedirs(os.path.join(output, folder), exist_ok=True)

# Get image names
images = [f for f in os.listdir(images_path)
          if f.endswith((".jpg", ".jpeg", ".png"))]

# Shuffle
random.shuffle(images)

# 80-20 split
split = int(len(images) * 0.8)

train = images[:split]
valid = images[split:]

def copy_files(file_list, folder):

    for img in file_list:

        shutil.copy(
            os.path.join(images_path, img),
            os.path.join(output, folder, "images", img)
        )

        label = os.path.splitext(img)[0] + ".txt"

        shutil.copy(
            os.path.join(labels_path, label),
            os.path.join(output, folder, "labels", label)
        )

copy_files(train, "train")
copy_files(valid, "valid")

print("Done!")
print("Training images:", len(train))
print("Validation images:", len(valid))