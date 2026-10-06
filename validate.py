import os

def verify_split(image_dir, label_dir, other_label_dir):

    for img in os.listdir(image_dir):

        label = os.path.splitext(img)[0] + ".txt"

        if not os.path.exists(os.path.join(label_dir, label)):
            print(f"❌ {label} missing from {label_dir}")

        if os.path.exists(os.path.join(other_label_dir, label)):
            print(f"❌ {label} is also present in {other_label_dir}")

print("Checking TRAIN...")
verify_split(
    "RoadDataset/train/images",
    "RoadDataset/train/labels",
    "RoadDataset/valid/labels"
)

print("\nChecking VALID...")
verify_split(
    "RoadDataset/valid/images",
    "RoadDataset/valid/labels",
    "RoadDataset/train/labels"
)

print("\nFinished checking.")