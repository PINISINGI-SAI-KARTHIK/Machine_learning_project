import os
import cv2
import albumentations as A

# Input folder: your original connector images
input_folder = "New folder/"
output_folder = "Aug/"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=10, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.MotionBlur(blur_limit=3, p=0.3),
    A.RandomScale(scale_limit=0.1, p=0.5),
])

# How many augmented versions per image
num_aug = 10

for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    # Save original copy too
    base_name = os.path.splitext(img_name)[0]
    cv2.imwrite(os.path.join(output_folder, f"{base_name}_orig.jpg"), image)

    # Generate augmentations
    for i in range(num_aug):
        augmented = transform(image=image)
        aug_img = augmented["image"]
        cv2.imwrite(os.path.join(output_folder, f"{base_name}_aug{i}.jpg"), aug_img)

print("✅ Augmentation complete! Check folder:", output_folder)
