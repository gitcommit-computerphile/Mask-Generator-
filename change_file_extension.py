import os
import cv2

# Directory containing images
image_dir = "masks"

for filename in os.listdir(image_dir):

    if not filename.lower().endswith(".png"):
        continue

    # Example: frame_000123.png
    number = filename.split("_")[1].split(".")[0]

    new_name = f"static_cans_march6_{number}.jpg"

    old_path = os.path.join(image_dir, filename)
    new_path = os.path.join(image_dir, new_name)

    # Read image
    img = cv2.imread(old_path)

    # Save as JPG
    cv2.imwrite(new_path, img)

    # Remove old PNG
    os.remove(old_path)

    print(f"{filename} -> {new_name}")

print("All images renamed and converted.")
