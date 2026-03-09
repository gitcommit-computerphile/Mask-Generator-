import json
import cv2
import numpy as np
import os

# Paths
json_path = "updated_binary_segmenter(7).json"
image_dir = "images"          # folder containing annotated images
mask_dir = "masks"            # output folder

os.makedirs(mask_dir, exist_ok=True)

# Load JSON
with open(json_path) as f:
    data = json.load(f)

img_metadata = data["_via_img_metadata"]

for key in img_metadata:

    item = img_metadata[key]
    filename = item["filename"]
    regions = item["regions"]

    image_path = os.path.join(image_dir, filename)

    if not os.path.exists(image_path):
        print("Missing image:", filename)
        continue

    # Load image to get shape
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # Create black mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw polygons
    for region in regions:

        shape = region["shape_attributes"]

        xs = shape["all_points_x"]
        ys = shape["all_points_y"]

        polygon = np.array(list(zip(xs, ys)), dtype=np.int32)

        cv2.fillPoly(mask, [polygon], 255)

    # Save mask
    mask_path = os.path.join(mask_dir, filename)
    cv2.imwrite(mask_path, mask)

    print("Saved mask:", mask_path)

print("All masks generated.")
