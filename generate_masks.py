import json
import cv2
import numpy as np
import os

# Paths
json_path = "updated_binary_segmenter(7).json"
image_dir = "images"          # folder containing annotated images
mask_dir = "masks"            # output folder

os.makedirs(mask_dir, exist_ok=True)

# -----------------------------
# Conveyor ROI polygons
# -----------------------------
polygon_rtmp_stream = np.array([
    [2,356],[93,325],[183,295],[289,264],[367,243],[409,237],
    [471,234],[583,228],[635,212],[727,183],[757,178],[781,181],
    [816,192],[874,228],[893,244],[894,270],[847,232],[823,216],
    [797,209],[773,208],[751,221],[715,237],[679,253],[651,264],
    [617,272],[545,280],[469,284],[413,289],[368,298],[291,322],
    [185,356],[67,392],[1,411]
])

conveyor2_polygon_rtmp = np.array([
[34,284],[78,268],[182,234],[278,209],[347,192],[417,177],
[444,171],[481,159],[522,151],[521,156],[491,163],[435,177],
[392,187],[350,197],[307,207],[272,215],[224,228],[179,240],
[117,259],[82,271],[39,290]
])

conveyor3_polygon_rtmp = np.array([
[45,306],[48,312],[110,292],[261,248],[361,222],[467,197],
[528,183],[529,176],[454,194],[369,214],[286,236],[160,270]
])

roi_polygons = [
    polygon_rtmp_stream,
    conveyor2_polygon_rtmp,
    conveyor3_polygon_rtmp
]

# -----------------------------
# Load VIA JSON
# -----------------------------
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

    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # -----------------------------
    # Mask from annotations
    # -----------------------------
    annotation_mask = np.zeros((h, w), dtype=np.uint8)

    for region in regions:

        shape = region["shape_attributes"]

        xs = shape["all_points_x"]
        ys = shape["all_points_y"]

        polygon = np.array(list(zip(xs, ys)), dtype=np.int32)

        cv2.fillPoly(annotation_mask, [polygon], 255)

    # -----------------------------
    # ROI mask (3 conveyors)
    # -----------------------------
    roi_mask = np.zeros((h, w), dtype=np.uint8)

    for poly in roi_polygons:
        cv2.fillPoly(roi_mask, [poly], 255)

    # -----------------------------
    # Remove pixels outside conveyors
    # -----------------------------
    final_mask = cv2.bitwise_and(annotation_mask, roi_mask)

    # -----------------------------
    # Save mask
    # -----------------------------
    mask_path = os.path.join(mask_dir, filename)
    cv2.imwrite(mask_path, final_mask)

    print("Saved:", mask_path)

print("All masks generated.")
