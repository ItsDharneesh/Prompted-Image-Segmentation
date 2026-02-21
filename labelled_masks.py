import os
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
from pycocotools import mask as maskUtils

ROOT_DIR = "Data"
folders = ["cracks_processed/valid", "drywall_processed/valid"]

OUTPUT_DIR = [os.path.join(ROOT_DIR, f) for f in folders]


CRACK_LABEL = 255
TAPING_LABEL = 255

os.makedirs(os.path.join(OUTPUT_DIR[0], "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR[1], "images"), exist_ok=True)

os.makedirs(os.path.join(OUTPUT_DIR[0], "masks"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR[1], "masks"), exist_ok=True)


def clean_filename(filename):
    """
    Removes .rf hash and extension.
    Example:
    IMG_8210_JPG_jpg.rf.e4b0f7.jpg -> IMG_8210_JPG
    2503_jpg.rf.7efcb7.jpg -> 2503
    """
    base = os.path.splitext(filename)  # remove extension
    return base[0]


def process_dataset(dataset_path, class_value, prompt_suffix,output_dir):

    annotation_path = os.path.join(dataset_path, "_annotations.coco.json")

    with open(annotation_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}

    ann_dict = {}
    for ann in coco["annotations"]:
        ann_dict.setdefault(ann["image_id"], []).append(ann)

    for img_id, img_info in tqdm(images.items()):

        img_filename = img_info["file_name"]
        width = img_info["width"]
        height = img_info["height"]

        img_path = os.path.join(dataset_path, img_filename)
        if not os.path.exists(img_path):
            continue

        mask = np.zeros((height, width), dtype=np.uint8)

        if img_id in ann_dict:
            for ann in ann_dict[img_id]:

                seg = ann.get("segmentation", None)

                # Polygon
                if isinstance(seg, list) and len(seg) > 0:
                    for poly in seg:
                        poly = np.array(poly, dtype=np.float32)
                        if len(poly) < 6:
                            continue
                        poly = poly.reshape(-1, 2)
                        poly[:, 0] = np.clip(poly[:, 0], 0, width - 1)
                        poly[:, 1] = np.clip(poly[:, 1], 0, height - 1)
                        poly = poly.astype(np.int32)
                        cv2.fillPoly(mask, [poly], class_value)

                # RLE
                elif isinstance(seg, dict):
                    decoded = maskUtils.decode(seg)
                    mask[decoded == 1] = class_value

                # BBOX
                else:
                    bbox = ann.get("bbox", None)
                    if bbox:
                        x, y, w, h = bbox
                        x1 = int(max(0, x))
                        y1 = int(max(0, y))
                        x2 = int(min(width, x + w))
                        y2 = int(min(height, y + h))
                        mask[y1:y2, x1:x2] = class_value

        # ---------- NEW NAMING ----------
        clean_name = clean_filename(img_filename)
        new_base = f"{clean_name}__{prompt_suffix}"

        # Save image
        Image.open(img_path).convert("RGB").save(
            os.path.join(output_dir, "images", new_base + ".jpg")
        )

        # Save mask
        cv2.imwrite(
            os.path.join(output_dir, "masks", new_base + ".png"),
            mask
        )


# RUN
process_dataset(
    os.path.join(ROOT_DIR, "cracks/valid"),
    CRACK_LABEL,
    "segment_crack",OUTPUT_DIR[0]
)

process_dataset(
    os.path.join(ROOT_DIR, "drywall/valid"),
    TAPING_LABEL,
    "segment_taping_area",OUTPUT_DIR[1]
)

print("Done.")
