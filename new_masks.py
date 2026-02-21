import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIR = "Data/drywall_processed/valid/images"
MASK_DIR = "Data/drywall_processed/valid/masks"
OUTPUT_DIR = "Data/drywall_processed/valid/refined_masks"

LINE_THICKNESS = 8  # width of seam mask

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = sorted(os.listdir(IMAGE_DIR))

for filename in tqdm(image_files):

    base = os.path.splitext(filename)[0]
    img_path = os.path.join(IMAGE_DIR, filename)
    mask_path = os.path.join(MASK_DIR, base + ".png")

    if not os.path.exists(mask_path):
        continue

    image = cv2.imread(img_path)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use mask to limit search region
    bbox_mask = cv2.imread(mask_path, 0)
    if bbox_mask is None:
        continue

    masked_gray = cv2.bitwise_and(gray, gray, mask=bbox_mask)

    edges = cv2.Canny(masked_gray, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=80,
        minLineLength=gray.shape[0] // 3,
        maxLineGap=20
    )

    if lines is None:
        continue

    # pick longest detected line
    best_line = None
    max_len = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length > max_len:
            max_len = length
            best_line = (x1, y1, x2, y2)

    if best_line is None:
        continue

    # Create new mask
    h, w = gray.shape
    new_mask = np.zeros((h, w), dtype=np.uint8)

    x1, y1, x2, y2 = best_line
    cv2.line(new_mask, (x1, y1), (x2, y2), 255, LINE_THICKNESS)

    cv2.imwrite(os.path.join(OUTPUT_DIR, base + ".png"), new_mask)

print("Done.")