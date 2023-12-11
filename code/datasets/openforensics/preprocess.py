import json
import argparse
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm
from pathlib import Path


def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dset-path",
        '-d',
        type=str,
        help="Path to the OpenForensics dataset",
    )
    parser.add_argument(
        "--mode",
        '-m',
        type=str,
        help="Dataset mode (train/val/test)",
    )
    parser.add_argument(
        "--margin",
        '-g',
        type=float,
        default=1.3,
        help="Margin for face bounding box",
    )

    return parser


def scale_bounding_box(coords, height, width, scale=1.2, minsize=None):
    """
        Transforms bounding boxes to square and muptilpies it with a scale  
    """
    if len(coords) == 3:
        x1, y1, size_bb = coords
        x1, y1, x2, y2 = x1, y1, x1+size_bb, y1+size_bb
    else:
        x1, y1, x2, y2 = coords 
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, x1+size_bb, y1+size_bb


def apply_bounding_box_to_image(image,bounding_box,frame_height=None,frame_width=None,scale=None):
    if bounding_box is not None:
        if  scale is not None:
            x1, y1, x2, y2 = scale_bounding_box(bounding_box,frame_height,frame_width,scale)
        else:
            x1, y1, x2, y2 = bounding_box
        return image[y1:y2, x1:x2]
    else:
        return None
        
        
parser = get_argparser()
args = parser.parse_args()
print(args)

dset_path = Path(args.dset_path)
mode = args.mode.lower()
margin_str = str(args.margin).replace(".", "_")
if mode == "train":
    images_path = dset_path / "Train"
    json_path = dset_path / "Train_poly.json"
    destination_path = Path(__file__).parent / f"Train_faces_{margin_str}"
elif mode == "val":
    images_path = dset_path / "Val"
    json_path = dset_path / "Val_poly.json"
    destination_path = Path(__file__).parent / f"Val_faces_{margin_str}"
elif mode == "test":
    images_path = dset_path / "Test-Dev"
    json_path = dset_path / "Test-Dev_poly.json"
    destination_path = Path(__file__).parent / f"Test-Dev_faces_{margin_str}"
else:
    raise Exception(f"Only modes 'train', 'val', 'test' are supported, received {mode}")

assert dset_path.exists()
assert images_path.exists()
assert json_path.exists()

(destination_path / "real").mkdir(exist_ok=True, parents=True)
(destination_path / "fake").mkdir(exist_ok=True, parents=True)

with open(json_path) as f:
    md = json.load(f)

ids = []
image_ids = []
crowds = []
areas = []
categories = []
bboxes = []
for annot in md["annotations"]:
    ids.append(annot["id"])
    image_ids.append(annot["image_id"])
    crowds.append(annot["iscrowd"])
    areas.append(annot["area"])
    categories.append(annot["category_id"])
    bboxes.append(annot["bbox"])

df = pd.DataFrame(
    dict(
        id=ids,
        image_id=image_ids,
        is_crowd=crowds,
        area=areas,
        category_id=categories,
        bbox=bboxes,
    )
)
print("imported annotations")

for image_data in tqdm(md["images"]):
    # get image path
    image_id = image_data["file_name"].split("/")[-1]
    image_path = images_path / image_id

    # load image from path
    image = Image.open(image_path).convert("RGB")
    image = np.asarray(image, dtype=np.float32)
    height, width, _ = image.shape

    # get bboxes and labels for all persons in the image
    bboxes = df[df.image_id == image_data['id']].bbox.tolist()
    labels = df[df.image_id == image_data['id']].category_id.tolist()

    image_id, image_ext = image_id.split(".")
    for i, (bbox, l) in enumerate(zip(bboxes, labels)):
        x, y, xp, yp = bbox
        new_box = [x, y, x+xp, y+yp]
        face = apply_bounding_box_to_image(
            image,
            new_box,
            frame_height=height,
            frame_width=width,
            scale=args.margin
        )
        face_path = destination_path / ("real" if l == 0 else "fake") / f"{image_id}_{i}.{image_ext}"
        face = Image.fromarray(face.astype(np.uint8))
        face.save(face_path)
