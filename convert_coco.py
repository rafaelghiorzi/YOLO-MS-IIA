import os
import cv2
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

def convert_to_coco(dataset: str, output_dir: str, train_ratio: float = 0.6, val_ratio: float = 0.2):
    """
    Convert pedrozamboni dataset to COCO format with 60/20/20 split.
    """

    dataset_dir = Path(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create COCO format structure with test folder
    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "test").mkdir(parents=True, exist_ok=True)  # Added test folder
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Get annotation files
    bbox_files = list((dataset_dir / "bbox_txt").glob("*.txt"))
    image_files = list((dataset_dir / "images").glob("*.png"))

    matched_data = []
    for bbox_file in bbox_files:
        img_id = bbox_file.stem
        img_file = None

        for img_file in image_files:
            if img_file.stem == img_id:
                img_file = img_file
                break

        if img_file and img_file.exists():
            matched_data.append((img_file, bbox_file))

    print(f"Found {len(matched_data)} matching image and bbox files.")
    
    # First split: 60% train, 40% temp (for val + test)
    train_data, temp_data = train_test_split(
        matched_data, 
        train_size=train_ratio, 
        random_state=42
    )
    
    # Second split: 20% val, 20% test from the remaining 40%
    val_ratio_adjusted = val_ratio / (1 - train_ratio)  # 0.2 / 0.4 = 0.5
    val_data, test_data = train_test_split(
        temp_data, 
        train_size=val_ratio_adjusted, 
        random_state=42
    )

    print(f"Data split:")
    print(f"  Train: {len(train_data)} images ({len(train_data)/len(matched_data)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} images ({len(val_data)/len(matched_data)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} images ({len(test_data)/len(matched_data)*100:.1f}%)")

    # Process train, val, and test datasets
    for split, data in [("train", train_data), ("val", val_data), ("test", test_data)]:
        images = []
        annotations = []
        annotation_id = 1

        for img_id, (img_file, bbox_file) in enumerate(data):
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Warning: Could not read image {img_file}. Skipping.")
                continue

            height, width = img.shape[:2]

            # Copy image to output directory
            output_img_path = output_dir / "images" / split / img_file.name
            cv2.imwrite(str(output_img_path), img)

            images.append({
                "id": img_id,
                "file_name": img_file.name,
                "width": width,
                "height": height
            })

            # Read bounding boxes
            with open(bbox_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split() 
                    if len(parts) == 4:
                        x1, y1, x2, y2 = map(float, parts)

                        # Convert to COCO format (x, y, width, height)
                        x = min(x1, x2)
                        y = min(y1, y2)
                        bbox_width = abs(x2 - x1)
                        bbox_height = abs(y2 - y1)
                        
                        # Ensure coordinates are within image bounds
                        x = max(0, min(x, width - 1))
                        y = max(0, min(y, height - 1))
                        bbox_width = min(bbox_width, width - x)
                        bbox_height = min(bbox_height, height - y)

                        if bbox_width > 0 and bbox_height > 0:
                            annotations.append({
                                "id": annotation_id,
                                "image_id": img_id,
                                "category_id": 1,  # Assuming a single category
                                "bbox": [x, y, bbox_width, bbox_height],
                                "area": bbox_width * bbox_height,
                                "iscrowd": 0
                            })
                            annotation_id += 1
                    else:
                        print(f"Warning: Unexpected bbox format in {bbox_file}. Expected 4 values, got {len(parts)}. Skipping line: {line}")
        
        # Create COCO format
        coco_format = {
            "images": images,
            "annotations": annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "tree",
                    "supercategory": "plant"
                }
            ]
        }
        
        # Save annotation file
        with open(output_dir / "annotations" / f"instances_{split}.json", 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        print(f"✅ {split}: {len(images)} images, {len(annotations)} annotations")

  
if __name__ == "__main__":
    convert_to_coco(
        dataset="dataset",
        output_dir="coco_dataset",  # Changed to match your existing dataset
        train_ratio=0.6,  # 60% train
        val_ratio=0.2     # 20% val (20% test will be calculated automatically)
    )