import json
import cv2
import numpy as np
from pathlib import Path
import random
import argparse

def load_coco_data(annotation_file):
    """Load COCO annotation data"""
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

def visualize_image_with_bboxes(coco_data, image_dir, image_id=None, save_path=None, show=False):
    """
    Visualize a single image with its bounding boxes
    """
    # Get images and annotations
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: cat for cat in coco_data['categories']}
    
    # Select image
    if image_id is None:
        image_id = random.choice(list(images.keys()))
    
    if image_id not in images:
        print(f"Image ID {image_id} not found!")
        return
    
    image_info = images[image_id]
    image_path = Path(image_dir) / image_info['file_name']
    
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Get annotations for this image
    annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
    
    print(f"Image: {image_info['file_name']}")
    print(f"Dimensions: {image_info['width']} x {image_info['height']}")
    print(f"Number of annotations: {len(annotations)}")
    
    # Draw bounding boxes
    for ann in annotations:
        x, y, w, h = ann['bbox']
        x, y, w, h = int(x), int(y), int(w), int(h)
        
        # Get category info
        category = categories[ann['category_id']]
        label = category['name']
        
        # Draw rectangle (Green color in BGR format)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(img, (x, y - label_size[1] - 10), (x + label_size[0], y), (0, 255, 0), -1)
        cv2.putText(img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        print(f"  - {label}: bbox=[{x}, {y}, {w}, {h}], area={ann['area']}")
    
    # Always save the image (since show doesn't work)
    if save_path is None:
        save_path = f"visualization_{image_info['file_name']}"
    
    cv2.imwrite(save_path, img)
    print(f"✅ Saved visualization to: {save_path}")
    
    return img

def visualize_multiple_images(coco_data, image_dir, num_images=5, save_dir=None):
    """
    Visualize multiple random images from the dataset
    """
    images = coco_data['images']
    selected_images = random.sample(images, min(num_images, len(images)))
    
    if save_dir is None:
        save_dir = "visualizations"
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for i, img_info in enumerate(selected_images):
        print(f"\n=== Image {i+1}/{len(selected_images)} ===")
        
        save_path = f"{save_dir}/visualization_{img_info['id']}_{img_info['file_name']}"
        
        visualize_image_with_bboxes(
            coco_data, 
            image_dir, 
            image_id=img_info['id'], 
            save_path=save_path,
            show=False  # Disable show since it doesn't work
        )

def get_dataset_statistics(coco_data):
    """
    Print dataset statistics
    """
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {len(coco_data['images'])}")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(coco_data['categories'])}")
    
    # Category distribution
    category_counts = {}
    for ann in coco_data['annotations']:
        cat_id = ann['category_id']
        category_counts[cat_id] = category_counts.get(cat_id, 0) + 1
    
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    print("\nCategory distribution:")
    for cat_id, count in category_counts.items():
        print(f"  {categories[cat_id]}: {count} annotations")
    
    # Image size statistics
    widths = [img['width'] for img in coco_data['images']]
    heights = [img['height'] for img in coco_data['images']]
    print(f"\nImage sizes:")
    print(f"  Width - Min: {min(widths)}, Max: {max(widths)}, Avg: {sum(widths)/len(widths):.1f}")
    print(f"  Height - Min: {min(heights)}, Max: {max(heights)}, Avg: {sum(heights)/len(heights):.1f}")

def main():
    parser = argparse.ArgumentParser(description='Visualize COCO dataset')
    parser.add_argument('--dataset', default='teste_coco', help='Path to COCO dataset directory')
    parser.add_argument('--split', choices=['train', 'val'], default='train', help='Dataset split to visualize')
    parser.add_argument('--image-id', type=int, help='Specific image ID to visualize')
    parser.add_argument('--num-images', type=int, default=3, help='Number of random images to show')
    parser.add_argument('--save-dir', default='visualizations', help='Directory to save visualizations')
    parser.add_argument('--stats-only', action='store_true', help='Only show dataset statistics')
    
    args = parser.parse_args()
    
    # Paths
    dataset_dir = Path(args.dataset)
    annotation_file = dataset_dir / "annotations" / f"instances_{args.split}.json"
    image_dir = dataset_dir / "images" / args.split
    
    # Check if files exist
    if not annotation_file.exists():
        print(f"Annotation file not found: {annotation_file}")
        return
    
    if not image_dir.exists():
        print(f"Image directory not found: {image_dir}")
        return
    
    # Load COCO data
    print(f"Loading COCO data from: {annotation_file}")
    coco_data = load_coco_data(annotation_file)
    
    # Show statistics
    get_dataset_statistics(coco_data)
    
    if args.stats_only:
        return
    
    # Visualize images
    if args.image_id:
        print(f"\nVisualizing specific image ID: {args.image_id}")
        save_path = f"{args.save_dir}/visualization_image_{args.image_id}.jpg"
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        visualize_image_with_bboxes(coco_data, image_dir, image_id=args.image_id, save_path=save_path)
    else:
        print(f"\nVisualizing {args.num_images} random images...")
        visualize_multiple_images(coco_data, image_dir, num_images=args.num_images, save_dir=args.save_dir)

if __name__ == "__main__":
    main()