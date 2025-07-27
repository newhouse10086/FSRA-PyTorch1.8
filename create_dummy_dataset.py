#!/usr/bin/env python3
"""
Create dummy dataset for testing FSRA training pipeline.
"""

import os
import numpy as np
from PIL import Image
import argparse

def create_dummy_image(size=(256, 256), color_mode='RGB'):
    """Create a dummy image with random colors."""
    if color_mode == 'RGB':
        # Create random RGB image
        image_array = np.random.randint(0, 256, (*size, 3), dtype=np.uint8)
    else:
        # Create grayscale image
        image_array = np.random.randint(0, 256, size, dtype=np.uint8)
    
    return Image.fromarray(image_array)

def create_dummy_dataset(data_dir='data', num_classes=10, images_per_class=5):
    """
    Create a dummy dataset with the University-1652 structure.
    
    Args:
        data_dir: Root directory for dataset
        num_classes: Number of classes to create
        images_per_class: Number of images per class
    """
    print(f"Creating dummy dataset in {data_dir}...")
    print(f"Classes: {num_classes}, Images per class: {images_per_class}")
    
    # Define the directory structure
    directories = [
        'train/satellite',
        'train/drone', 
        'test/query_satellite',
        'test/query_drone',
        'test/gallery_satellite'
    ]
    
    # Create directories
    for directory in directories:
        full_path = os.path.join(data_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"✓ Created directory: {full_path}")
    
    # Create dummy images for each directory
    for directory in directories:
        full_path = os.path.join(data_dir, directory)
        
        for class_id in range(num_classes):
            class_dir = os.path.join(full_path, f"{class_id:04d}")
            os.makedirs(class_dir, exist_ok=True)
            
            # Determine number of images for this directory
            if 'train' in directory:
                num_images = images_per_class
            else:
                # Fewer images for test sets
                num_images = max(1, images_per_class // 2)
            
            for img_id in range(num_images):
                # Create different image patterns for satellite vs drone
                if 'satellite' in directory:
                    # Satellite images - more structured patterns
                    img = create_dummy_image((256, 256), 'RGB')
                    # Add some structure (grid-like pattern)
                    img_array = np.array(img)
                    img_array[::32, :] = [100, 150, 200]  # Horizontal lines
                    img_array[:, ::32] = [100, 150, 200]  # Vertical lines
                    img = Image.fromarray(img_array)
                else:
                    # Drone images - more random patterns
                    img = create_dummy_image((256, 256), 'RGB')
                
                # Save image
                if 'train' in directory:
                    img_name = f"{class_id:04d}_{img_id:03d}.jpg"
                else:
                    img_name = f"{class_id:04d}_{img_id:03d}.jpg"
                
                img_path = os.path.join(class_dir, img_name)
                img.save(img_path, 'JPEG', quality=85)
        
        print(f"✓ Created {num_classes} classes with images in {directory}")
    
    print(f"\n🎉 Dummy dataset created successfully!")
    print(f"Dataset structure:")
    print(f"  {data_dir}/")
    for directory in directories:
        full_path = os.path.join(data_dir, directory)
        class_count = len([d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))])
        print(f"    {directory}: {class_count} classes")

def verify_dataset(data_dir='data'):
    """Verify the created dataset structure."""
    print(f"\n🔍 Verifying dataset structure in {data_dir}...")
    
    required_dirs = [
        'train/satellite',
        'train/drone',
        'test/query_satellite', 
        'test/query_drone',
        'test/gallery_satellite'
    ]
    
    all_good = True
    
    for directory in required_dirs:
        full_path = os.path.join(data_dir, directory)
        if os.path.exists(full_path):
            class_dirs = [d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))]
            total_images = 0
            for class_dir in class_dirs:
                class_path = os.path.join(full_path, class_dir)
                images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                total_images += len(images)
            
            print(f"  ✓ {directory}: {len(class_dirs)} classes, {total_images} images")
        else:
            print(f"  ❌ {directory}: NOT FOUND")
            all_good = False
    
    if all_good:
        print("✅ Dataset structure verification passed!")
        return True
    else:
        print("❌ Dataset structure verification failed!")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Create dummy dataset for FSRA testing')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Root directory for dataset (default: data)')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes to create (default: 10)')
    parser.add_argument('--images_per_class', type=int, default=5,
                       help='Number of images per class (default: 5)')
    parser.add_argument('--verify_only', action='store_true',
                       help='Only verify existing dataset structure')
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_dataset(args.data_dir)
    else:
        create_dummy_dataset(args.data_dir, args.num_classes, args.images_per_class)
        verify_dataset(args.data_dir)
        
        print(f"\n🚀 Ready to test training!")
        print(f"Run this command to test:")
        print(f"python scripts/train.py --config config/default_config.yaml --model FSRA --data_dir {args.data_dir} --batch_size 2 --num_epochs 1")

if __name__ == '__main__':
    main()
