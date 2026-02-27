import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms(stage='train', img_size=224):
    """
    Returns strong clinical augmentations for skin lesions.
    """
    if stage == 'train':
        return A.Compose([
            # 1. Geometric Invariance (Lesions have no "up" or "down")
            A.Transpose(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            
            # 2. Morphological Variation (Simulate skin stretching/camera angle)
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=45, p=0.5),
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.5, shift_limit=0.0, p=1),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1),
            ], p=0.3),
            
            # 3. Color Constancy (Simulate different lighting conditions)
            # We keep Hue shifts low because color is diagnostic (red vs brown)
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
            
            # 4. Regularization (Cutout forces model to look at context, not just center)
            A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, p=0.3),
            
            # 5. Normalization (ImageNet stats or calculated dataset stats)
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    else: # Validation / Test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])