import os
import cv2
import albumentations as A
import natsort

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=1),  # Flip horizontally with 100% probability
        A.RandomBrightnessContrast(brightness_limit=(-0.26, -0.1), contrast_limit=(-0.1, 0), p=1.0),  # decrease
        # A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=(0, 0.1), p=1.0),  # increase
        A.GaussianBlur(blur_limit=(3, 7), p=1),  # Apply Gaussian blur with a random kernel size
        A.MotionBlur(blur_limit=7, p=1),  # Apply motion blur with a maximum kernel size of 7
    ])


def generate_consistent_params(augmentations):
    """Manually generate consistent parameters for all augmentations."""
    params = {}
    for aug in augmentations:
        params[type(aug).__name__] = aug.get_params()
    return params

def apply_augmentations(image, augmentations, params):
    """Apply augmentations to an image using the same parameters."""
    for aug in augmentations:
        aug_type = type(aug).__name__
        image = aug.apply(image, **params[aug_type])
    return image

def augment_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    augmentations = get_augmentation_pipeline()

    params = generate_consistent_params(augmentations)

    frame_files = natsort.natsorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for item in frame_files:
        img_path = os.path.join(input_dir, item)
        img = cv2.imread(img_path)

        if img is not None:
            augmented = apply_augmentations(img, augmentations, params)
            save_path = os.path.join(output_dir, f"augmented_{item}")
            cv2.imwrite(save_path, augmented)

    print(f"Augmented frames saved to {output_dir}")

input_dir = r"E:\Frames\biceps_curl_frame"
output_dir = r"E:\Frames\biceps_curl_frame"

augment_frames(input_dir, output_dir)
print("Augmentation Done!")
