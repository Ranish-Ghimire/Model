import os
import cv2
import albumentations as A
import natsort

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=1.0),
    ])


def augment_frames(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    augment = get_augmentation_pipeline()
    frame_files = natsort.natsorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for item in frame_files:
            img_path = os.path.join(input_dir, item)
            img = cv2.imread(img_path)

            if img is not None:
                augmented = augment(image=img)['image']
                save_path = os.path.join(output_dir, f"augmented_{item}")
                cv2.imwrite(save_path, augmented)



input_dir = r"E:\Frames\squats_frames"
output_dir = r"E:\Frames\squats_frames"

augment_frames(input_dir, output_dir)
print("Augmentation Done!")
