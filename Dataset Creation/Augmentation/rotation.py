import os
import cv2
import numpy as np
import natsort


def rotate_image(image, angle=25):
    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))

    return rotated_image


def augment_frames(input_dir, output_dir, rotation_angle=25):
    os.makedirs(output_dir, exist_ok=True)
    frame_files = natsort.natsorted([f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for item in frame_files:
            img_path = os.path.join(input_dir, item)
            img = cv2.imread(img_path)

            if img is not None:
                    rotated_img = rotate_image(img, rotation_angle)

                    save_path = os.path.join(output_dir, f"augmented_{item}")
                    cv2.imwrite(save_path, rotated_img)



def main():
    input_dir = r"E:\Frames\squats_frames"
    output_dir = r"E:\Frames\squats_frames"
    rotation_angle = 25

    augment_frames(input_dir, output_dir, rotation_angle)
    print("Rotation Augmentation Done!")


if __name__ == "__main__":
    main()
