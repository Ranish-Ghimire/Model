import os
import numpy as np
from PIL import Image

def create_sequences(data_dir, seq_length, stride):
    sequences = []
    labels = []
    class_labels = {'biceps_curl': 0, 'squats': 1}

    for class_name, label in class_labels.items():
        class_dir = os.path.join(data_dir, class_name)
        frame_files = sorted([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        for i in range(0, len(frame_files) - seq_length + 1, stride):
            seq = []
            for j in range(seq_length):
                frame_path = os.path.join(class_dir, frame_files[i + j])
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img)
                seq.append(img_array)
            sequences.append(np.array(seq))
            labels.append(label)

    return np.array(sequences), np.array(labels)

data_directory = r"E:\PosePerfect\Video_Frames\Augmented_Frames\Flipped"
SEQ_LENGTH = 10
STRIDE = 5

X, y = create_sequences(data_directory, SEQ_LENGTH, STRIDE)
print(f"Sequences shape: {X.shape}")  # (num_sequences, SEQ_LENGTH, 224, 224, 3) (3088, 10, 224, 224, 3)
print(f"Labels shape: {y.shape}")  # (num_sequences,) (3088,)

np.save('X_flipped.npy', X)
np.save('y_flipped.npy', y)


