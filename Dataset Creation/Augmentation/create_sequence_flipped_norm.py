import os
import numpy as np
from PIL import Image
import natsort

def create_sequences(data_dir, seq_length, stride, output_x_path, output_y_path):
    class_labels = {'biceps_curl': 0, 'squats': 1}
    total_sequences = 0

    for class_name in class_labels:
        class_dir = os.path.join(data_dir, class_name)
        frame_files = natsort.natsorted([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
        total_sequences += ((len(frame_files) - seq_length) // stride) + 1
        print(f"Number of sequences in {class_name}: {((len(frame_files) - seq_length) // stride) + 1}") # 1107, 1434

    print(f"Total sequences to create: {total_sequences}") # 2541

    X_mmap = np.memmap(output_x_path, dtype='float32', mode='w+', shape=(total_sequences, seq_length, 224, 224, 3))
    y_mmap = np.memmap(output_y_path, dtype='int32', mode='w+', shape=(total_sequences,))

    sequence_idx = 0

    for class_name, label in class_labels.items():
        class_dir = os.path.join(data_dir, class_name)
        frame_files = natsort.natsorted([f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

        for i in range(0, len(frame_files) - seq_length + 1, stride):
            seq = []
            for j in range(seq_length):
                frame_path = os.path.join(class_dir, frame_files[i + j])
                img = Image.open(frame_path).convert('RGB')
                img_array = np.array(img) / 255.0
                seq.append(img_array)

            X_mmap[sequence_idx] = np.array(seq, dtype=np.float32)
            y_mmap[sequence_idx] = label
            sequence_idx += 1

        print(f"Processed {class_name}: {sequence_idx} sequences so far.")

    X_mmap.flush()
    y_mmap.flush()

    print(f"Data successfully written to {output_x_path} and {output_y_path}")

data_directory = r"E:/PosePerfect/Video_Frames/Augmented_Frames/Flipped"
SEQ_LENGTH = 10
STRIDE = 5

output_x_path = 'X_flipped.dat'
output_y_path = 'y_flipped.dat'

create_sequences(data_directory, SEQ_LENGTH, STRIDE, output_x_path, output_y_path)
