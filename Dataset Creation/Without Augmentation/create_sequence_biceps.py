import os
import numpy as np
from PIL import Image
import natsort

def create_sequences(data_dir, seq_length, stride, output_x_path, output_y_path):

    X_mmap = np.memmap(output_x_path, dtype='float32', mode='w+', shape=(806, seq_length, 224, 224, 3))
    y_mmap = np.memmap(output_y_path, dtype='int32', mode='w+', shape=(806,))

    sequence_idx = 0
    frame_files = natsort.natsorted([f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    for i in range(0, len(frame_files) - seq_length + 1, stride):
        seq = []
        for j in range(seq_length):
            frame_path = os.path.join(data_dir, frame_files[i + j])
            img = Image.open(frame_path).convert('RGB')
            img_array = np.array(img) / 255.0
            seq.append(img_array)

        X_mmap[sequence_idx] = np.array(seq, dtype=np.float32)
        y_mmap[sequence_idx] = 0
        sequence_idx += 1

        print(f"Processed {sequence_idx} sequences so far.")

    X_mmap.flush()
    y_mmap.flush()

    print(f"Data successfully written to {output_x_path} and {output_y_path}")

data_directory = r"E:\PosePerfect\Frames\biceps_curl"
SEQ_LENGTH = 15
STRIDE = 15

output_x_path = 'X_biceps_curl_806.dat'
output_y_path = 'y_biceps_curl_806.dat'

create_sequences(data_directory, SEQ_LENGTH, STRIDE, output_x_path, output_y_path)
