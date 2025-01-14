import numpy as np

def combine_mmap_files(input_x_paths, input_y_paths, output_x_path, output_y_path):

    X_mmaps = [np.memmap(path, dtype='float32', mode='r', shape=(500, 15, 224, 224, 3)) for path in input_x_paths]
    y_mmaps = [np.memmap(path, dtype='int32', mode='r', shape=(500,)) for path in input_y_paths]

    total_sequences = sum(X.shape[0] for X in X_mmaps)
    seq_length, height, width, channels = 15, 224, 224, 3

    print(f"Total sequences in the combined dataset: {total_sequences}")

    X_final = np.memmap(output_x_path, dtype='float32', mode='w+', shape=(total_sequences, seq_length, height, width, channels))
    y_final = np.memmap(output_y_path, dtype='int32', mode='w+', shape=(total_sequences,))

    offset = 0
    for X, y in zip(X_mmaps, y_mmaps):
        num_sequences = X.shape[0]
        X_final[offset : offset + num_sequences] = X[:]
        y_final[offset : offset + num_sequences] = y[:]
        offset += num_sequences
        print(f"Processed {num_sequences} sequences from one file.")

    X_final.flush()
    y_final.flush()

    print(f"Data successfully combined into {output_x_path} and {output_y_path}")


input_x_paths = [r'E:\PosePerfect\Dataset Creation\Without Augmentation\X_biceps_7.5k.dat', r'E:\PosePerfect\Dataset Creation\Without Augmentation\X_squats_7.5k.dat']
input_y_paths = [r'E:\PosePerfect\Dataset Creation\Without Augmentation\y_biceps_7.5k.dat', r'E:\PosePerfect\Dataset Creation\Without Augmentation\y_squats_7.5k.dat']

output_x_path = 'X_combined_15k.dat'
output_y_path = 'y_combined_15k.dat'

combine_mmap_files(input_x_paths, input_y_paths, output_x_path, output_y_path)
