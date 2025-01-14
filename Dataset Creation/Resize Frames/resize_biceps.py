from PIL import Image
import os

input_path = r"E:\Frames\biceps_curl_frame"
output_path = r"E:\Frames\biceps_curl_frame"

os.makedirs(output_path, exist_ok=True)

size = 224

for item in os.listdir(input_path):
    input_file = os.path.join(input_path, item)
    output_file = os.path.join(output_path, item)

    if os.path.isfile(input_file):
        try:
            with Image.open(input_file) as im:
                im_resized = im.resize((size, size), Image.LANCZOS)
                im_resized.save(output_file, 'JPEG', quality=90)
        except Exception as e:
            print(f"Error processing {item}: {e}")
    else:
        print(f"Skipping non-file item: {item}")

print("All images resized and saved!")
