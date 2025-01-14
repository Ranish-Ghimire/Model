import cv2
import os

def extract_frames(input_dir, output_dir, frame_rate):

    os.makedirs(output_dir, exist_ok=True)

    for video_file in os.listdir(input_dir):
        if video_file.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / frame_rate))

            frame_count = 0
            success, frame = cap.read()
            frame_index = 1

            while success:
                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(output_dir, f"{video_name}_frame{frame_index}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_index += 1
                success, frame = cap.read()
                frame_count += 1

            cap.release()
            print(f"Frames extracted for video: {video_file}")

input_videos_directory = r"E:\Videos\squats"
output_frames_directory = r"E:\Frames\squats_frames"
extract_frames(input_videos_directory, output_frames_directory, frame_rate=5)