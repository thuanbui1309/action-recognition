import cv2
import numpy as np
import os
from tqdm import tqdm
from PIL import Image, ImageEnhance
from moviepy.editor import VideoFileClip, vfx
import argparse
import concurrent.futures
import shutil
import random
from sklearn.model_selection import train_test_split

# Resize video
def resize_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (172, 172), interpolation=cv2.INTER_CUBIC)
        out.write(frame)

    cap.release()
    out.release()

    return output_path if os.path.exists(output_path) else None

# Flip video horizontally
def flip_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.flip(frame, 1))  # Flip horizontally

    cap.release()
    out.release()

# Apply Gaussian blur to video
def blur_video(input_path, output_path, kernel_size=(5, 5)):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(cv2.GaussianBlur(frame, kernel_size, 0))

    cap.release()
    out.release()

# Add noise to video
def add_noise_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        noise = np.random.normal(0, 20, frame.shape).astype(np.uint8)
        out.write(cv2.add(frame, noise))

    cap.release()
    out.release()

# Reduce brightness of video
def darken_video(input_path, output_path, factor=0.3):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        darkened_frame = ImageEnhance.Brightness(frame).enhance(factor)
        out.write(cv2.cvtColor(np.array(darkened_frame), cv2.COLOR_RGB2BGR))

    cap.release()
    out.release()

# Drop frames randomly from video
def drop_frames(input_path, output_path, drop_prob=0.2):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (172, 172))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if np.random.rand() > drop_prob:
            out.write(frame)

    cap.release()
    out.release()

# Change video speed
def change_speed(input_path, output_path, speed_factor):
    if not os.path.exists(input_path):
        return

    clip = VideoFileClip(input_path).fx(vfx.speedx, speed_factor)
    clip.write_videofile(output_path, codec="libx264", fps=clip.fps, verbose=False, logger=None)

# Perform augmentation
def augment_video(video_name, video_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    resized_video = resize_video(video_path, f"{os.path.join(output_dir, video_name)}.mp4")

    if not resized_video:
        print(f"Skipping {video_name} due to resize error!")
        return
    if resized_video:
        flip_video(resized_video, os.path.join(output_dir, f"{video_name}_flipped.mp4"))
        blur_video(resized_video, os.path.join(output_dir, f"{video_name}_blurred.mp4"))
        # add_noise_video(resized_video, os.path.join(output_dir, f"{video_name}_noisy.mp4"))
        darken_video(resized_video, os.path.join(output_dir, f"{video_name}_darkened.mp4"))
        # drop_frames(resized_video, os.path.join(output_dir, f"{video_name}_frame_dropped.mp4"))
        # change_speed(resized_video, os.path.join(output_dir, f"{video_name}_sped_up.mp4"), 1.2)
        # change_speed(resized_video, os.path.join(output_dir, f"{video_name}_slowed_down.mp4"), 0.8)
    else:
        print(f"Skipping {video_name} due to resize error!")

def split_data(input_folder, output_folder, train_ratio=0.8, test_ratio=0.2):
    assert train_ratio + test_ratio == 1.0, "Train and test ratios must sum to 1"

    os.makedirs(output_folder, exist_ok=True)

    for split in ["train", "test"]:
        os.makedirs(os.path.join(output_folder, split), exist_ok=True)

    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        videos = [f for f in os.listdir(class_path) if f.endswith(".mp4")]
        random.shuffle(videos)

        train_videos, test_videos = train_test_split(videos, test_size=test_ratio, random_state=42)

        for split, video_list in zip(["train", "test"], [train_videos, test_videos]):
            split_class_path = os.path.join(output_folder, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            for video in video_list:
                shutil.copy(os.path.join(class_path, video), os.path.join(split_class_path, video))

    print("Data splitting completed!")

def split_data_with_validation(input_folder, output_folder, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Tổng của train, val, test phải bằng 1"

    os.makedirs(output_folder, exist_ok=True)

    # Tạo các thư mục train, val, test
    for split in ["train", "val", "test"]:
        split_path = os.path.join(output_folder, split)
        os.makedirs(split_path, exist_ok=True)

    # Lặp qua từng lớp (folder)
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue

        videos = [f for f in os.listdir(class_path) if f.endswith(".mp4")]

        # Shuffle dữ liệu để tránh bias
        random.shuffle(videos)

        # Chia dữ liệu
        train_videos, temp_videos = train_test_split(videos, test_size=(1 - train_ratio), random_state=42)
        val_videos, test_videos = train_test_split(temp_videos, test_size=(test_ratio / (val_ratio + test_ratio)),
                                                   random_state=42)

        # Copy vào thư mục tương ứng
        for split, video_list in zip(["train", "val", "test"], [train_videos, val_videos, test_videos]):
            split_class_path = os.path.join(output_folder, split, class_name)
            os.makedirs(split_class_path, exist_ok=True)

            for video in video_list:
                src = os.path.join(class_path, video)
                dst = os.path.join(split_class_path, video)
                shutil.copy(src, dst)

    print("Data splitting with validation completed!")

# Main function (parallel version)
def main():
    parser = argparse.ArgumentParser(description="Data Augmentation")
    parser.add_argument("--input", default="data/UCF101", help="Path to raw videos")
    parser.add_argument("--output", default="data/UCF101_augmented", help="Path to output folder")
    parser.add_argument("--split_output", default="data/UCF101_split", help="Path to split folder")
    parser.add_argument(
        "--labels",
        default="BenchPress,Biking,PushUps,PullUps,Diving,Basketball,TennisSwing,GolfSwing,BaseballPitch,SoccerPenalty",
        help="Augment only chosen labels"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    if args.labels == "all":
        files = [[file.split("_")[1], file] for file in os.listdir(args.input)]
    else:
        files = [[file.split("_")[1], file] for file in os.listdir(args.input) if
                 file.split("_")[1] in args.labels.split(",")]

    print(f"Starting augmentation for {len(files)} videos:\n")

    for idx, (label, file) in enumerate(files):
        print(f"Processing video {idx + 1}/{len(files)}: {file}")
        video_path = os.path.join(args.input, file)
        output_path = os.path.join(args.output, label)
        augment_video(file[:-4], video_path, output_path)

    # # Verify
    # count = 0
    # for folder in os.listdir(args.output):
    #     count += len(os.listdir(os.path.join(args.output, folder)))
    #
    # if count == len(files) * 8:
    #     print("No files lost")

    # split_data(args.output, args.split_output)
    split_data_with_validation(args.output, f"{args.split_output}_with_val")

    # # Verify split data
    # count = 0
    # for folder in os.listdir(f"{args.split_output}_with_val"):
    #     for class_name in os.listdir(os.path.join(f"{args.split_output}_with_val", folder)):
    #         count += len(os.listdir(os.path.join(f"{args.split_output}_with_val", folder, class_name)))
    #     # count += len(os.listdir(os.path.join(f"{args.split_output}_with_val", folder)))
    #
    # if count == len(files) * 8:
    #     print("No files lost after splitting")

if __name__ == "__main__":
    main()

# python augment.py - -labels all && python augment.py --output data/UCF101_subset_augmented --split_output data/UCF101_subset_split
