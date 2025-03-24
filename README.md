# Final Project: Human Action Recognition (HAR) - Bui Minh Thuan (104486358)

This README file provides instructions on how to run the code for final project, unit COS30028 - Spring 2025

## 1. Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/thuanbui1309/action-recognition.git
cd action-recognition
pip install -r requirements.txt
```

## 2. Data and Model Download

Data and models for all  tasks are uploaded in this Google Drive folder. Please go there and download them:

```bash
https://drive.google.com/drive/folders/1tElWQyrQ2OA5MMUxUwf_UelbDUpp1czJ?usp=sharing
```

After downloading, extract the files and move them to the appropriate directories. The correct structure will look like this:
```
action-recognition/
│
├── data/
│   └── demo/
│       ├── test1.mp4/
│       ├── test2.mp4/
│       └── test3.mp4/
│   └── HGP/
│       ├── images/
│       │   ├── annotations/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│       │   ├── train/
│       │   ├── val/
│       └── labels_old/
│           ├── train/
│           ├── val/
│   └── HGP_phone_hand/
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│       │   ├── train/
│       │   ├── val/
│       └── data.yaml
│   └── UCF101/
│       ├── v_ApplyEyeMakeup_g01_c01.avi
│       ├── v_ApplyEyeMakeup_g01_c02.avi
│       └── ...
├── models/
│   ├── movinet/
│   │   └── a0
│   │   └── trainings
│   │   └── labels.npy
│   └── yolo phone hand detection/
│   └── yolo pose/
├── augment.py
├── classify.py
├── pose_estimation.py
├── process_hgp.py
├── requirements.txt
└── README.md
```

## 3. Running MoViNet Video Classification

### 3.1 Data Preprocessing

To run data preprocessing, you can run file `augment.py`. This will automatically augment and split the data into correct directory. You can customize the augmentation parameters in the file.

```bash
python augment.py
```

Parameters:
- `--input`: Path to raw videos
- `--output`: Path to output videos
- `--split_output`: Path to splited folder
- `--labels`: Augment only chosen labels
- `--workers`: Number of parallel workers

### 3.2 Training

The model is trained on Google Colab. You can access the training notebook and saved model in folder `models/movinet`.

### 3.3 Inference

To run inference on a video, you can use the `classify.py` script:

```bash
# Example command
python classify.py --input data/demo/test1.mp4
```

Parameters:
- `--input`: Path to raw video
- `--augmented`: Set to True if to inference on model trained on augmented data
- `--lables`: Labels for prediction, needs to match the training labels
- `--env`: Set to `xvfb` or `xcb` for headless display

## 4. Running Pose Estimation based Classification

### 4.1 Data Preprocessing

We need data preprocessing to fine tune the object detection model. Please run file `process_hgp.py`. This will automatically augment and split the `HGP` dataset into correct directory

```bash
python process_hgp.py
```

### 4.2 Training

The model is trained on Google Colab. You can access the training notebook and saved model in folder `models/yolo phone hand detection`.

### 4.3 Inference

To run inference on a video, you can use the `pose_estimation.py` script:

```bash
# Example command
python pose_estimation.py
```

Parameters:
- `--pose_model`: Path to model for pose_estimation
- `--object_detection_model`: Path to model for object detection
- `--cam_idx`: Camera index
- `--env`: Set to `xvfb` or `xcb` for headless display
- `--history_frames`: Number of frames to keep in history for motion analysis
- `--smoothing_window`: Window size for temporal smoothing

## 5. Additional Notes

- The MoViNet model works best with videos that contain a single dominant action.
- The pose estimation approach can handle multiple people performing different actions simultaneously.