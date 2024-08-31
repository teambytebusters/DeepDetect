import os
import cv2
import h5py
import numpy as np
from tqdm import tqdm
import json
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from DeepfakeDetection import logger
from DeepfakeDetection.entity.config_entity import DataPreprocessingConfig
from DeepfakeDetection.utils.common import create_directories, save_h5py

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config
        create_directories([self.config.output_dir])

    def preprocess_frame(self, frame, target_size):
        """Preprocess a single frame: convert to RGB, resize, and normalize."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(target_size)
        frame = np.array(frame) / 255.0  # Normalize to [0, 1]
        return frame

    def extract_faces_from_frame(self, frame):
        """Extract faces from a single frame using Haar cascades."""
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=self.config.scale_factor, minNeighbors=self.config.min_neighbors, minSize=tuple(self.config.min_size))

        face_crops = []
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_crops.append(face)
        
        return face_crops

    def process_video(self, video_path, label, max_frames):
        """Process a video: extract frames and faces, preprocess them."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(frame_count // max_frames, 1)

        processed_frames = []
        
        for i in range(0, frame_count, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = self.extract_faces_from_frame(frame)
            
            if faces:
                processed_frame = self.preprocess_frame(faces[0], tuple(self.config.target_size))
            else:
                processed_frame = self.preprocess_frame(frame, tuple(self.config.target_size))
            
            processed_frames.append(processed_frame)
            
            if len(processed_frames) >= max_frames:
                break

        cap.release()
        return processed_frames

    def save_dataset(self, data, labels, split_name):
        """Save the processed data and labels to HDF5 format."""
        split_dir = os.path.join(self.config.output_dir, split_name)
        data_dir = os.path.join(split_dir, "data")
        labels_dir = os.path.join(split_dir, "labels")

        create_directories([split_dir, data_dir, labels_dir])
        
        data_file = os.path.join(data_dir, "data.h5")
        labels_file = os.path.join(labels_dir, "labels.h5")

        # Save data and labels using h5py utility functions
        save_h5py(data, Path(data_file), dataset_name="data")
        save_h5py(labels, Path(labels_file), dataset_name="labels")

    def execute(self):
        """Execute the preprocessing pipeline: process videos and save datasets."""
        logger.info("Starting data preprocessing...")

        # Load metadata from the JSON file
        with open(os.path.join(self.config.data_path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        # Split video files into train, validation, and test sets
        video_files = list(metadata.keys())
        train_files, test_files = train_test_split(video_files, test_size=0.2, random_state=42)
        train_files, val_files = train_test_split(train_files, test_size=0.1, random_state=42)

        splits = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        # Process and save each split
        for split_name, split_files in splits.items():
            data, labels = [], []

            for video_file in tqdm(split_files, desc=f"Processing {split_name} data"):
                video_path = os.path.join(self.config.data_path, video_file)
                label = metadata[video_file]["label"]

                frames = self.process_video(video_path, label, self.config.max_frames)

                for frame in frames:
                    data.append(frame)
                    labels.append(0 if label == "REAL" else 1)

            # Save processed data and labels for the current split
            self.save_dataset(data, labels, split_name)

        logger.info("Data preprocessing completed successfully.")
