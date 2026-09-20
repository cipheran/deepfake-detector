import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class CelebDFDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, transform=None):
        self.samples = []
        self.num_frames = num_frames
        self.transform = transform

        for label, folder in enumerate(["real", "fake"]):
            path = os.path.join(root_dir, folder)
            for f in os.listdir(path):
                self.samples.append((os.path.join(path, f), label))

    def extract_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 🔥 hybrid sampling
        uniform = np.linspace(0, max(total-1, 1), self.num_frames//2)
        random_idx = np.random.choice(range(max(total, self.num_frames)), self.num_frames//2)

        indices = np.concatenate([uniform, random_idx]).astype(int)
        indices = np.sort(indices)

        frames = []

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((224,224,3), dtype=np.uint8)] * self.num_frames
        else:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        return frames

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self.extract_frames(path)

        if self.transform:
            frames = [self.transform(f) for f in frames]

        return torch.stack(frames), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)