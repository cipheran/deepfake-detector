import torch
import cv2
import numpy as np
from torchvision import transforms
from model import DeepfakeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD MODEL
model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# FRAME EXTRACTION
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # hybrid sampling
    uniform = np.linspace(0, max(total-1, 1), num_frames//2)
    random_idx = np.random.choice(range(max(total, num_frames)), num_frames//2)

    indices = np.concatenate([uniform, random_idx]).astype(int)
    indices = np.sort(indices)

    frames = []

    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224))
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(3, 224, 224)] * num_frames
    else:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    return torch.stack(frames)

# 🔥 FINAL PREDICTION FUNCTION
def predict(video_path, num_samples=5):
    probs = []

    for _ in range(num_samples):
        frames = extract_frames(video_path).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(frames)
            prob = torch.sigmoid(output).item()
            probs.append(prob)

    final_prob = np.mean(probs)

    # 🔥 calibrated threshold
    if final_prob > 0.6:
        label = "FAKE"
    else:
        label = "REAL"

    return {
        "label": label,
        "confidence": float(final_prob),
        "all_probs": probs
    }


# TEST RUN
if __name__ == "__main__":
    video_path = "test.mp4"
    result = predict(video_path)

    print("Prediction:", result["label"])
    print("Confidence:", result["confidence"])