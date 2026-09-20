import os
import uuid
import torch
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from torchvision import transforms
import torch.nn as nn
import timm
import tempfile

app = Flask(__name__, static_folder=".")

# ─── Model Definition (same as model.py) ────────────────────────────────────

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = timm.create_model('efficientnet_b0', pretrained=False)
        self.cnn.classifier = nn.Identity()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        self.vit.head = nn.Identity()
        self.feature_dim = 1280 + 768
        self.lstm = nn.LSTM(
            input_size=self.feature_dim, hidden_size=512,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(1024)
        self.fc = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = x[:, t]
            cnn_f = self.cnn(f)
            vit_f = self.vit(f)
            feats.append(torch.cat([cnn_f, vit_f], dim=1))
        feats = torch.stack(feats, dim=1)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        lstm_out, _ = self.lstm(feats)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        out = self.norm(attn_out.mean(dim=1))
        return self.fc(out)

# ─── Load Model ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

model = DeepfakeModel().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("[INFO] Model loaded successfully.")

# ─── Transform ──────────────────────────────────────────────────────────────

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ─── Frame Extraction ───────────────────────────────────────────────────────

def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    uniform = np.linspace(0, max(total - 1, 1), num_frames // 2)
    random_idx = np.random.choice(range(max(total, num_frames)), num_frames // 2)
    indices = np.sort(np.concatenate([uniform, random_idx]).astype(int))

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

# ─── Prediction ─────────────────────────────────────────────────────────────

def predict(video_path, num_samples=5):
    probs = []
    for _ in range(num_samples):
        frames = extract_frames(video_path).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(frames)
            prob = torch.sigmoid(output).item()
            probs.append(prob)

    final_prob = float(np.mean(probs))
    label = "FAKE" if final_prob > 0.6 else "REAL"
    return {"label": label, "confidence": final_prob, "all_probs": probs}

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400
    
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")
    file.save(tmp_path)

    try:
        result = predict(tmp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    app.run(
    debug=False,
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 7860))
)
