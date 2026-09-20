import torch
import torch.nn as nn
import timm

class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()

        # CNN
        self.cnn = timm.create_model('efficientnet_b0', pretrained=True)
        self.cnn.classifier = nn.Identity()

        # ViT
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        self.vit.head = nn.Identity()

        self.feature_dim = 1280 + 768

        # LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=512,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Attention
        self.attn = nn.MultiheadAttention(
            embed_dim=1024,
            num_heads=8,
            batch_first=True
        )

        self.norm = nn.LayerNorm(1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
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