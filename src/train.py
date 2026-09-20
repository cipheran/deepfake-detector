import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import CelebDFDataset
from model import DeepfakeModel
import torch.nn as nn
from tqdm import tqdm

def main():
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    dataset = CelebDFDataset("data/Celeb-DF-v2", transform=transform)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = DeepfakeModel().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    scaler = torch.amp.GradScaler("cuda")

    def loss_fn(pred, target):
        target = target.unsqueeze(1)
        target = target * 0.9 + 0.05  # label smoothing
        return nn.functional.binary_cross_entropy_with_logits(pred, target)

    for epoch in range(20):
        model.train()
        loop = tqdm(train_loader)

        for x, y in loop:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            loop.set_description(f"Epoch {epoch+1}")
            loop.set_postfix(loss=loss.item())

        scheduler.step()

        print("Epoch done")

    torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()