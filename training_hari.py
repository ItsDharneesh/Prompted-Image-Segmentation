import os
import torch
import clip
from tqdm import tqdm
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn as nn
import torch.optim as optim
from torchvision.ops import sigmoid_focal_loss
from torchvision.transforms import InterpolationMode
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_GPUS = torch.cuda.device_count()
print("GPUs available:", NUM_GPUS)

CONFIG = {
    "img_size": 768,
    "batch_size": 8,
    "epochs": 30,
    "lr": 1e-4,
    "weight_decay": 5e-3,
    "num_workers": 4,
    "drop_out": 0.3,
    "early_stop_patience": 4,
}

DATASETS = [
    "Data/cracks_processed",
    "Data/drywall_processed"
]

OUT_DIR = "training_hrnet_prompted"
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# DATASET
# =====================================================

class PromptSegDataset(DATASETS):
    def __init__(self, split):
        self.samples = []

        for dataset_root in DATASETS:
            img_dir = os.path.join(dataset_root, split, "images")
            mask_dir = os.path.join(dataset_root, split, "masks")

            for fname in sorted(os.listdir(img_dir)):
                img_path = os.path.join(img_dir, fname)
                mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))

                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        img = T.Resize((CONFIG["img_size"], CONFIG["img_size"]))(img)
        mask = T.Resize(
            (CONFIG["img_size"], CONFIG["img_size"]),
            interpolation=InterpolationMode.NEAREST
        )(mask)

        img = T.ToTensor()(img)
        mask = T.ToTensor()(mask)
        mask = (mask > 0).float()

        if "cracks_processed" in img_path:
            prompt_id = torch.tensor(0)
        else:
            prompt_id = torch.tensor(1)

        return img, mask, prompt_id


# =====================================================
# MODEL
# =====================================================

class PromptedHRNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            "hrnet_w18",
            pretrained=True,
            features_only=True
        )

        backbone_out = self.backbone.feature_info.channels()[-1]

        self.seg_head = nn.Sequential(
            nn.Conv2d(backbone_out, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(CONFIG["drop_out"]),
            nn.Conv2d(256, 1, 1)
        )

        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(512, backbone_out)
        self.prompt_cache = {}

    def forward(self, x, prompt_ids):

        device = x.device
        text_embeddings = []

        for pid in prompt_ids:
            pid = pid.item()

            if pid not in self.prompt_cache:
                text = "segment crack" if pid == 0 else "segment taping area"
                with torch.no_grad():
                    tokens = clip.tokenize([text]).to(device)
                    feat = self.clip_model.encode_text(tokens).float()[0].cpu()
                self.prompt_cache[pid] = feat

            text_embeddings.append(self.prompt_cache[pid].to(device))

        text_features = torch.stack(text_embeddings)
        text_features = self.text_proj(text_features)
        text_features = text_features.unsqueeze(-1).unsqueeze(-1)

        features = self.backbone(x)[-1]
        features = features + text_features

        out = self.seg_head(features)

        out = nn.functional.interpolate(
            out,
            size=(CONFIG["img_size"], CONFIG["img_size"]),
            mode="bilinear",
            align_corners=False
        )

        return out


# =====================================================
# METRICS
# =====================================================

def compute_metrics(pred, gt):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = intersection / (union + 1e-6)
    dice = (2 * intersection) / (pred.sum() + gt.sum() + 1e-6)
    return iou.item(), dice.item()


def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


def boundary_loss(pred, target):
    pred = torch.sigmoid(pred)

    sobel_x = torch.tensor(
        [[1,0,-1],[2,0,-2],[1,0,-1]],
        dtype=torch.float32,
        device=pred.device
    ).view(1,1,3,3)

    sobel_y = sobel_x.permute(0,1,3,2)

    pred_edge = torch.abs(nn.functional.conv2d(pred, sobel_x, padding=1)) + \
                torch.abs(nn.functional.conv2d(pred, sobel_y, padding=1))

    target_edge = torch.abs(nn.functional.conv2d(target, sobel_x, padding=1)) + \
                  torch.abs(nn.functional.conv2d(target, sobel_y, padding=1))

    return nn.functional.l1_loss(pred_edge, target_edge)


def evaluate(loader, model):
    model.eval()

    crack_iou, crack_dice = [], []
    tap_iou, tap_dice = [], []

    with torch.no_grad():
        for imgs, masks, prompt_ids in loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            prompt_ids = prompt_ids.to(DEVICE)

            preds = model(imgs, prompt_ids)

            for i in range(len(prompt_ids)):
                iou, dice = compute_metrics(preds[i], masks[i])
                if prompt_ids[i].item() == 0:
                    crack_iou.append(iou)
                    crack_dice.append(dice)
                else:
                    tap_iou.append(iou)
                    tap_dice.append(dice)

    return (np.mean(crack_iou), np.mean(crack_dice),
            np.mean(tap_iou), np.mean(tap_dice))


# =====================================================
# TRAIN
# =====================================================

def train():

    train_dataset = PromptSegDataset("train")
    val_dataset = PromptSegDataset("valid")

    print("Total Train Samples:", len(train_dataset))
    print("Total Val Samples:", len(val_dataset))

    crack_count = 0
    tap_count = 0
    for img_path, _ in train_dataset.samples:
        if "cracks_processed" in img_path:
            crack_count += 1
        else:
            tap_count += 1

    tap_weight = crack_count / max(tap_count, 1)
    print("Tap weight:", tap_weight)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
        pin_memory=True
    )

    model = PromptedHRNet()

    if NUM_GPUS > 1:
        model = nn.DataParallel(model)

    model = model.to(DEVICE)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"]
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=5, gamma=0.5
    )

    best_val_dice = 0
    early_stop_counter = 0

    for epoch in range(CONFIG["epochs"]):

        model.train()

        for imgs, masks, prompt_ids in tqdm(train_loader):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)
            prompt_ids = prompt_ids.to(DEVICE)

            preds = model(imgs, prompt_ids)

            # ===== CORRECTED WEIGHTING =====
            sample_weights = torch.ones(len(prompt_ids), device=DEVICE)
            sample_weights[prompt_ids == 1] = tap_weight
            sample_weights = sample_weights.view(-1,1,1,1)

            focal = sigmoid_focal_loss(
                preds, masks,
                alpha=0.75,
                gamma=2,
                reduction="none"
            )

            focal = focal * sample_weights
            focal = focal.mean()
            # ===============================

            loss = focal + dice_loss(preds, masks) + 0.1 * boundary_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        train_ci, train_cd, train_ti, train_td = evaluate(train_loader, model)
        val_ci, val_cd, val_ti, val_td = evaluate(val_loader, model)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Crack IoU: {train_ci:.4f}")
        print(f"Train Crack Dice: {train_cd:.4f}")
        print(f"Train Tap IoU: {train_ti:.4f}")
        print(f"Train Tap Dice: {train_td:.4f}")
        print(f"Val Crack IoU: {val_ci:.4f}")
        print(f"Val Crack Dice: {val_cd:.4f}")
        print(f"Val Tap IoU: {val_ti:.4f}")
        print(f"Val Tap Dice: {val_td:.4f}")

        val_mean_dice = (val_cd + val_td) / 2

        if val_mean_dice > best_val_dice:
            best_val_dice = val_mean_dice
            early_stop_counter = 0
            torch.save(model.state_dict(),
                       os.path.join(OUT_DIR, "best_model.pth"))
            print("Saved Best Model")
        else:
            early_stop_counter += 1
            if early_stop_counter >= CONFIG["early_stop_patience"]:
                print("Early stopping triggered.")
                break

    print("Training complete.")


if __name__ == "__main__":
    train()