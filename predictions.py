import os
import time
import torch
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
from transformers import SegformerForSemanticSegmentation

# ================= CONFIG =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    "img_size": 768,
}

CHECKPOINT_PATH = "training_segformer_prompted_final/best_model.pth"

CRACKS_TEST_DIR = "Data/cracks_processed/test/images"
DRYWALL_TEST_DIR = "Data/drywall_processed/test/images"

SAVE_ROOT = "predictions"
CRACKS_SAVE_DIR = os.path.join(SAVE_ROOT, "cracks")
DRYWALL_SAVE_DIR = os.path.join(SAVE_ROOT, "drywall")

VALID_EXT = [".jpg", ".jpeg", ".png"]

os.makedirs(CRACKS_SAVE_DIR, exist_ok=True)
os.makedirs(DRYWALL_SAVE_DIR, exist_ok=True)

# ================= MODEL =================
class PromptedSegFormer(nn.Module):
    def __init__(self):
        super().__init__()

        # SAME backbone you trained with
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            num_labels=1,
            ignore_mismatched_sizes=True
        )

        hidden_dim = self.segformer.config.hidden_sizes[-1]

        self.clip_model, _ = clip.load("ViT-B/32", device=DEVICE)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        self.text_proj = nn.Linear(512, hidden_dim)
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

        # EXACT SAME FORWARD AS TRAINING
        outputs = self.segformer.segformer(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True
        )

        hidden_states = list(outputs.hidden_states)

        hidden_states[-1] = hidden_states[-1] + text_features

        logits = self.segformer.decode_head(hidden_states)

        logits = F.interpolate(
            logits,
            size=(CONFIG["img_size"], CONFIG["img_size"]),
            mode="nearest"
        )

        return logits

# ================= LOAD MODEL =================
model = PromptedSegFormer().to(DEVICE)

state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

# handle DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k[7:]] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict)
model.eval()

transform = T.Compose([
    T.Resize((CONFIG["img_size"], CONFIG["img_size"])),
    T.ToTensor()
])

# ================= INFERENCE FUNCTION =================
def run_inference(images_dir, save_dir, prompt_id):

    total_time = 0
    num_images = 0

    image_files = [
        f for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in VALID_EXT
    ]

    for img_name in tqdm(image_files):

        base_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        prompt_ids = torch.tensor([prompt_id]).to(DEVICE)

        start = time.time()
        with torch.no_grad():
            output = model(image_tensor, prompt_ids)
            pred = torch.sigmoid(output)
        end = time.time()

        total_time += (end - start)
        num_images += 1

        pred_np = pred.squeeze().cpu().numpy()

        pred_resized = Image.fromarray((pred_np * 255).astype(np.uint8))
        pred_resized = pred_resized.resize(image.size, Image.NEAREST)

        pred_bin = (np.array(pred_resized) > 127).astype(np.uint8)

        save_name = f"{base_id}__pred.png"
        Image.fromarray(pred_bin * 255).save(
            os.path.join(save_dir, save_name)
        )

    print(f"\nDone: {images_dir}")
    print("Avg inference time per image:", total_time / max(1, num_images))


# ================= RUN BOTH DATASETS =================
print("\nRunning CRACKS predictions...")
run_inference(CRACKS_TEST_DIR, CRACKS_SAVE_DIR, prompt_id=0)

print("\nRunning DRYWALL predictions...")
run_inference(DRYWALL_TEST_DIR, DRYWALL_SAVE_DIR, prompt_id=1)

print("\nAll predictions complete.")