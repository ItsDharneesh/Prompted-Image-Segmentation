# Prompted Image Segmentation

## 🎯 Goal

Train (or fine-tune) a **text-conditioned segmentation model** that, given an image and a natural-language prompt, produces a **binary segmentation mask** for:

- **“segment crack”** → Dataset 2 (Cracks)
- **“segment taping area”** → Dataset 1 (Drywall-Join-Detect)

---

## 📂 Datasets

### 🧱 Dataset 1 — Taping Area (Drywall Joints)
Source:  
https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect  

Prompt mappings:
- “segment taping area”
- “segment joint”
- “segment tape”
- “segment drywall seam”

---

### 🪨 Dataset 2 — Cracks
Source:  
https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36  

Prompt mappings:
- “segment crack”
- “segment wall crack”

---

## 🛠 Data Preparation

The datasets were downloaded in **COCO JSON format**.

Bounding box annotations were converted into segmentation masks using:

- `labelled_masks.py`
- `new_masks.py`

These scripts extract mask information from bounding box data and generate binary segmentation masks used for training.

---

## 🧠 Models & Experiments

### 1️⃣ HRNet-Based Experiments

Training script:
- `HRNet.py`

Prediction script:
- `predictions_HRNet.py`

Experiment result folders:
- `training_hrnet_prompted/`
- `training_hrnet_prompted_01/`

These folders contain results from multiple HRNet training trials.

---

### 2️⃣ SegFormer-Based Experiments

Training scripts:
- `segformer.py`
- `segformer_tvloss.py`

Prediction script:
- `predictions_SegFormer.py`

Experiment result folders:
- `training_segformer_prompted/`
- `training_segformer_prompted_01/`
- `training_segformer_prompted_02/`

Final optimized version:
- `training_segformer_prompted_final/` ✅

The **final satisfactory results** were obtained from the SegFormer architecture (`training_segformer_prompted_final`).

---

## 📊 Predictions

After training:

- Saved models were used to generate segmentation masks.
- Prediction scripts load trained weights and generate binary masks.
- Generated masks are stored inside the `predictions/` folder.

---

## 💾 Trained Models & Outputs

All trained models (.pth files) and final prediction outputs are stored externally due to GitHub size limits:

Google Drive Link:  
https://drive.google.com/drive/folders/1_5igPYyJcyaHVtK6BSqeseuEJdWyZMH4?usp=sharing

---

## 🗂 Project Structure
```text
├── HRNet.py
├── segformer.py
├── segformer_tvloss.py
├── predictions_HRNet.py
├── predictions_SegFormer.py
├── labelled_masks.py
├── new_masks.py
│
├── training_hrnet_prompted/
├── training_hrnet_prompted_01/
│
├── training_segformer_prompted/
├── training_segformer_prompted_01/
├── training_segformer_prompted_02/
├── training_segformer_prompted_final/
│
├── predictions/
└── Prompted_Segmentation_for_Drywall_OA-1.pdf
```


---

## 🚀 Workflow

1. Download dataset (COCO format)
2. Convert bounding boxes → binary masks (`labelled_masks.py`, `new_masks.py`)
3. Train model (HRNet or SegFormer)
4. Save trained weights
5. Run prediction script to generate segmentation masks

---

## 🏆 Final Outcome

- Successfully fine-tuned text-conditioned segmentation models
- Generated binary masks from natural language prompts
- Best performance achieved using **SegFormer (final configuration)**

---

## 📌 Notes

- Large files (datasets, trained models, predictions) are excluded from this repository.
- Refer to the Google Drive link for full models and outputs.
- Full project assignment repoert is available in the included PDF "Report.pdf".

---

