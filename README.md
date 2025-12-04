# Metal Nut Defect Segmentation (MVTec AD)

End-to-end pipeline for multi-class defect segmentation on the MVTec AD **metal_nut** dataset: preprocessing, U-Net training, inference helpers, FastAPI serving, and a simple web UI.

## Labels
- 0: background / normal metal  
- 1: bent  
- 2: color  
- 3: flip  
- 4: scratch  

All training images are treated as defect-free (all-zero masks). Each test image contains at most one defect type, mapped to the class IDs above.

## Project Layout
```
data/
  raw/metal_nut/          # place the original MVTec metal_nut here (train/, test/, ground_truth/)
  processed/metal_nut/    # created by preprocessing
models/                   # saved checkpoints
training/                 # preprocessing, training, inference helpers
api/main.py               # FastAPI server
frontend/                 # static web UI
requirements.txt
README.md
```

## Setup
```bash
python -m venv .venv
. .venv/Scripts/activate  # on PowerShell use: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1) Preprocess the dataset
Place the original dataset at `data/raw/metal_nut` (or point `--raw-dir` to your location, e.g., the existing `metal_nut/` folder in this repo).
```bash
python training/preprocess_metal_nut.py ^
  --raw-dir data/raw/metal_nut ^
  --processed-dir data/processed/metal_nut ^
  --train-ratio 0.6 ^
  --val-ratio 0.2 ^
  --seed 42 ^
  --overwrite
```
Outputs go to `data/processed/metal_nut/images|masks/{train,val,test}` with masks encoded as single-channel PNGs containing class IDs 0–4. The ratios apply per defect folder, so defects are now included in train to allow learning positive pixels.

## 2) Train the U-Net
```bash
python training/train_unet_metalnut.py \
  --data-dir data/processed/metal_nut \
  --image-size 256 \
  --batch-size 4 \
  --epochs 50 \
  --lr 1e-4
```
The best model (by validation loss) is saved to `models/best_unet_metalnut_multiclass.pth` and evaluated on the test split at the end.

## 3) FastAPI inference server
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Endpoints:
- `GET /health` → status and device info
- `POST /predict` → multipart upload field `image`; returns defect flags, per-class percentages, and base64-encoded mask/overlay images.

## 4) Frontend (Vite React UI)
Start the API (above), then run the new frontend:
```bash
cd frontend
npm install
npm run dev -- --host --port 5500
```
Visit `http://localhost:5500` and upload an image. The UI calls the API (`VITE_API_URL`, default derived from the page origin or `http://localhost:8000`), shows the overlay, stats, and lets you download a PDF report (with original + overlay visuals).

## Inference helper (scriptable)
`training/inference.py` exposes `predict_metal_nut_defects` plus helpers to load the model and encode overlays. Use it directly for batch/offline inference if preferred.

## Notes
- Default image size is 256×256 with ImageNet normalization; adjust `--image-size` if you prefer 512×512 (may require more GPU memory).
- Masks are resized with nearest-neighbor to preserve class IDs.
- If you retrain and save a new checkpoint, ensure it lives at `models/best_unet_metalnut_multiclass.pth` (or update `WEIGHTS_PATH` in `api/main.py`).
"# metal_quality_control" 
