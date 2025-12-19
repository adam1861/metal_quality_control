## Step-by-step guide

1) Create and activate a virtual environment  
```bash
python -m venv .venv
.\.venv\Scripts\activate.bat  # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2) Place the raw dataset  
- Put the original `metal_nut` MVTec folder at `data/raw/metal_nut/`  
  (or point `--raw-dir` to the existing `metal_nut/` in this repo).

3) Preprocess to build train/val/test masks (now moves a portion of defects into train)  
```bash
python training/preprocess_metal_nut.py ^
  --raw-dir data/raw/metal_nut ^
  --processed-dir data/processed/metal_nut ^
  --train-ratio 0.6 ^
  --val-ratio 0.2 ^
  --seed 42 ^
  --overwrite
```

4) Train the U-Net (saves best weights to models/)  
```bash
python training/train_unet_metalnut.py \
  --data-dir data/processed/metal_nut \
  --image-size 256 \
  --batch-size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --augment \
  --balance-sampler \
  --weights-path models/best_unet_metalnut_colorscratch.pth
```
This creates a 3-class model (background, color, scratch) and saves weights to `models/best_unet_metalnut_colorscratch.pth`.

5) Launch the FastAPI server  
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
- Health check: `curl http://localhost:8000/health`
- Predict: POST multipart field `image` to `/predict`.

6) Open the web UI (new React/Vite frontend)  
```bash
cd frontend
npm install
npm run dev -- --host --port 5500
```
- Visit `http://localhost:5500` and upload an image; it will call the API (uses `VITE_API_URL` if set, otherwise maps the frontend origin to port 8000) and offers PDF download with overlay/original embedded.
  - To point the frontend elsewhere, set `VITE_API_URL` (e.g., `http://your-host:8000`) before `npm run dev`.

7) Optional: reuse inference helper in scripts  
- See `training/inference.py` for `predict_metal_nut_defects` to batch images offline.












backend:

uvicorn api.main:app --host 0.0.0.0 --port 8000

frontend:

cd frontend
npm install
npm run dev -- --host --port 5500
