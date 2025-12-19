# Metal Quality Control — Technical Overview (Languages + Model)

This document explains:

1) which programming languages are used in this project (and where), and  
2) what deep learning model is used to identify defects, plus why it fits the problem.

---

## 1) What the app does (one sentence)

You upload an image of a **metal nut**, and the system returns:

- a **pixel-by-pixel defect map** (segmentation mask),
- a colored **overlay** showing where the defect is,
- **percentages** per defect class, and an overall defect ratio.

The dataset and labels are based on MVTec AD `metal_nut`:

- `0` = background/normal metal
- `1` = color
- `2` = scratch

Code sources: `README.md`, `training/inference.py`, `api/main.py`.

---

## 2) Programming languages used (and where)

### Python (backend + ML training/inference)

Python is used for:

- **Model training** and evaluation: `training/train_unet_metalnut.py`
- **Dataset preprocessing** (creating masks and train/val/test splits): `training/preprocess_metal_nut.py`
- **Model definition** (U‑Net): `training/model.py`
- **Inference utilities** (preprocess image, run model, build overlay): `training/inference.py`
- **API server** that exposes inference endpoints: `api/main.py` (FastAPI + Uvicorn)

Main Python libraries (from `requirements.txt`):

- `torch`, `torchvision` (deep learning + image preprocessing)
- `numpy`, `pillow` (array + image handling)
- `fastapi`, `uvicorn`, `python-multipart` (web API + file upload)

### TypeScript / TSX (frontend web app)

TypeScript is used for the browser UI:

- React app entry: `frontend/src/main.tsx`
- Main UI + API calls: `frontend/src/App.tsx`
- UI components: `frontend/src/components/*.tsx`

Frontend tooling:

- Vite dev server/bundler: `frontend/vite.config.ts`
- npm dependencies: `frontend/package.json`

### HTML + CSS (frontend layout + styling)

- HTML entry point: `frontend/index.html`
- Styles: `frontend/src/index.css` (generated Tailwind CSS output) and `frontend/src/styles/*.css`

---

## 3) What model is used to identify the defect?

### The model: U‑Net (multi-class semantic segmentation)

This project uses a **U‑Net** implemented in PyTorch (`training/model.py`) to do **semantic segmentation**.

That means:

- Input: an RGB image
- Output: for **every pixel**, the model predicts one of the 3 class IDs (0–2).

U‑Net is not a “single yes/no classifier”; it produces a **mask** the same width/height as the input (after resizing).

### What the U‑Net looks like (as implemented here)

The code uses a lightweight U‑Net:

- **Encoder (“down path”)**
  - Repeated blocks of:
    - `MaxPool2d(2)` (shrinks image by 2×)
    - `DoubleConv` (two 3×3 convs + BatchNorm + ReLU)
- **Bottleneck**
  - One more downsampling block at the deepest resolution
- **Decoder (“up path”)**
  - Upsampling by bilinear interpolation (`F.interpolate`)
  - Concatenate skip-connection features from the encoder (keeps detail)
  - Another `DoubleConv`
- **Output head**
  - A final `1×1` convolution (`nn.Conv2d`) producing `num_classes=3` channels (logits)

Key point: skip connections are what let it keep fine details like thin scratches.

---

## 4) Why this model is suitable (in plain terms)

You want the system to answer:

1) **Is there a defect?**  
2) **Where exactly is it on the nut?**  
3) **What type is it (color/scratch)?**

A U‑Net is suitable because:

- **It finds the location**: it labels pixels, so you get a “heatmap/mask” of the defect region, not just a yes/no answer.
- **It keeps detail**: the skip connections help preserve edges and small regions (important for scratches).
- **It’s efficient**: U‑Net is a strong baseline that trains and runs fast compared to heavier segmentation models.
- **It’s interpretable**: the overlay image is easy to understand for quality control (operators can see *why* it flagged a defect).

---

## 5) How the model is trained (data → masks → weights)

### Step A — Preprocessing: creating the training dataset

MVTec `metal_nut` ships as:

- `train/good/*` (only normal examples)
- `test/<defect_type>/*` (defect examples)
- `ground_truth/<defect_type>/*_mask.png` (pixel masks for defects)

`training/preprocess_metal_nut.py` creates a supervised segmentation dataset by:

1) Copying `train/good` images into the new `train` split and generating **all‑zero masks** (class 0).
2) Taking images from `test/*`, shuffling them per defect type, and splitting them into:
   - `train` (a fraction of defects is moved into train),
   - `val`,
   - `test`.
3) For defect images, it builds a single-channel label mask:
   - start with zeros,
   - read the ground truth mask,
   - write the class ID into pixels where ground truth is non-zero.

Output folders (created under `data/processed/metal_nut`):

- `images/train`, `images/val`, `images/test`
- `masks/train`, `masks/val`, `masks/test`

Each mask is a grayscale PNG where pixel values are `{0,1,2}`.

### Step B — Training: learning to predict class IDs

`training/train_unet_metalnut.py` trains the U‑Net using:

- Loss: `nn.CrossEntropyLoss()` (standard for multi-class segmentation with integer labels)
- Optimizer: Adam
- Metrics: pixel accuracy + IoU (intersection-over-union) per class and mean IoU
- Best checkpoint selection: saves the weights with the lowest validation loss to:
  - `models/best_unet_metalnut_colorscratch.pth`

That `.pth` file is a PyTorch `state_dict` (the learned weights).

---

## 6) How prediction works (image → overlay + percentages)

Prediction happens in `training/inference.py` and is called by the API.

### Step 1 — Prepare the image

- Convert to RGB
- Resize to `256×256` (default)
- Convert to tensor
- Normalize using ImageNet mean/std (same as training)

### Step 2 — Run the model

- The model outputs **logits**: shape `[1, num_classes, H, W]`
- `argmax` chooses the most likely class per pixel → predicted mask `[H, W]`

### Step 3 — Make results usable

The code then:

- Resizes the predicted mask back to the **original image size** (nearest neighbor so class IDs stay intact)
- Computes coverage metrics:
  - `defect_ratio_on_nut` (aka `defect_ratio` in the API response) = defect pixels **within the nut mask** / nut pixels
  - `defect_ratio_image` = defect pixels / total image pixels (includes background, mainly useful as a secondary/debug metric)
  - per-class pixel percentages (reported **on-nut** in the API response)
- Builds a colored RGBA overlay (using `CLASS_COLOR_MAP`) and alpha-blends it onto the original image
- Encodes the mask/overlay as base64 strings so the browser can display them directly

---

## 7) Backend API (FastAPI) — what endpoints exist

All backend code is in `api/main.py`.

### Inference endpoints

- `GET /health`
  - returns whether the model weights loaded and which device is used (CPU/GPU)
- `POST /predict`
  - input: multipart form upload field named `image`
  - output: JSON containing:
    - `is_defective`, `defect_ratio` (on-nut), and `defect_ratio_image` (whole-image)
    - `class_pixel_percentages`
    - `dominant_defect` and `dominant_defect_ratio`
    - `mask_encoded` and `overlay_image_encoded` (base64 `data:image/png;base64,...`)
  - note: there is no auth layer; `/predict` is publicly accessible by default.

## 8) Frontend — how the UI connects everything

The UI lives in `frontend/` and is a Vite + React app.

What it does:

- Lets the user upload an image (drag-drop or file picker).
- Sends the file to the API `/predict` endpoint.
- Displays:
  - original image,
  - overlay mask,
  - defect percentages and dominant defect type.
- Generates a simple PDF report in the browser using `jspdf`.

How it finds the API URL (`frontend/src/App.tsx`):

- If `VITE_API_URL` is set: uses that.
- Else, if frontend is served on port `5500`, it assumes API is on `8000`.
- Else, it tries same origin with `/predict`.
- Fallback: `http://localhost:8000/predict`

---

## 9) If you want to change the model or behavior

Common knobs:

- Change image size:
  - API: `IMAGE_SIZE` in `api/main.py`
  - Training: `--image-size` in `training/train_unet_metalnut.py`
- Change number of classes / labels:
  - `CLASS_ID_TO_NAME` + `CLASS_COLOR_MAP` in `training/inference.py`
  - `NUM_CLASSES` in `training/train_unet_metalnut.py`
  - `num_classes` passed into `load_model()` in `api/main.py`
- Replace U‑Net with another segmentation model:
  - swap `training/model.py` and keep the same training/inference contract (logits → argmax → mask)

---

## 10) Entry points and execution flow (what runs what)

### Main entry points

- Preprocess raw dataset → supervised segmentation dataset:
  - `training/preprocess_metal_nut.py` (CLI script)
- Train model weights:
  - `training/train_unet_metalnut.py` (CLI script)
- Serve predictions as an API:
  - `api/main.py` (FastAPI app, typically run via `uvicorn api.main:app ...`)
- Run the web UI:
  - `frontend/src/main.tsx` (Vite + React entrypoint that renders `frontend/src/App.tsx`)

### High-level call graph

`frontend` → `POST /predict` (`api/main.py`) → `training/inference.predict_metal_nut_defects()` → `training/data_utils.preprocess_image()` + `training/model.UNet`

---

## 11) File-by-file guide (what each file is for)

This is a “map” of the repository. For datasets/build output, files are grouped (otherwise there are thousands of images).

### Root files

- `.gitignore` — ignores Python caches/venvs, Node `node_modules/`, and build artifacts (so things like `frontend/build/` are typically not committed).
- `README.md` — main documentation: setup, preprocessing, training, API, and frontend usage.
- `STEP_BY_STEP.md` — condensed checklist version of `README.md`.
- `PROJECT_TECHNICAL_OVERVIEW.md` — model + architecture overview (this document).
- `requirements.txt` — Python dependency list for training + API.
- `tasks` — a tiny TODO note (currently: “add the scroll bar to the pages…”).

### Backend API (`api/`)

- `api/main.py` — FastAPI server that:
  - Loads the trained U‑Net weights on startup (`WEIGHTS_PATH`) via `training/inference.load_model`.
  - Exposes endpoints:
    - `GET /health` (device + model_loaded)
    - `POST /predict` (multipart upload field `image`)

### ML / training / inference (`training/`)

- `training/data_utils.py`
  - Purpose: shared preprocessing and dataset loading for training + inference.
  - Key items:
    - `preprocess_image(image, image_size, normalize)` — resize + tensor conversion + optional ImageNet normalization.
    - `MetalNutSegmentationDataset` — PyTorch `Dataset` yielding `(image_tensor, mask_tensor, filename)`.
    - `IMAGENET_MEAN`, `IMAGENET_STD` — normalization constants.
- `training/model.py`
  - Purpose: the PyTorch U‑Net architecture used for multi-class segmentation.
  - Key classes:
    - `DoubleConv`, `Down`, `Up` — building blocks.
    - `UNet` — full model; outputs per-class logits shaped `[B, 3, H, W]`.
- `training/inference.py`
  - Purpose: load weights and run single-image inference, returning both mask + overlay visuals.
  - Key items:
    - `CLASS_ID_TO_NAME` and `CLASS_COLOR_MAP` — label names + RGB colors.
    - `load_model(weights_path, device, num_classes=3)` — loads `state_dict` into `UNet`, sets `eval()`.
    - `predict_metal_nut_defects(model, image_input, device, image_size)` — returns:
      - `resized_mask` (`np.ndarray` of class IDs, resized back to original size),
      - `original_image` (`PIL.Image`),
      - `overlay_image` (`PIL.Image` RGBA composited),
      - `per_class_pixel_counts` (`dict[int,int]`),
      - `defect_ratio` (non-background pixels / total pixels; the API also computes an on-nut metric via `segment_nut_mask(...)`).
    - `mask_to_color(mask, color_map)` and `pil_to_base64(image)` — helpers used by the API response.
- `training/preprocess_metal_nut.py`
  - Purpose: convert raw MVTec `metal_nut` folder structure into a supervised segmentation dataset under `data/processed/metal_nut`.
  - Key behavior:
    - Copies `train/good/*` into processed `train/` and generates all-zero masks.
    - Splits `test/<type>/*` into processed `train/val/test` according to `--train-ratio` and `--val-ratio` (per folder).
    - Converts `ground_truth/<type>/*_mask.png` into single-channel masks whose pixel values are class IDs.
    - Prefixes filenames with the defect type (e.g., `color_012.png`, `scratch_012.png`) so you can trace the source class later.
    - Good samples are kept distinct across sources: `good_train_*.png` (from `train/good`) and `good_test_*.png` (from `test/good`).
  - Key items:
    - `DEFECT_CLASS_MAP` (color/scratch → 1–2)
    - `create_mask(...)`, `split_test_images(...)`, `process_dataset(...)`
- `training/train_unet_metalnut.py`
  - Purpose: train the U‑Net on the processed dataset and save `models/best_unet_metalnut_colorscratch.pth`.
  - Key items:
    - `train_one_epoch(...)`, `evaluate(...)` (pixel accuracy + per-class IoU + mean IoU)
    - CLI args include `--data-dir`, `--image-size`, `--epochs`, `--batch-size`, `--lr`, `--device`, `--weights-path`.

### Frontend web app (`frontend/`)

- `frontend/README.md` — minimal “how to run” for the Vite UI (points back to the original Figma design link).
- `frontend/package.json` — frontend dependencies + scripts (`npm run dev`, `npm run build`).
- `frontend/package-lock.json` — exact npm dependency tree pinned for reproducible installs.
- `frontend/vite.config.ts`
  - Sets `build.outDir` to `frontend/build`.
  - Contains many `resolve.alias` entries to support imports like `react@18.3.1` and `@radix-ui/react-dialog@1.1.6` (this is a Figma-export pattern).
  - Default dev server is `port: 3000` (the README runs with `--port 5500` to match the API URL heuristic in `App.tsx` unless you set `VITE_API_URL`).
- `frontend/index.html` — HTML shell with `<div id="root"></div>` and the module script loading `src/main.tsx`.
- `frontend/build/` — Vite build output (generated by `npm run build`).

#### Frontend source (`frontend/src/`)

- `frontend/src/main.tsx` — bootstraps React and renders the `App` component into `#root`.
- `frontend/src/App.tsx`
  - Main UI “router” (single-page state machine) with sections: `dashboard`, `upload`, `results`, `about`.
  - Defines `PredictionResult` (shape of `/predict` response the UI expects).
  - Builds `API_URL`:
    - Uses `VITE_API_URL` if provided.
    - Otherwise, if running on port `5500`, rewrites to port `8000`.
    - Otherwise, uses same origin (`${url.origin}/predict`).
  - Calls the API via `fetch()` with `FormData` field `image`.
- `frontend/src/index.css` — prebuilt Tailwind CSS bundle (Tailwind v4 output) used by the UI.
- `frontend/src/styles/globals.css` — theme variables + Tailwind `@apply` rules + custom keyframes (currently not imported by `main.tsx`).
- `frontend/src/Attributions.md` — attribution notes (shadcn/ui + Unsplash).
- `frontend/src/guidelines/Guidelines.md` — placeholder template for AI/editor guidelines.

#### UI components (`frontend/src/components/`)

- `frontend/src/components/Header.tsx` — sticky top navigation + dark mode toggle + account icon.
- `frontend/src/components/HeroSection.tsx` — landing “Dashboard” hero and CTA.
- `frontend/src/components/UploadPanel.tsx` — drag/drop upload + “Analyze” button + progress indicator.
- `frontend/src/components/ResultsDashboard.tsx` — shows original + overlay and per-class bars.
- `frontend/src/components/PDFReportSection.tsx` — uses `jspdf` to generate a downloadable PDF report.
- `frontend/src/components/Footer.tsx` — footer with placeholder links.
- `frontend/src/components/figma/ImageWithFallback.tsx` — image component that swaps to an inline SVG placeholder on error.

#### Component library (`frontend/src/components/ui/`)

These are shadcn/ui-style wrappers around Radix UI primitives plus some helper utilities. Exports per file:

- `frontend/src/components/ui/accordion.tsx` — `Accordion`, `AccordionItem`, `AccordionTrigger`, `AccordionContent`
- `frontend/src/components/ui/alert-dialog.tsx` — `AlertDialog`, `AlertDialogTrigger`, `AlertDialogPortal`, `AlertDialogOverlay`, `AlertDialogContent`, `AlertDialogHeader`, `AlertDialogFooter`, `AlertDialogTitle`, `AlertDialogDescription`, `AlertDialogAction`, `AlertDialogCancel`
- `frontend/src/components/ui/alert.tsx` — `Alert`, `AlertTitle`, `AlertDescription`
- `frontend/src/components/ui/aspect-ratio.tsx` — `AspectRatio`
- `frontend/src/components/ui/avatar.tsx` — `Avatar`, `AvatarImage`, `AvatarFallback`
- `frontend/src/components/ui/badge.tsx` — `Badge`, `badgeVariants`
- `frontend/src/components/ui/breadcrumb.tsx` — `Breadcrumb`, `BreadcrumbList`, `BreadcrumbItem`, `BreadcrumbLink`, `BreadcrumbPage`, `BreadcrumbSeparator`, `BreadcrumbEllipsis`
- `frontend/src/components/ui/button.tsx` — `Button`, `buttonVariants`
- `frontend/src/components/ui/calendar.tsx` — `Calendar`
- `frontend/src/components/ui/card.tsx` — `Card`, `CardHeader`, `CardTitle`, `CardDescription`, `CardContent`, `CardFooter`, `CardAction`
- `frontend/src/components/ui/carousel.tsx` — `Carousel`, `CarouselApi`, `CarouselContent`, `CarouselItem`, `CarouselPrevious`, `CarouselNext`
- `frontend/src/components/ui/chart.tsx` — `ChartConfig`, `ChartContainer`, `ChartStyle`, `ChartTooltip`, `ChartTooltipContent`, `ChartLegend`, `ChartLegendContent`
- `frontend/src/components/ui/checkbox.tsx` — `Checkbox`
- `frontend/src/components/ui/collapsible.tsx` — `Collapsible`, `CollapsibleTrigger`, `CollapsibleContent`
- `frontend/src/components/ui/command.tsx` — `Command`, `CommandDialog`, `CommandInput`, `CommandList`, `CommandEmpty`, `CommandGroup`, `CommandItem`, `CommandSeparator`, `CommandShortcut`
- `frontend/src/components/ui/context-menu.tsx` — `ContextMenu`, `ContextMenuTrigger`, `ContextMenuPortal`, `ContextMenuContent`, `ContextMenuGroup`, `ContextMenuItem`, `ContextMenuCheckboxItem`, `ContextMenuRadioGroup`, `ContextMenuRadioItem`, `ContextMenuLabel`, `ContextMenuSeparator`, `ContextMenuShortcut`, `ContextMenuSub`, `ContextMenuSubTrigger`, `ContextMenuSubContent`
- `frontend/src/components/ui/dialog.tsx` — `Dialog`, `DialogTrigger`, `DialogPortal`, `DialogOverlay`, `DialogContent`, `DialogHeader`, `DialogFooter`, `DialogTitle`, `DialogDescription`, `DialogClose`
- `frontend/src/components/ui/drawer.tsx` — `Drawer`, `DrawerTrigger`, `DrawerPortal`, `DrawerOverlay`, `DrawerContent`, `DrawerHeader`, `DrawerFooter`, `DrawerTitle`, `DrawerDescription`, `DrawerClose`
- `frontend/src/components/ui/dropdown-menu.tsx` — `DropdownMenu`, `DropdownMenuTrigger`, `DropdownMenuPortal`, `DropdownMenuContent`, `DropdownMenuGroup`, `DropdownMenuItem`, `DropdownMenuCheckboxItem`, `DropdownMenuRadioGroup`, `DropdownMenuRadioItem`, `DropdownMenuLabel`, `DropdownMenuSeparator`, `DropdownMenuShortcut`, `DropdownMenuSub`, `DropdownMenuSubTrigger`, `DropdownMenuSubContent`
- `frontend/src/components/ui/form.tsx` — `Form`, `FormField`, `FormItem`, `FormLabel`, `FormControl`, `FormDescription`, `FormMessage`, `useFormField`
- `frontend/src/components/ui/hover-card.tsx` — `HoverCard`, `HoverCardTrigger`, `HoverCardContent`
- `frontend/src/components/ui/input-otp.tsx` — `InputOTP`, `InputOTPGroup`, `InputOTPSlot`, `InputOTPSeparator`
- `frontend/src/components/ui/input.tsx` — `Input`
- `frontend/src/components/ui/label.tsx` — `Label`
- `frontend/src/components/ui/menubar.tsx` — `Menubar`, `MenubarMenu`, `MenubarTrigger`, `MenubarPortal`, `MenubarContent`, `MenubarGroup`, `MenubarItem`, `MenubarCheckboxItem`, `MenubarRadioGroup`, `MenubarRadioItem`, `MenubarSub`, `MenubarSubTrigger`, `MenubarSubContent`, `MenubarLabel`, `MenubarSeparator`, `MenubarShortcut`
- `frontend/src/components/ui/navigation-menu.tsx` — `NavigationMenu`, `NavigationMenuList`, `NavigationMenuItem`, `NavigationMenuTrigger`, `NavigationMenuContent`, `NavigationMenuLink`, `NavigationMenuViewport`, `NavigationMenuIndicator`, `navigationMenuTriggerStyle`
- `frontend/src/components/ui/pagination.tsx` — `Pagination`, `PaginationContent`, `PaginationItem`, `PaginationLink`, `PaginationPrevious`, `PaginationNext`, `PaginationEllipsis`
- `frontend/src/components/ui/popover.tsx` — `Popover`, `PopoverTrigger`, `PopoverContent`, `PopoverAnchor`
- `frontend/src/components/ui/progress.tsx` — `Progress`
- `frontend/src/components/ui/radio-group.tsx` — `RadioGroup`, `RadioGroupItem`
- `frontend/src/components/ui/resizable.tsx` — `ResizablePanelGroup`, `ResizablePanel`, `ResizableHandle`
- `frontend/src/components/ui/scroll-area.tsx` — `ScrollArea`, `ScrollBar`
- `frontend/src/components/ui/select.tsx` — `Select`, `SelectTrigger`, `SelectValue`, `SelectContent`, `SelectGroup`, `SelectItem`, `SelectLabel`, `SelectSeparator`, `SelectScrollUpButton`, `SelectScrollDownButton`
- `frontend/src/components/ui/separator.tsx` — `Separator`
- `frontend/src/components/ui/sheet.tsx` — `Sheet`, `SheetTrigger`, `SheetClose`, `SheetPortal`, `SheetOverlay`, `SheetContent`, `SheetHeader`, `SheetFooter`, `SheetTitle`, `SheetDescription`
- `frontend/src/components/ui/sidebar.tsx` — `SidebarProvider`, `Sidebar`, `SidebarTrigger`, `SidebarRail`, `SidebarInset`, `SidebarHeader`, `SidebarFooter`, `SidebarSeparator`, `SidebarContent`, `SidebarGroup`, `SidebarGroupLabel`, `SidebarGroupAction`, `SidebarGroupContent`, `SidebarMenu`, `SidebarMenuItem`, `SidebarMenuButton`, `SidebarMenuAction`, `SidebarMenuBadge`, `SidebarMenuSkeleton`, `SidebarMenuSub`, `SidebarMenuSubItem`, `SidebarMenuSubButton`, `SidebarInput`, `useSidebar`
- `frontend/src/components/ui/skeleton.tsx` — `Skeleton`
- `frontend/src/components/ui/slider.tsx` — `Slider`
- `frontend/src/components/ui/sonner.tsx` — `Toaster`
- `frontend/src/components/ui/switch.tsx` — `Switch`
- `frontend/src/components/ui/table.tsx` — `Table`, `TableHeader`, `TableBody`, `TableFooter`, `TableHead`, `TableRow`, `TableCell`, `TableCaption`
- `frontend/src/components/ui/tabs.tsx` — `Tabs`, `TabsList`, `TabsTrigger`, `TabsContent`
- `frontend/src/components/ui/textarea.tsx` — `Textarea`
- `frontend/src/components/ui/toggle-group.tsx` — `ToggleGroup`, `ToggleGroupItem`
- `frontend/src/components/ui/toggle.tsx` — `Toggle`, `toggleVariants`
- `frontend/src/components/ui/tooltip.tsx` — `Tooltip`, `TooltipTrigger`, `TooltipContent`, `TooltipProvider`
- `frontend/src/components/ui/use-mobile.ts` — `useIsMobile`
- `frontend/src/components/ui/utils.ts` — `cn` (Tailwind className merge helper)

### Data, model artifacts, and generated files

- `data/raw/metal_nut/` — raw MVTec AD dataset copy (expected input to preprocessing).
  - In this repo’s current contents:
    - `train/good/`: 220 images
    - `test/`: color 22, good 22, scratch 23 (67 total)
    - `ground_truth/`: defect masks (binary masks per defect type)
  - Includes `license.txt` / `readme.txt` from MVTec (CC BY-NC-SA 4.0).
- `data/processed/metal_nut/` — output of `training/preprocess_metal_nut.py` (images + single-channel class-ID masks).
  - In this repo’s current contents:
    - `train/`: 259 image/mask pairs (good 233, color 13, scratch 13)
    - `val/`: 12 image/mask pairs (good 4, color 4, scratch 4)
    - `test/`: 16 image/mask pairs (good 5, color 5, scratch 6)
- `models/best_unet_metalnut_colorscratch.pth` — trained PyTorch weights loaded by `api/main.py`.
- `metal_nut/` — another copy of the raw dataset at repo root (same structure as `data/raw/metal_nut/`).
- `__pycache__/` and `*.pyc` files — runtime Python caches (ignored by `.gitignore`, safe to delete).
