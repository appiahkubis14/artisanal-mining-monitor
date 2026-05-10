# Atewa Forest Reserve — Illegal Mining Detection Pipeline

**Master's Thesis | KNUST, Ghana**

A multi-sensor deep learning system for detecting illegal artisanal gold mining (*galamsey*)
in the Atewa Forest Reserve, Ghana. Fuses Sentinel-2, Sentinel-1 SAR, Landsat 8/9, and UAV
imagery with automatic mask generation, Roboflow-assisted equipment annotation, U-Net
segmentation, YOLOv8 object detection, multi-date change tracking, and an interactive map.
Every expensive step has checkpoint/resume support — re-running a command after a crash or
power cut continues from where it stopped.

---

## Background

Atewa Forest Reserve (23,663 ha, Eastern Region, Ghana) is a globally significant Pleistocene
refugium and the headwater catchment for five major rivers supplying clean water to over five
million people. Between 2017 and 2023 approximately 890 ha were destroyed by illegal mining.
Cloud cover exceeds 180 days per year, making single-sensor optical monitoring insufficient.

This pipeline combines multi-temporal satellite imagery with UAV ground surveys and two
deep-learning models: a U-Net segmentation model for satellite-scale site delineation and a
YOLOv8 object detector for individual equipment identification in UAV tiles. Outputs — priority-
ranked GeoJSONs, change reports, and an interactive HTML map — are designed for direct use by
rangers and enforcement agencies.

---

## Repository Layout

```
atewa_mining_detection/
├── config.yaml                          # Single source of truth for all parameters
├── main.py                              # CLI: python main.py --step <name>
├── requirements.txt
├── README.md
│
├── scripts/
│   ├── utils.py                         # Logging, I/O helpers, checkpointing
│   ├── data_loader.py                   # Load Sentinel-2/1, Landsat, UAV, ground-truth
│   ├── preprocessor.py                  # Cloud masking, speckle filter, normalisation (resumable)
│   ├── tiler.py                         # Overlapping tile extraction + mask co-tiling (resumable)
│   ├── feature_engineering.py           # NDVI/NDWI/MNDWI/NDBI/SAVI, GLCM, SAR, temporal (resumable)
│   ├── generate_uav_masks.py            # Auto-annotation: bare-soil + entropy + edges → mask (resumable)
│   ├── prepare_ground_truth.py          # Mask priority loader + per-tile mask creation
│   ├── dataset.py                       # PyTorch Dataset, augmentation, class-weight sampler
│   ├── models.py                        # U-Net (ResNet-50 + AttentionGate) + YOLO wrapper
│   ├── train_unet.py                    # Focal+Dice loss, AMP, TensorBoard, encoder freeze (resumable)
│   ├── uav_tiler_roboflow.py            # 640x640 JPEG tiles + coordinate map for Roboflow
│   ├── import_roboflow_annotations.py   # API or manual import, validate, georeference annotations
│   ├── train_yolo.py                    # YOLOv8 training with annotation-source priority (resumable)
│   ├── inference_satellite.py           # U-Net inference + Hanning blend, per-date resumable
│   ├── inference_uav.py                 # YOLOv8 inference + cross-tile NMS + lon/lat projection
│   ├── postprocess.py                   # Polygons, confidence tiers, priority scoring
│   ├── fusion.py                        # RANSAC alignment + satellite/UAV attribute fusion
│   ├── change_detection.py              # Multi-date tracking, Kalman smoothing, expansion rate
│   ├── validation.py                    # ROC/PR, spatial block CV, confusion matrix, threshold sweep
│   └── dashboard.py                     # Interactive Folium HTML map
│
└── data/
    ├── raw/
    │   ├── sentinel2/                   # Sentinel-2 L2A SAFE archives or GeoTIFFs
    │   ├── sentinel1/                   # Sentinel-1 GRD SAFE archives or VV/VH GeoTIFFs
    │   ├── landsat/                     # Landsat 8/9 Collection-2 Level-2 TIFs (optional)
    │   └── uav/                         # UAV orthomosaic GeoTIFF (RGB or multispectral)
    ├── boundary/
    │   └── atewa_boundary.geojson       # Reserve boundary in WGS84
    ├── ground_truth/
    │   └── mining_sites.csv             # Optional GPS waypoints (lat, lon, date, notes)
    ├── roboflow_upload/                 # Auto-generated tiles + coordinate map for Roboflow
    ├── roboflow_export/                 # Place Roboflow YOLO export here (or use API)
    ├── processed/                       # Preprocessed composites — auto-generated, resumable
    ├── tiles/                           # Image + mask patches — auto-generated, resumable
    ├── features/                        # Feature stack GeoTIFF — auto-generated, resumable
    ├── masks/                           # Training masks — auto-generated, resumable
    ├── models/                          # Trained weights — auto-generated
    └── outputs/                         # Detections, reports, dashboard — auto-generated
```

---

## Installation

### 1. Set up Python 3.10 environment

```bash
cd ~/projects/atewa_mining_detection
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 2. Install PyTorch

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install all other dependencies

```bash
pip install -r requirements.txt
```

---

## Data Sources

### Sentinel-2 (required)
1. Go to https://browser.dataspace.copernicus.eu/
2. Draw the Atewa AOI (approx. 6.1-6.4 N, 0.70-0.45 W)
3. Filter: Sentinel-2 L2A, cloud cover < 30%, 2021-present
4. Download SAFE archives and place in data/raw/sentinel2/

### Sentinel-1 (required)
1. Same Copernicus portal: Sentinel-1 GRD, mode IW, polarisation VV+VH
2. Download and place in data/raw/sentinel1/

### Landsat 8/9 (optional)
- USGS EarthExplorer: https://earthexplorer.usgs.gov/
- Collection 2, Level-2 Surface Reflectance
- Place in data/raw/landsat/

### UAV Orthomosaic (required for mask generation and equipment detection)
- Fly at GSD <= 5 cm; process in Agisoft Metashape with GCP correction
- Export as a single GeoTIFF (RGB, or multispectral with NIR as band 4)
- Place in data/raw/uav/

### Atewa Reserve Boundary
- Source: Ghana Forestry Commission or IUCN WDPA
- Save as data/boundary/atewa_boundary.geojson (WGS84)

### Ground-Truth GPS Points (optional)
- Only needed if skipping UAV auto-annotation
- CSV with columns: latitude, longitude, date (YYYY-MM-DD), notes
- Save as data/ground_truth/mining_sites.csv

---

## Pipeline Steps

All 15 steps run in sequence with --step all, or individually with --step <name>.
Steps 6-7 (Roboflow) are optional and skipped gracefully when no annotations exist.

| # | CLI name | Script | What it does | Resumable |
|---|----------|--------|-------------|-----------|
| 1 | preprocess | preprocessor.py | Cloud masking, temporal median composite, Lee speckle filter, UAV resampling + alignment, normalisation | Yes — per sensor |
| 2 | tile | tiler.py | 512x512 satellite patches, 640x640 UAV patches, mask co-tiling | Yes — every 100 tiles |
| 3 | features | feature_engineering.py | NDVI, NDWI, MNDWI, NDBI, SAVI, brightness; GLCM entropy/contrast/homogeneity; temporal NDVI trend; SAR VV/VH ratio; UAV features | Yes — full stack |
| 4 | generate_masks | generate_uav_masks.py | Bare-soil detection, entropy, edge density, rule-based score, morphological clean, resample to 10 m | Yes — all 3 outputs |
| 5 | ground_truth | prepare_ground_truth.py | Load best available mask (UAV auto -> GPS -> spectral) and create per-tile mask .npy files | No |
| 6 | prep_roboflow | uav_tiler_roboflow.py | 640x640 JPEG tiles + tile_coordinates.json + README for Roboflow upload | No |
| 7 | import_roboflow | import_roboflow_annotations.py | API or manual download, validate boxes, copy to pipeline, build georeferenced GeoJSON + stats | No |
| 8 | train_unet | train_unet.py | ResNet-50 U-Net, Focal+Dice loss, AMP, encoder freeze schedule, early stopping, TensorBoard | Yes — epoch level |
| 9 | train_yolo | train_yolo.py | YOLOv8m; annotation priority: Roboflow -> GPS pseudo-labels -> skip; resumes from last.pt | Yes — epoch level |
| 10 | infer | inference_satellite.py + inference_uav.py | U-Net probability maps (Hanning blend); YOLO equipment detections with cross-tile NMS | Yes — per date |
| 11 | postprocess | postprocess.py | Threshold -> polygons -> IoU dedup -> confidence/water/activity scoring -> priority ranking | No |
| 12 | fusion | fusion.py | RANSAC coordinate alignment; spatial join of equipment to satellite polygons; fused confidence | No |
| 13 | change | change_detection.py | Centroid+IoU site matching across dates; Kalman-smoothed area time-series; expansion rate | No |
| 14 | validate | validation.py | Pixel F1/IoU/ROC-AUC/PR-AUC; object detection rate; 2 km spatial block CV; threshold sweep | No |
| 15 | dashboard | dashboard.py | Interactive Folium HTML: basemaps, confidence polygons, equipment markers, change layer, priority panel | No |

---

## Checkpoint and Resume

Every expensive step saves a sentinel file to disk. Re-running the same command after an
interruption detects the sentinel file and skips straight to the next step, printing [RESUME].

| Step | Sentinel file(s) | What is skipped on resume |
|------|-----------------|--------------------------|
| preprocess S2 | data/processed/sentinel2/sentinel2_processed.tif | Cloud masking, gap-filling, normalisation |
| preprocess S1 | data/processed/sentinel1/sentinel1_processed.tif | Sigma0 calibration, Lee speckle filter |
| preprocess UAV | data/processed/uav/uav_highres.tif + uav_10m.tif | Resampling, RANSAC alignment attempt |
| preprocess Landsat | data/processed/landsat/landsat_processed.tif | Scaling and normalisation |
| tile | data/tiles/satellite/tiling_progress.json | Already-saved .npy tile files |
| features | data/features/feature_stack.tif + feature_names.json | All index, texture, SAR, UAV computation |
| generate_masks | data/masks/uav_mining_mask_fullres.tif + uav_mining_mask_10m.tif + mining_mask_10m.npy | All 8 auto-annotation steps |
| train_unet | data/models/unet_last.pth | Resumes from saved epoch + optimiser state |
| train_yolo | data/models/yolo_mining/weights/last.pt | Passes resume=True to ultralytics |
| infer | data/outputs/prob_map_<date>.tif (per date) | Dates already inferred are skipped individually |

To force a full re-run of any step, delete its sentinel file(s) before running.

---

## Running the Pipeline

### Option A — Full automatic run

```bash
python main.py --step all
```

Runs all 15 steps. Steps 6-7 (Roboflow) are included but train_yolo logs a guidance message
and continues if no annotations have been imported. Every step resumes automatically if
interrupted.

### Option B — Standard workflow with Roboflow equipment annotation

```bash
# Phase 1 — preprocessing and mask generation
python main.py --step preprocess
python main.py --step tile
python main.py --step features
python main.py --step generate_masks

# Phase 2 — U-Net training labels and model
python main.py --step ground_truth
python main.py --step train_unet

# Phase 3 — Roboflow equipment annotation
python main.py --step prep_roboflow
# Upload data/roboflow_upload/images/ to https://roboflow.com
# Annotate equipment, export as YOLOv8
# Either fill API credentials in config.yaml (see below) OR
# place the exported labels/ folder in data/roboflow_export/
python main.py --step import_roboflow
python main.py --step train_yolo

# Phase 4 — inference, analysis, outputs
python main.py --step infer
python main.py --step postprocess
python main.py --step fusion
python main.py --step change
python main.py --step validate
python main.py --step dashboard
```

### Option C — Satellite-only (no UAV)

```bash
python main.py --step all --no-uav
```

### Other options

```bash
python main.py --config my_config.yaml --step all    # custom config file
python main.py --step all --log-level DEBUG           # verbose logging
```

---

## Auto-Annotation: How generate_masks Works

generate_uav_masks.py produces binary mining masks from the UAV orthomosaic automatically.

| Step | What happens |
|------|-------------|
| 1 Load | Reads UAV GeoTIFF from data/processed/uav/, normalises to 0-1 |
| 2 Bare-soil | RGB thresholds (R>100, G>80, R/G in 0.8-1.5) + HSV range (hue 20-60, low saturation); OR-combined with NDVI<0.2 if NIR band present |
| 3 Entropy | Shannon entropy on 5x5 sliding window — high entropy signals excavated/disturbed ground |
| 4 Edge density | Canny edges summed over 50x50 pixel blocks — mining clearings have high boundary density |
| 5 Score | score = 0.4 x bare_soil + 0.3 x (entropy > 2.0) + 0.3 x (edge_density > 0.3) |
| 6 Threshold | Pixels with score >= 0.6 classified as mining |
| 7 Clean | Morphological opening (3x3) removes speckle; closing (5x5) fills holes; components < 50 m2 removed |
| 8 Resample | Majority-vote to 10 m: a 10 m cell is mining if > 30% of its UAV pixels are mining |

Outputs saved to data/masks/:
- uav_mining_mask_fullres.tif — full UAV-resolution binary mask
- uav_mining_mask_10m.tif — 10 m mask aligned to Sentinel-2
- mining_mask_10m.npy — NumPy array loaded by the U-Net dataset

### Mask priority in prepare_ground_truth

The pipeline always picks the best available mask:

1. data/masks/uav_mining_mask_10m.npy — from generate_masks (preferred)
2. data/ground_truth/mining_sites.csv — GPS points buffered to 20 m circles
3. UAV spectral pseudo-labels — legacy NDVI + brightness thresholding
4. Error with instructions if nothing is found

### Tuning thresholds

```yaml
# config.yaml -> uav_annotation
uav_annotation:
  bare_soil_red_threshold: 100      # lower = more bare soil detected
  bare_soil_green_threshold: 80
  bare_soil_ratio_min: 0.8
  bare_soil_ratio_max: 1.5
  entropy_threshold: 2.0            # raise to be more selective
  edge_density_threshold: 0.3
  mining_score_threshold: 0.6       # raise to reduce false positives
  min_mining_area_m2: 50
  resample_cell_size_m: 10
```

---

## Equipment Annotation with Roboflow

### Step 1 — Create the upload package

```bash
python main.py --step prep_roboflow
```

Produces in data/roboflow_upload/:
- images/ — 640x640 JPEG tiles (one per UAV tile)
- tile_coordinates.json — precise affine + WGS84 bounds per tile (do not delete)
- labels/ — empty .txt placeholders
- README_roboflow.txt — upload instructions

### Step 2 — Annotate on Roboflow

1. Go to https://roboflow.com and create an Object Detection project
2. Upload the images/ folder
3. Annotate using these class IDs (order is critical):

| Class ID | Name |
|----------|------|
| 0 | excavator |
| 1 | truck |
| 2 | water_pump |
| 3 | settling_pond |
| 4 | pit |

4. Generate Dataset -> Export -> YOLOv8 format

### Step 3 — Import annotations (two options)

Option A — Roboflow API (automatic, recommended):

Fill in config.yaml -> roboflow:

```yaml
roboflow:
  api_key:   "YOUR_KEY"           # https://app.roboflow.com -> profile -> API Keys
  workspace: "YOUR_WORKSPACE"     # slug shown in the URL
  project:   "YOUR_PROJECT"       # project name slug
  version:   1                    # dataset version number
```

Then run:

```bash
python main.py --step import_roboflow
```

The SDK downloads the export directly — no ZIP handling needed.

Option B — Manual download:

1. Download the YOLOv8 export ZIP from Roboflow
2. Extract and place the labels/ folder at data/roboflow_export/labels/
3. Run:

```bash
python main.py --step import_roboflow
```

The import step automatically uses the API if credentials are set, otherwise falls back to
the manual export. Either way, it then:
- Verifies each label file matches a known tile in tile_coordinates.json
- Validates bounding boxes (all coords in [0,1], class IDs 0-4)
- Copies validated labels to data/tiles/uav/labels/ (pipeline working directory)
- Saves data/outputs/annotated_equipment.geojson — WGS84 point GeoJSON of every detection
- Saves data/outputs/equipment_annotation_stats.json — per-class instance counts

### Step 4 — Train

```bash
python main.py --step train_yolo
```

Annotation source priority in train_yolo:
1. Roboflow-imported labels in data/tiles/uav/labels/ — trains immediately
2. Roboflow export detected but not imported — error with import instructions
3. GPS pseudo-annotations available — generates approximate boxes and trains
4. Nothing found — skips with guidance for both Roboflow and LabelImg

### LabelImg alternative (offline)

```bash
pip install labelImg
labelImg data/tiles/uav/images/ data/tiles/uav/labels/
# Save in YOLO format, then:
python main.py --step train_yolo
```

---

## Key Outputs

| Path | Description |
|------|-------------|
| data/processed/sentinel2/sentinel2_processed.tif | Sentinel-2 cloud-free composite |
| data/processed/sentinel1/sentinel1_processed.tif | Sentinel-1 speckle-filtered composite |
| data/processed/uav/uav_highres.tif | UAV at native resolution (for YOLO) |
| data/processed/uav/uav_10m.tif | UAV resampled to 10 m (for fusion) |
| data/features/feature_stack.tif | All features stacked [C, H, W] |
| data/features/feature_names.json | Channel names for interpretability |
| data/masks/uav_mining_mask_fullres.tif | Full-resolution binary mining mask |
| data/masks/uav_mining_mask_10m.tif | 10 m binary mask aligned to Sentinel-2 |
| data/masks/mining_mask_10m.npy | NumPy array for U-Net training |
| data/roboflow_upload/images/ | 640x640 JPEG tiles for Roboflow |
| data/roboflow_upload/tile_coordinates.json | Per-tile affine + WGS84 bounds |
| data/outputs/annotated_equipment.geojson | Georeferenced equipment points |
| data/outputs/equipment_annotation_stats.json | Instance counts per equipment class |
| data/outputs/prob_map_YYYY-MM-DD.tif | U-Net probability raster per date |
| data/outputs/binary_{low,medium,high}_YYYY-MM-DD.tif | Thresholded binary maps |
| data/outputs/mining_detections_YYYY-MM-DD.geojson | Postprocessed mining polygons |
| data/outputs/mining_summary_YYYY-MM-DD.csv | Per-site summary with priority rank |
| data/outputs/detection_summary_YYYY-MM-DD.json | Detection counts and statistics |
| data/outputs/fused_detections_YYYY-MM-DD.geojson | Satellite + UAV fused polygons |
| data/outputs/fused_summary_YYYY-MM-DD.csv | Fused summary sorted by priority |
| data/outputs/fused_detections_all.geojson | All dates combined |
| data/outputs/change_report.geojson | Multi-date site tracks with status |
| data/outputs/change_report.csv | Per-site expansion rate (ha/month) |
| data/outputs/change_summary.json | Aggregate change statistics |
| data/outputs/site_time_series.json | Kalman-smoothed area time-series per site |
| data/outputs/validation/accuracy_report.json | Full accuracy metrics |
| data/outputs/validation/roc_pr_*.png | ROC and precision-recall curves |
| data/outputs/validation/confusion_matrix_*.png | Confusion matrix plots |
| data/outputs/validation/threshold_analysis.png | F1/precision/recall vs threshold |
| data/outputs/validation/spatial_cv_f1.png | F1 score per spatial block |
| data/outputs/dashboard.html | Interactive map — open in any browser |
| data/models/unet_best.pth | Best U-Net checkpoint (by validation F1) |
| data/models/unet_last.pth | Last U-Net checkpoint (for resume) |
| data/models/yolo_best.pt | Best YOLOv8 checkpoint |
| data/models/yolo_mining/weights/last.pt | Last YOLOv8 checkpoint (for resume) |

---

## Monitoring Training

TensorBoard logs are written to runs/unet_training/ during U-Net training:

```bash
tensorboard --logdir runs/
# Open http://localhost:6006
```

Plots include train/val loss, F1, IoU, and confusion matrix every 5 epochs.

---

## Configuration Reference

All parameters are in config.yaml. Key sections:

```yaml
preprocessing:
  sentinel2_resolution: 10          # metres per pixel
  cloud_mask_threshold: 0.30        # reject cloud probability > 30%
  speckle_window: 5                 # Lee filter window (Sentinel-1)

tiling:
  satellite_tile_size: 512          # pixels; U-Net input
  uav_tile_size: 640                # pixels; YOLO input
  overlap: 0.20                     # 20% tile overlap

uav_annotation:                     # generate_masks thresholds
  mining_score_threshold: 0.6       # lower -> more detections
  entropy_threshold: 2.0
  resample_cell_size_m: 10

roboflow:                           # prep_roboflow / import_roboflow
  tile_size: 640
  overlap_fraction: 0.10
  output_dir: "data/roboflow_upload"
  export_dir: "data/roboflow_export"
  api_key:   ""                     # fill in for automatic API download
  workspace: ""
  project:   ""
  version:   1
  classes:
    excavator: 0
    truck: 1
    water_pump: 2
    settling_pond: 3
    pit: 4

training_unet:
  encoder: resnet50
  epochs: 100
  learning_rate: 0.001
  focal_loss_gamma: 2.0
  focal_loss_alpha: 0.25
  early_stopping_patience: 15

training_yolo:
  model: yolov8m.pt
  epochs: 100
  image_size: 640

inference:
  thresholds: [0.3, 0.5, 0.7]      # low / medium / high confidence tiers
  min_polygon_area_m2: 50

prioritization:
  weights:
    area: 0.30
    equipment: 0.25
    water_proximity: 0.20
    confidence: 0.15
    activity: 0.10
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| [RESUME] logged but results look wrong | Delete the sentinel file for that step to force re-run |
| No UAV GeoTIFF in data/processed/uav/ | Run --step preprocess before --step generate_masks |
| Auto-mask is all zeros | Lower mining_score_threshold (e.g. 0.4) in config.yaml |
| Auto-mask covers too much | Raise mining_score_threshold (e.g. 0.7) or entropy_threshold |
| No tile_coordinates.json when importing | Run --step prep_roboflow first |
| Roboflow export detected but not imported | Run python main.py --step import_roboflow then retry train_yolo |
| API download fails silently | Check api_key, workspace, project in config.yaml; pip install roboflow |
| Annotation validation warnings | Check Roboflow exported in YOLOv8 format (not v5 or COCO) |
| GDAL / rasterio install errors | pip install rasterio --no-binary rasterio |
| Out-of-memory during U-Net training | Reduce batch_size in config.yaml (try 8 or 4) |
| train_yolo restarts from epoch 0 | data/models/yolo_mining/weights/last.pt was deleted; resume is lost |
| YOLOv8 import error | pip install ultralytics --upgrade |
| folium missing for dashboard | pip install folium branca |
| No probability maps for postprocess | Run --step infer first |
| Low F1 (< 0.5) | More training epochs; check generate_masks produced a balanced mask |

---

## Citation

```
[Your Name] (2025). Detecting Illegal Artisanal Mining in Atewa Forest Reserve
Using Multi-Sensor Deep Learning, UAV Auto-Annotation, and Roboflow Equipment
Labelling. Master's Thesis, Kwame Nkrumah University of Science and Technology
(KNUST), Ghana.
```

---

## Licence

For academic and non-commercial use only.
(c) 2025 KNUST -- Department of [Your Department]