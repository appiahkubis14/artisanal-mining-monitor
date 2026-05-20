"""
main.py
=======
CLI orchestrator for the Atewa Illegal Mining Detection Pipeline.

Usage
-----
    python main.py --step all
    python main.py --step preprocess
    python main.py --step train_unet
    python main.py --step train_yolo
    python main.py --step infer
    python main.py --step postprocess
    python main.py --step fusion
    python main.py --step change
    python main.py --step validate
    python main.py --step dashboard

Options
-------
    --config    Path to config.yaml (default: config.yaml)
    --step      Pipeline step to run (default: all)
    --no-uav    Skip UAV-dependent steps even if UAV data present
    --log-level DEBUG | INFO | WARNING (default: INFO)

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import load_config, setup_logging, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def step_generate_masks(config: dict) -> None:
    """Auto-generate UAV mining masks. Skips gracefully if no UAV data."""
    import os

    log.info("=" * 60)
    log.info("STEP: GENERATE UAV MASKS (auto-annotation)")
    log.info("=" * 60)

    uav_dir = os.path.join(
        config["paths"].get("processed_data", "data/processed"), "uav"
    )
    has_uav = os.path.isdir(uav_dir) and any(
        f.endswith(".tif") for f in os.listdir(uav_dir)
    ) if os.path.isdir(uav_dir) else False

    if not has_uav:
        log.warning(
            "No UAV data found — skipping mask generation. "
            "Pipeline will use GPS ground-truth masks if available."
        )
        return

    from scripts.generate_uav_masks import generate_uav_masks
    stats = generate_uav_masks(config)
    log.info(
        f"Masks ready — 10 m mining coverage: "
        f"{stats.get('mask_10m_mining_pct', '?')}%"
    )


def step_prep_roboflow(config: dict) -> None:
    """Tile UAV orthomosaic into 640×640 JPEGs for Roboflow annotation."""
    import os

    log.info("=" * 60)
    log.info("STEP: PREP ROBOFLOW — UAV tile package")
    log.info("=" * 60)

    uav_dir = os.path.join(
        config["paths"].get("processed_data", "data/processed"), "uav"
    )
    has_uav = os.path.isdir(uav_dir) and any(
        f.endswith(".tif") for f in os.listdir(uav_dir)
    ) if os.path.isdir(uav_dir) else False

    if not has_uav:
        log.warning(
            "No UAV data found — skipping Roboflow tile preparation. "
            "Add UAV orthomosaic to data/raw/uav/ and re-run to enable."
        )
        return

    from scripts.uav_tiler_roboflow import run_uav_tiler_roboflow
    result = run_uav_tiler_roboflow(config)
    log.info(
        f"Roboflow package ready: {result['n_tiles']} tiles "
        f"in {result['images_dir']}"
    )


def step_import_roboflow(config: dict) -> None:
    """Import Roboflow-exported YOLO annotations back into the pipeline."""
    from scripts.import_roboflow_annotations import run_import_roboflow

    log.info("=" * 60)
    log.info("STEP: IMPORT ROBOFLOW — annotations import")
    log.info("=" * 60)
    result = run_import_roboflow(config)

    if result.get("skipped"):
        log.info(
            "Roboflow import skipped — no dataset version available yet. "
            "Pipeline continuing with GPS pseudo-labels for YOLO training."
        )
    else:
        log.info(
            f"Import complete: {result['total_instances']} equipment instances, "
            f"{result['annotated_tiles']} annotated tiles"
        )


def step_preprocess(config: dict, uav_available: bool) -> None:
    """Run all preprocessing steps."""
    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all

    log.info("=" * 60)
    log.info("STEP: PREPROCESSING")
    log.info("=" * 60)

    data = load_all_data(config)
    preprocess_all(data, config)


def step_tile(config: dict) -> None:
    """Tile preprocessed imagery."""
    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all
    from scripts.tiler import run_tiling

    log.info("=" * 60)
    log.info("STEP: TILING")
    log.info("=" * 60)

    data = load_all_data(config)
    preprocessed = preprocess_all(data, config)
    uav_available = data.get("uav_available", False)
    run_tiling(preprocessed, config, uav_available=uav_available)


def step_features(config: dict) -> None:
    """Compute spectral indices, texture, and temporal features."""
    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all
    from scripts.feature_engineering import compute_all_features

    log.info("=" * 60)
    log.info("STEP: FEATURE ENGINEERING")
    log.info("=" * 60)

    data = load_all_data(config)
    preprocessed = preprocess_all(data, config)
    uav_available = data.get("uav_available", False)
    compute_all_features(preprocessed, config, uav_available=uav_available)


def step_ground_truth(config: dict) -> None:
    """Prepare ground-truth masks."""
    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all
    from scripts.tiler import run_tiling
    from scripts.prepare_ground_truth import prepare_ground_truth

    log.info("=" * 60)
    log.info("STEP: GROUND TRUTH PREPARATION")
    log.info("=" * 60)

    data = load_all_data(config)
    preprocessed = preprocess_all(data, config)
    uav_available = data.get("uav_available", False)
    tiling_results = run_tiling(preprocessed, config, uav_available=uav_available)
    prepare_ground_truth(data, preprocessed, tiling_results, config)


def step_train_unet(config: dict) -> None:
    """Train the U-Net segmentation model."""
    import numpy as np
    import rasterio
    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all
    from scripts.tiler import run_tiling
    from scripts.train_unet import train_unet

    log.info("=" * 60)
    log.info("STEP: TRAINING U-NET")
    log.info("=" * 60)

    # Load tile metadata from tiling step (uses checkpoints so fast if already done)
    data = load_all_data(config)
    preprocessed = preprocess_all(data, config)
    uav_available = data.get("uav_available", False)
    tiling_results = run_tiling(preprocessed, config, uav_available=uav_available)

    # satellite tile_metadata is under the sentinel2 key
    tile_metadata = tiling_results.get("sentinel2", tiling_results)

    # Load feature stack to infer channel count
    features_dir = Path(config["paths"]["features"])
    feature_tif  = features_dir / "feature_stack.tif"
    if feature_tif.exists():
        with rasterio.open(feature_tif) as src:
            feature_stack = src.read().astype(np.float32)
    else:
        log.warning("feature_stack.tif not found — channel count will be inferred from tiles.")
        feature_stack = None

    Path(config["paths"]["models"]).mkdir(parents=True, exist_ok=True)
    train_unet(tile_metadata, feature_stack, config)


def step_train_yolo(config: dict, uav_available: bool) -> None:
    """Train the YOLOv8 equipment detector."""
    if not uav_available:
        log.warning("UAV data not available – skipping YOLO training.")
        return

    from scripts.data_loader import load_all_data
    from scripts.preprocessor import preprocess_all
    from scripts.tiler import run_tiling
    from scripts.train_yolo import train_yolo

    log.info("=" * 60)
    log.info("STEP: TRAINING YOLO")
    log.info("=" * 60)

    data = load_all_data(config)
    preprocessed = preprocess_all(data, config)
    tiling_results = run_tiling(preprocessed, config, uav_available=True)

    uav_tiling_results = tiling_results.get("uav", {})
    if not uav_tiling_results:
        log.warning("No UAV tiling results found – skipping YOLO training.")
        return

    gt_df    = data.get("ground_truth")
    uav_meta = preprocessed.get("uav", {}).get("meta_highres")

    train_yolo(
        uav_tiling_results=uav_tiling_results,
        config=config,
        gt_df=gt_df,
        uav_meta=uav_meta,
    )


def step_infer(config: dict, uav_available: bool) -> None:
    """Run inference on all dates."""
    import rasterio
    import numpy as np
    from scripts.inference_satellite import run_multi_date_inference

    log.info("=" * 60)
    log.info("STEP: SATELLITE INFERENCE")
    log.info("=" * 60)

    # Load feature stack from disk (produced by step_features)
    features_dir = Path(config["paths"]["features"])
    feature_tif  = features_dir / "feature_stack.tif"

    if not feature_tif.exists():
        log.error("feature_stack.tif not found. Run --step features first.")
        return

    with rasterio.open(feature_tif) as src:
        feature_stack = src.read().astype(np.float32)
        meta = dict(src.meta)

    # Build a per-date dict — use dates from processed S2 files if available
    processed_s2_dir = Path(config["paths"]["processed_data"]) / "sentinel2"
    date_tifs = sorted(processed_s2_dir.glob("sentinel2_2*.tif"))

    if date_tifs:
        feature_stacks_by_date = {}
        for tif in date_tifs:
            date_str = tif.stem.replace("sentinel2_", "")
            feature_stacks_by_date[date_str] = feature_stack
    else:
        # Fallback: single inference labelled 'latest'
        feature_stacks_by_date = {"latest": feature_stack}

    run_multi_date_inference(feature_stacks_by_date, meta, config)

    if uav_available:
        from scripts.inference_uav import run_uav_inference
        log.info("--- UAV Inference ---")
        uav_tiles_dir = Path(config["paths"]["tiles"]) / "uav"
        output_dir    = Path(config["paths"]["outputs"])
        model_path    = Path(config["paths"]["models"]) / "yolo_best.pt"
        if not model_path.exists():
            log.warning("yolo_best.pt not found — skipping UAV inference.")
            return
        run_uav_inference(
            tiles_dir=uav_tiles_dir,
            model_path=model_path,
            output_dir=output_dir,
            config=config,
        )
    else:
        log.info("UAV not available — skipping UAV inference.")


def step_postprocess(config: dict) -> None:
    """Postprocess probability maps into mining site polygons."""
    from scripts.postprocess import run_postprocess

    log.info("=" * 60)
    log.info("STEP: POSTPROCESSING")
    log.info("=" * 60)
    run_postprocess(config)


def step_fusion(config: dict) -> None:
    """Fuse satellite and UAV detections."""
    from scripts.fusion import run_fusion

    log.info("=" * 60)
    log.info("STEP: FUSION")
    log.info("=" * 60)
    run_fusion(config)


def step_change(config: dict) -> None:
    """Run multi-date change detection."""
    from scripts.change_detection import run_change_detection

    log.info("=" * 60)
    log.info("STEP: CHANGE DETECTION")
    log.info("=" * 60)
    run_change_detection(config)


def step_validate(config: dict) -> None:
    """Validate pipeline accuracy."""
    from scripts.validation import run_validation

    log.info("=" * 60)
    log.info("STEP: VALIDATION")
    log.info("=" * 60)
    run_validation(config)


def step_dashboard(config: dict) -> None:
    """Build interactive HTML dashboard."""
    from scripts.dashboard import build_dashboard

    log.info("=" * 60)
    log.info("STEP: DASHBOARD")
    log.info("=" * 60)
    out = build_dashboard(config)
    log.info(f"Dashboard saved: {out}")
    log.info(f"Open in browser: file://{out.resolve()}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

VALID_STEPS = [
    "all",
    "preprocess",
    "tile",
    "features",
    "generate_masks",     # auto-annotation from UAV
    "ground_truth",
    "prep_roboflow",      # create Roboflow upload package
    "import_roboflow",    # import Roboflow-exported annotations
    "train_unet",
    "train_yolo",
    "infer",
    "postprocess",
    "fusion",
    "change",
    "validate",
    "dashboard",
]

ALL_STEPS_ORDER = [
    "preprocess",
    "tile",
    "features",
    "generate_masks",     # runs before ground_truth
    "ground_truth",
    "prep_roboflow",      # optional: create Roboflow package
    "import_roboflow",    # optional: import after Roboflow annotation
    "train_unet",
    "train_yolo",         # uses imported Roboflow annotations if present
    "infer",
    "postprocess",
    "fusion",
    "change",
    "validate",
    "dashboard",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Atewa Forest Illegal Mining Detection Pipeline\n"
            "Master's Thesis — KNUST, Ghana\n\n"
            "Multi-sensor deep learning system combining Sentinel-2, "
            "Sentinel-1 SAR, and UAV imagery."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    parser.add_argument(
        "--step",
        default="all",
        choices=VALID_STEPS,
        help="Pipeline step to run (default: all)",
    )
    parser.add_argument(
        "--no-uav",
        action="store_true",
        help="Skip UAV-dependent steps even if UAV data is present",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Logging
    setup_logging(log_level=args.log_level)
    log.info("=" * 70)
    log.info("  ATEWA FOREST RESERVE — ILLEGAL MINING DETECTION PIPELINE")
    log.info("  Master's Thesis | KNUST, Ghana")
    log.info("=" * 70)

    # Config
    config_path = Path(args.config)
    if not config_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    config = load_config(config_path)
    log.info(f"Loaded config: {config_path}")

    # UAV availability check
    uav_dir = Path(config["paths"].get("uav", "data/raw/uav"))
    uav_has_data = any(uav_dir.glob("*.tif")) if uav_dir.exists() else False
    uav_available = uav_has_data and not args.no_uav
    log.info(f"UAV data available: {uav_available}")

    # Ensure output dirs exist
    for key in ("processed", "tiles", "features", "masks", "models", "outputs", "logs"):
        Path(config["paths"].get(key, f"data/{key}")).mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    try:
        step = args.step

        if step == "all":
            log.info("Running FULL pipeline...")
            for s in ALL_STEPS_ORDER:
                _run_single_step(s, config, uav_available)
        else:
            _run_single_step(step, config, uav_available)

    except KeyboardInterrupt:
        log.warning("\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        log.error(f"Pipeline failed at step '{args.step}': {e}", exc_info=True)
        sys.exit(1)

    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)
    log.info("=" * 70)
    log.info(f"  PIPELINE COMPLETE — elapsed time: {h:02d}h {m:02d}m {s:02d}s")
    log.info("=" * 70)


def _run_single_step(step: str, config: dict, uav_available: bool) -> None:
    """Dispatch a single step with timing."""
    t0 = time.time()
    log.info(f"\n>>> Starting step: {step}")

    dispatch = {
        "preprocess":       lambda: step_preprocess(config, uav_available),
        "tile":             lambda: step_tile(config),
        "features":         lambda: step_features(config),
        "generate_masks":   lambda: step_generate_masks(config),
        "ground_truth":     lambda: step_ground_truth(config),
        "prep_roboflow":    lambda: step_prep_roboflow(config),
        "import_roboflow":  lambda: step_import_roboflow(config),
        "train_unet":       lambda: step_train_unet(config),
        "train_yolo":       lambda: step_train_yolo(config, uav_available),
        "infer":            lambda: step_infer(config, uav_available),
        "postprocess":      lambda: step_postprocess(config),
        "fusion":           lambda: step_fusion(config),
        "change":           lambda: step_change(config),
        "validate":         lambda: step_validate(config),
        "dashboard":        lambda: step_dashboard(config),
    }

    if step not in dispatch:
        log.error(f"Unknown step: {step}")
        return

    dispatch[step]()
    elapsed = time.time() - t0
    log.info(f"<<< Step '{step}' completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()