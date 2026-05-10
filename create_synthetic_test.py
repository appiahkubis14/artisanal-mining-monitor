"""
create_synthetic_test.py
========================
Generate a complete set of synthetic data that matches the exact shapes,
file paths, dict structures, and rasterio metadata that every pipeline
script expects. Then run a smoke-test that calls each step end-to-end.

Usage
-----
    # 1. Generate synthetic data only
    python create_synthetic_test.py --mode generate

    # 2. Run the full pipeline smoke-test (generates data first)
    python create_synthetic_test.py --mode test

    # 3. Generate then run a single named step
    python create_synthetic_test.py --mode test --step preprocess

    # 4. Clean all synthetic outputs before regenerating
    python create_synthetic_test.py --mode clean

What is generated
-----------------
    data/raw/sentinel2/          synthetic_20230115.tif  (6 bands, 100x100)
    data/raw/sentinel2/          synthetic_20230215.tif  (6 bands, 100x100)
    data/raw/sentinel1/          synthetic_s1_20230115.tif (2 bands, 100x100)
    data/raw/uav/                synthetic_uav.tif       (3 bands, 2000x2000)
    data/boundary/               atewa_boundary.geojson
    data/ground_truth/           mining_sites.csv

All rasters use EPSG:32630 (UTM zone 30N, covers Atewa),
pixel size 10 m, origin at a point within Atewa Forest.

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants — geographic anchor (within Atewa Forest, UTM 30N)
# ---------------------------------------------------------------------------

EPSG_UTM   = "EPSG:32630"
EPSG_WGS84 = "EPSG:4326"

# UTM 30N coordinates of the top-left corner of our synthetic scene
# (real Atewa Forest area: approx 6.2°N, -0.55°E)
ORIGIN_X = 698_000.0   # easting  (m)
ORIGIN_Y = 686_000.0   # northing (m)

PIXEL_M     = 10.0          # Sentinel-2 pixel size
SAT_H       = 100           # satellite raster height (pixels)
SAT_W       = 100           # satellite raster width  (pixels)
UAV_SCALE   = 20            # UAV pixels per satellite pixel (gives ~0.5 m GSD)
UAV_H       = SAT_H * UAV_SCALE
UAV_W       = SAT_W * UAV_SCALE
UAV_PIXEL_M = PIXEL_M / UAV_SCALE

N_S2_BANDS  = 6             # Blue, Green, Red, NIR, SWIR1, SWIR2
N_S1_BANDS  = 2             # VV, VH
N_UAV_BANDS = 3             # R, G, B
N_DATES     = 2             # two acquisition dates


def _make_affine(pixel_m: float = PIXEL_M):
    """Build a rasterio Affine transform anchored at ORIGIN_X / ORIGIN_Y."""
    from rasterio.transform import from_origin
    return from_origin(ORIGIN_X, ORIGIN_Y, pixel_m, pixel_m)


def _make_meta(height, width, count, pixel_m=PIXEL_M, dtype="float32"):
    """Build a rasterio-compatible metadata dict."""
    from rasterio.crs import CRS
    from rasterio.transform import from_origin
    transform = from_origin(ORIGIN_X, ORIGIN_Y, pixel_m, pixel_m)
    return {
        "driver":    "GTiff",
        "dtype":     dtype,
        "width":     width,
        "height":    height,
        "count":     count,
        "crs":       CRS.from_epsg(32630),
        "transform": transform,
        "nodata":    None,
        "bounds":    (
            ORIGIN_X,
            ORIGIN_Y - height * pixel_m,
            ORIGIN_X + width  * pixel_m,
            ORIGIN_Y,
        ),
    }


# ---------------------------------------------------------------------------
# Random-seed reproducibility
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def _random_float(shape, lo=0.0, hi=1.0):
    return RNG.random(shape).astype(np.float32) * (hi - lo) + lo


def _mining_patch(array: np.ndarray, row: int, col: int, size: int = 15) -> np.ndarray:
    """
    Burn a synthetic mining signature into a [bands, H, W] float32 array.
    Mining areas are bright, low-vegetation, high-SAR backscatter patches.
    """
    r2 = min(row + size, array.shape[1])
    c2 = min(col + size, array.shape[2])
    n_bands = array.shape[0]
    for b in range(n_bands):
        # High red + SWIR (bare soil), low NIR (no veg)
        factor = 0.8 if b in (2, 4, 5) else 0.15
        array[b, row:r2, col:c2] = factor + RNG.random((r2 - row, c2 - col)) * 0.1
    return array


# ---------------------------------------------------------------------------
# File writers
# ---------------------------------------------------------------------------

def write_geotiff(path: Path, array: np.ndarray, meta: dict) -> None:
    """Write a [bands, H, W] float32 array to a GeoTIFF."""
    import rasterio
    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":    "GTiff",
        "dtype":     str(array.dtype),
        "width":     array.shape[2],
        "height":    array.shape[1],
        "count":     array.shape[0],
        "crs":       meta["crs"],
        "transform": meta["transform"],
        "compress":  "lzw",
    }
    if meta.get("nodata") is not None:
        profile["nodata"] = meta["nodata"]
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array)
    print(f"  wrote {path}  shape={array.shape}")


def write_boundary_geojson(path: Path) -> None:
    """Write a simple rectangular boundary GeoJSON (WGS84) around our scene."""
    from pyproj import Transformer
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert UTM corners to WGS84
    t = Transformer.from_crs(EPSG_UTM, EPSG_WGS84, always_xy=True)
    # corners: TL, TR, BR, BL
    corners_utm = [
        (ORIGIN_X,               ORIGIN_Y),
        (ORIGIN_X + SAT_W * PIXEL_M, ORIGIN_Y),
        (ORIGIN_X + SAT_W * PIXEL_M, ORIGIN_Y - SAT_H * PIXEL_M),
        (ORIGIN_X,               ORIGIN_Y - SAT_H * PIXEL_M),
    ]
    coords = [list(t.transform(x, y)) for x, y in corners_utm]
    coords.append(coords[0])   # close ring

    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type":       "Feature",
            "properties": {"name": "Atewa Forest Reserve (synthetic)"},
            "geometry":   {
                "type":        "Polygon",
                "coordinates": [coords],
            },
        }],
    }
    path.write_text(json.dumps(geojson, indent=2))
    print(f"  wrote {path}")


def write_ground_truth_csv(path: Path) -> None:
    """Write 5 synthetic GPS mining-site waypoints inside the scene."""
    from pyproj import Transformer
    path.parent.mkdir(parents=True, exist_ok=True)

    t = Transformer.from_crs(EPSG_UTM, EPSG_WGS84, always_xy=True)
    rows = []
    for i, (px, py) in enumerate([
        (20, 20), (25, 25), (70, 70), (72, 72), (50, 50)
    ]):
        x_utm = ORIGIN_X + px * PIXEL_M + PIXEL_M / 2
        y_utm = ORIGIN_Y - py * PIXEL_M - PIXEL_M / 2
        lon, lat = t.transform(x_utm, y_utm)
        rows.append({
            "latitude":     round(lat, 6),
            "longitude":    round(lon, 6),
            "date":         "2023-01-15",
            "site_type":    "galamsey",
            "active_status": "active",
            "size_m2":      500,
            "notes":        f"synthetic site {i+1}",
        })

    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} waypoints)")


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_synthetic_data(root: Path) -> None:
    """
    Create the full set of synthetic input files under `root`.

    Directory layout produced
    -------------------------
    root/data/raw/sentinel2/synthetic_20230115.tif
    root/data/raw/sentinel2/synthetic_20230215.tif
    root/data/raw/sentinel1/synthetic_s1_20230115.tif
    root/data/raw/sentinel1/synthetic_s1_20230215.tif
    root/data/raw/uav/synthetic_uav.tif
    root/data/boundary/atewa_boundary.geojson
    root/data/ground_truth/mining_sites.csv
    """
    print("\n" + "=" * 60)
    print("GENERATING SYNTHETIC TEST DATA")
    print("=" * 60)

    sat_meta = _make_meta(SAT_H, SAT_W, N_S2_BANDS, PIXEL_M)
    s1_meta  = _make_meta(SAT_H, SAT_W, N_S1_BANDS, PIXEL_M)
    uav_meta = _make_meta(UAV_H, UAV_W, N_UAV_BANDS, UAV_PIXEL_M)

    dates = ["20230115", "20230215"]

    # -- Sentinel-2 --
    print("\nSentinel-2 scenes:")
    s2_dir = root / "data" / "raw" / "sentinel2"
    for date in dates:
        arr = _random_float((N_S2_BANDS, SAT_H, SAT_W), 0.02, 0.6)
        arr = _mining_patch(arr, row=20, col=20)   # fake mining site 1
        arr = _mining_patch(arr, row=70, col=65)   # fake mining site 2
        write_geotiff(s2_dir / f"synthetic_{date}.tif", arr, sat_meta)

    # -- Sentinel-1 --
    print("\nSentinel-1 scenes:")
    s1_dir = root / "data" / "raw" / "sentinel1"
    for date in dates:
        # SAR: VV higher than VH; mining gives stronger backscatter
        vv = _random_float((1, SAT_H, SAT_W), 0.15, 0.55)
        vh = _random_float((1, SAT_H, SAT_W), 0.05, 0.35)
        vv[0, 20:35, 20:35] = 0.75    # stronger backscatter over mining
        arr = np.concatenate([vv, vh], axis=0)
        write_geotiff(s1_dir / f"synthetic_s1_{date}.tif", arr, s1_meta)

    # -- UAV orthomosaic --
    print("\nUAV orthomosaic:")
    uav_dir = root / "data" / "raw" / "uav"
    # Dense forest background: high green, low red/blue
    uav_arr = np.zeros((N_UAV_BANDS, UAV_H, UAV_W), dtype=np.float32)
    uav_arr[0] = _random_float((UAV_H, UAV_W), 0.05, 0.25)   # R low
    uav_arr[1] = _random_float((UAV_H, UAV_W), 0.30, 0.65)   # G high (forest)
    uav_arr[2] = _random_float((UAV_H, UAV_W), 0.05, 0.20)   # B low
    # Mining patches: bright, low saturation, brownish
    for (pr, pc) in [(400, 400), (1400, 1300)]:
        r2, c2 = pr + 200, pc + 200
        uav_arr[0, pr:r2, pc:c2] = _random_float((200, 200), 0.50, 0.75)  # R high
        uav_arr[1, pr:r2, pc:c2] = _random_float((200, 200), 0.45, 0.68)  # G medium
        uav_arr[2, pr:r2, pc:c2] = _random_float((200, 200), 0.30, 0.50)  # B medium
    write_geotiff(uav_dir / "synthetic_uav.tif", uav_arr, uav_meta)

    # -- Boundary GeoJSON --
    print("\nBoundary GeoJSON:")
    write_boundary_geojson(root / "data" / "boundary" / "atewa_boundary.geojson")

    # -- Ground-truth CSV --
    print("\nGround-truth CSV:")
    write_ground_truth_csv(root / "data" / "ground_truth" / "mining_sites.csv")

    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nAll files written under: {root}")
    print("Run the pipeline with:")
    print("  python main.py --step all")
    print("  -- or --")
    print("  python create_synthetic_test.py --mode test")


# ---------------------------------------------------------------------------
# Smoke-test runner
# ---------------------------------------------------------------------------

SMOKE_STEPS = [
    "preprocess",
    "tile",
    "features",
    "generate_masks",
    "ground_truth",
    "train_unet",
    "train_yolo",
    "infer",
    "postprocess",
    "fusion",
    "change",
    "validate",
    "dashboard",
]

# Steps that require GPU / real data / take too long for a quick smoke-test
# Set FAST=True to skip these and just verify the data-prep steps
FAST_SKIP = {"train_unet", "train_yolo"}


def patch_config_for_testing(config: dict) -> dict:
    """
    Override config values to make training steps fast enough for smoke-testing.
    Reduces epochs, tiles, and batch sizes to minimal values.
    """
    config = config.copy()

    config["training_unet"] = {
        **config.get("training_unet", {}),
        "epochs":                   2,
        "batch_size":               2,
        "early_stopping_patience":  2,
        "architecture":             "lightweight",   # smaller model
        "pretrained":               False,
    }
    config["training_yolo"] = {
        **config.get("training_yolo", {}),
        "epochs":                   2,
        "batch_size":               2,
        "early_stopping_patience":  2,
    }
    config["tiling"] = {
        **config.get("tiling", {}),
        "satellite_tile_size":  64,    # tiny tiles for speed
        "uav_tile_size":       128,
        "overlap":              0.0,
    }
    config["inference"] = {
        **config.get("inference", {}),
        "batch_size":  4,
    }
    return config


def run_smoke_test(
    root: Path,
    steps: list[str] | None = None,
    fast: bool = True,
) -> None:
    """
    Run each pipeline step against synthetic data and report pass/fail.

    Parameters
    ----------
    root  : project root (contains config.yaml and main.py)
    steps : list of step names to run (None = SMOKE_STEPS)
    fast  : if True, skip train_unet and train_yolo
    """
    import subprocess

    steps = steps or SMOKE_STEPS
    if fast:
        steps = [s for s in steps if s not in FAST_SKIP]

    print("\n" + "=" * 60)
    print("PIPELINE SMOKE-TEST (synthetic data)")
    print("=" * 60)

    results: dict[str, str] = {}
    start_all = time.time()

    for step in steps:
        print(f"\n>>> Running step: {step}")
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "main.py", "--step", step, "--log-level", "WARNING"],
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=300,        # 5-minute per-step timeout
            )
            elapsed = time.time() - t0
            if proc.returncode == 0:
                results[step] = f"PASS  ({elapsed:.1f}s)"
                print(f"  PASS  ({elapsed:.1f}s)")
            else:
                results[step] = f"FAIL  (exit {proc.returncode})"
                print(f"  FAIL  (exit {proc.returncode})")
                # Print last 20 lines of stderr
                err_lines = proc.stderr.strip().splitlines()
                for line in err_lines[-20:]:
                    print(f"    {line}")
        except subprocess.TimeoutExpired:
            results[step] = "TIMEOUT (>300s)"
            print(f"  TIMEOUT (>300s)")
        except Exception as exc:
            results[step] = f"ERROR  ({exc})"
            print(f"  ERROR  ({exc})")

    total = time.time() - start_all
    print("\n" + "=" * 60)
    print("SMOKE-TEST RESULTS")
    print("=" * 60)
    max_len = max(len(s) for s in results)
    for step, result in results.items():
        icon = "✓" if result.startswith("PASS") else "✗"
        print(f"  {icon}  {step:<{max_len}}  {result}")

    n_pass = sum(1 for r in results.values() if r.startswith("PASS"))
    n_fail = len(results) - n_pass
    print(f"\n  Total: {n_pass}/{len(results)} passed  ({total:.1f}s)")
    if n_fail:
        print(f"  {n_fail} step(s) failed — check output above for details.")
        sys.exit(1)
    else:
        print("  All steps passed.")


# ---------------------------------------------------------------------------
# Sentinel-file checker (verifies outputs exist after each step)
# ---------------------------------------------------------------------------

EXPECTED_OUTPUTS = {
    "preprocess": [
        "data/processed/sentinel2/sentinel2_processed.tif",
        "data/processed/sentinel1/sentinel1_processed.tif",
        "data/processed/uav/uav_highres.tif",
        "data/processed/uav/uav_10m.tif",
    ],
    "tile": [
        "data/tiles/satellite/tiling_progress.json",
    ],
    "features": [
        "data/features/feature_stack.tif",
        "data/features/feature_names.json",
    ],
    "generate_masks": [
        "data/masks/uav_mining_mask_fullres.tif",
        "data/masks/uav_mining_mask_10m.tif",
        "data/masks/mining_mask_10m.npy",
    ],
    "ground_truth": [
        "data/masks/full_scene_mask.npy",
    ],
    "train_unet": [
        "data/models/unet_last.pth",
    ],
    "infer": [
        "data/outputs",   # directory; individual files depend on dates
    ],
    "dashboard": [
        "data/outputs/dashboard.html",
    ],
}


def verify_outputs(root: Path, step: str) -> None:
    """Check that expected output files exist after a step completes."""
    expected = EXPECTED_OUTPUTS.get(step, [])
    if not expected:
        return
    print(f"\n  Output check for '{step}':")
    for rel_path in expected:
        full = root / rel_path
        exists = full.exists()
        icon = "✓" if exists else "✗"
        print(f"    {icon}  {rel_path}")


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

GENERATED_DIRS = [
    "data/raw/sentinel2",
    "data/raw/sentinel1",
    "data/raw/uav",
    "data/boundary",
    "data/ground_truth",
    "data/processed",
    "data/tiles",
    "data/features",
    "data/masks",
    "data/outputs",
    "data/models",
    "runs",
    "logs",
]


def clean(root: Path) -> None:
    """Remove all synthetic inputs and pipeline outputs."""
    print("\nCleaning synthetic data and pipeline outputs...")
    for rel in GENERATED_DIRS:
        d = root / rel
        if d.exists():
            shutil.rmtree(d)
            print(f"  removed {d}")
    print("Clean complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic data generator and pipeline smoke-tester.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Generate synthetic data only
  python create_synthetic_test.py --mode generate

  # Full smoke-test (generates data, runs all steps, skips training)
  python create_synthetic_test.py --mode test

  # Full smoke-test INCLUDING training (slow — uses 2 epochs)
  python create_synthetic_test.py --mode test --include-training

  # Test a single step
  python create_synthetic_test.py --mode test --step preprocess

  # Clean all generated files
  python create_synthetic_test.py --mode clean
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "test", "clean"],
        default="generate",
        help="generate = create synthetic data only; "
             "test = generate then run all steps; "
             "clean = remove all generated files",
    )
    parser.add_argument(
        "--step",
        default=None,
        help="Run only this step during --mode test (e.g. --step preprocess)",
    )
    parser.add_argument(
        "--include-training",
        action="store_true",
        help="Include train_unet and train_yolo in smoke-test (slow)",
    )
    args = parser.parse_args()

    root = Path(__file__).parent.resolve()

    if args.mode == "clean":
        clean(root)
        return

    if args.mode in ("generate", "test"):
        generate_synthetic_data(root)

    if args.mode == "test":
        steps = [args.step] if args.step else None
        fast = not args.include_training
        run_smoke_test(root, steps=steps, fast=fast)
        if steps:
            for s in (steps or SMOKE_STEPS):
                verify_outputs(root, s)


if __name__ == "__main__":
    main()
