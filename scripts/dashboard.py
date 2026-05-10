"""
dashboard.py
============
Generate an interactive HTML dashboard for Atewa mining detections.

Layers
------
- Basemap (OpenStreetMap / Satellite)
- Atewa boundary polygon
- Mining detections (colour-coded by confidence tier)
- Equipment detections (UAV)
- Change detection heatmap
- Site time-series popups
- Priority ranking list
- Time slider for multi-date comparison

Output
------
data/outputs/dashboard.html  — standalone HTML file (no server required)

Author : Atewa Mining Detection Pipeline
Thesis  : KNUST, Ghana
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd

from scripts.utils import get_logger, load_config, timer

log = get_logger(__name__)

# Colour scheme
CONFIDENCE_COLORS = {
    "HIGH": "#d32f2f",      # deep red
    "MEDIUM": "#f57c00",    # orange
    "LOW": "#fbc02d",       # amber
}
ACTIVITY_COLORS = {
    "ACTIVE": "#b71c1c",
    "LIKELY_ACTIVE": "#e64a19",
    "INACTIVE_INFRASTRUCTURE": "#5d4037",
    "INACTIVE": "#757575",
    "UNKNOWN": "#9e9e9e",
}
EQUIPMENT_COLORS = {
    "excavator": "#1a237e",
    "truck": "#283593",
    "water_pump": "#0d47a1",
    "settling_pond": "#4a148c",
    "pit": "#880e4f",
}


# ---------------------------------------------------------------------------
# Build popup HTML for a mining site
# ---------------------------------------------------------------------------

def _site_popup(row: pd.Series) -> str:
    """Build HTML popup content for a mining site polygon."""
    conf = row.get("confidence_fused", row.get("confidence", "N/A"))
    conf_str = f"{float(conf):.1%}" if conf != "N/A" else "N/A"
    area = row.get("area_ha", "N/A")
    area_str = f"{float(area):.2f} ha" if area != "N/A" else "N/A"
    activity = row.get("activity_fused", row.get("activity", "UNKNOWN"))
    priority = row.get("priority_rank_fused", row.get("priority_rank", "N/A"))
    eq_types = row.get("equipment_types", "")
    date = row.get("date", "unknown")
    site_id = row.get("site_id", "UNKNOWN")

    equipment_html = ""
    if eq_types:
        eq_list = [e.strip() for e in str(eq_types).split(",") if e.strip()]
        equipment_html = (
            "<b>Equipment detected:</b><br>"
            + "".join(f"&nbsp;&bull; {e}<br>" for e in eq_list)
        )

    color = ACTIVITY_COLORS.get(str(activity), "#9e9e9e")

    return f"""
    <div style="font-family: Arial, sans-serif; min-width:220px;">
        <h4 style="margin:4px 0; color:{color};">{site_id}</h4>
        <hr style="margin:4px 0;">
        <table style="font-size:12px; width:100%;">
            <tr><td><b>Date:</b></td><td>{date}</td></tr>
            <tr><td><b>Area:</b></td><td>{area_str}</td></tr>
            <tr><td><b>Confidence:</b></td><td>{conf_str}</td></tr>
            <tr><td><b>Activity:</b></td><td style="color:{color};">{activity}</td></tr>
            <tr><td><b>Priority rank:</b></td><td>#{priority}</td></tr>
        </table>
        {equipment_html}
    </div>
    """


def _equipment_popup(row: pd.Series) -> str:
    """Build HTML popup for a UAV equipment detection."""
    class_name = row.get("class_name", "unknown")
    conf = row.get("confidence", "N/A")
    conf_str = f"{float(conf):.1%}" if conf != "N/A" else "N/A"
    color = EQUIPMENT_COLORS.get(str(class_name), "#333")
    return f"""
    <div style="font-family: Arial, sans-serif;">
        <h4 style="margin:4px 0; color:{color};">{class_name.upper()}</h4>
        <hr style="margin:4px 0;">
        <b>Confidence:</b> {conf_str}<br>
    </div>
    """


def _change_popup(row: pd.Series) -> str:
    """Build popup for a change-tracked site."""
    tid = row.get("track_id", "UNKNOWN")
    first = row.get("first_seen", "?")
    last = row.get("last_seen", "?")
    n_obs = row.get("n_observations", "?")
    init_area = row.get("area_initial_ha", "?")
    latest_area = row.get("area_latest_ha", "?")
    expansion = row.get("total_expansion_ha", "?")
    rate = row.get("expansion_rate_ha_per_month", "?")
    status = row.get("status", "UNKNOWN")

    try:
        expansion_str = f"{float(expansion):+.3f} ha"
    except Exception:
        expansion_str = str(expansion)
    try:
        rate_str = f"{float(rate):+.4f} ha/month"
    except Exception:
        rate_str = str(rate)

    return f"""
    <div style="font-family: Arial, sans-serif; min-width:220px;">
        <h4 style="margin:4px 0;">{tid}</h4>
        <hr style="margin:4px 0;">
        <b>Status:</b> {status}<br>
        <b>First seen:</b> {first}<br>
        <b>Last seen:</b> {last}<br>
        <b>Observations:</b> {n_obs}<br>
        <b>Initial area:</b> {init_area} ha<br>
        <b>Latest area:</b> {latest_area} ha<br>
        <b>Total expansion:</b> {expansion_str}<br>
        <b>Expansion rate:</b> {rate_str}<br>
    </div>
    """


# ---------------------------------------------------------------------------
# Layer builders
# ---------------------------------------------------------------------------

def _add_boundary(m, boundary_path: Path) -> None:
    """Add Atewa boundary to folium map."""
    import folium

    if not boundary_path.exists():
        log.warning(f"Boundary file not found: {boundary_path}")
        return

    gdf = gpd.read_file(boundary_path).to_crs("EPSG:4326")
    folium.GeoJson(
        gdf.__geo_interface__,
        name="Atewa Boundary",
        style_function=lambda _: {
            "color": "#1b5e20",
            "weight": 3,
            "fillOpacity": 0.05,
            "fillColor": "#4caf50",
        },
        tooltip="Atewa Forest Reserve",
    ).add_to(m)
    log.info("Added boundary layer.")


def _add_detections_layer(
    m,
    geojson_path: Path,
    layer_name: str,
    date_str: str,
) -> None:
    """Add mining detection polygons as a styled GeoJson layer."""
    import folium

    if not geojson_path.exists():
        log.warning(f"Detection file not found: {geojson_path}")
        return

    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    if len(gdf) == 0:
        return

    feature_group = folium.FeatureGroup(name=layer_name, show=True)

    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue

        tier = row.get("confidence_tier", "LOW")
        color = CONFIDENCE_COLORS.get(str(tier), "#fbc02d")

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.5,
            },
            highlight_function=lambda _: {"weight": 4, "fillOpacity": 0.8},
            popup=folium.Popup(
                _site_popup(row), max_width=280
            ),
            tooltip=f"Conf: {row.get('confidence_tier','?')} | "
                    f"Area: {row.get('area_ha', '?'):.2f} ha",
        ).add_to(feature_group)

    feature_group.add_to(m)
    log.info(f"Added detections layer: {layer_name} ({len(gdf)} polygons)")


def _add_equipment_layer(m, equip_path: Path) -> None:
    """Add UAV equipment detection markers."""
    import folium

    if not equip_path.exists():
        log.warning(f"Equipment file not found: {equip_path}")
        return

    gdf = gpd.read_file(equip_path).to_crs("EPSG:4326")
    if len(gdf) == 0:
        return

    feature_group = folium.FeatureGroup(name="UAV Equipment Detections", show=True)

    for _, row in gdf.iterrows():
        if row.geometry is None:
            continue
        centroid = row.geometry.centroid
        class_name = str(row.get("class_name", "unknown"))
        color = EQUIPMENT_COLORS.get(class_name, "#333333")

        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=folium.Popup(_equipment_popup(row), max_width=200),
            tooltip=f"{class_name} ({row.get('confidence', 0):.0%})",
        ).add_to(feature_group)

    feature_group.add_to(m)
    log.info(f"Added equipment layer ({len(gdf)} detections).")


def _add_change_layer(m, change_path: Path) -> None:
    """Add change detection polygons with expansion status colouring."""
    import folium

    if not change_path.exists():
        log.warning(f"Change report not found: {change_path}")
        return

    gdf = gpd.read_file(change_path).to_crs("EPSG:4326")
    if len(gdf) == 0:
        return

    STATUS_COLORS = {
        "EXPANDING": "#b71c1c",
        "GROWN": "#e64a19",
        "NEW": "#f57c00",
        "STABLE": "#388e3c",
        "STABLE_INACTIVE": "#757575",
        "REDUCED": "#1565c0",
    }

    feature_group = folium.FeatureGroup(name="Change Detection", show=False)

    for _, row in gdf.iterrows():
        if row.geometry is None or row.geometry.is_empty:
            continue
        status = str(row.get("status", "STABLE"))
        color = STATUS_COLORS.get(status, "#9e9e9e")

        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.45,
                "dashArray": "6",
            },
            popup=folium.Popup(_change_popup(row), max_width=280),
            tooltip=f"{row.get('track_id','?')}: {status}",
        ).add_to(feature_group)

    feature_group.add_to(m)
    log.info(f"Added change detection layer ({len(gdf)} tracks).")


def _add_heatmap(m, geojson_path: Path, layer_name: str = "Density Heatmap") -> None:
    """Add mining probability heatmap from centroids."""
    try:
        from folium.plugins import HeatMap
    except ImportError:
        log.warning("folium.plugins.HeatMap not available – skipping heatmap.")
        return

    if not geojson_path.exists():
        return

    gdf = gpd.read_file(geojson_path).to_crs("EPSG:4326")
    if len(gdf) == 0:
        return

    centroids = gdf.geometry.centroid
    conf_col = "confidence_fused" if "confidence_fused" in gdf.columns else "confidence"
    heat_data = [
        [c.y, c.x, float(row.get(conf_col, 0.5))]
        for c, (_, row) in zip(centroids, gdf.iterrows())
        if c is not None and not c.is_empty
    ]

    if heat_data:
        HeatMap(
            heat_data,
            name=layer_name,
            radius=25,
            blur=20,
            max_zoom=13,
            show=False,
        ).add_to(m)
        log.info(f"Added heatmap layer ({len(heat_data)} points).")


def _add_legend(m) -> None:
    """Add HTML legend to the map."""
    import folium

    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background-color: white; padding: 12px 16px;
        border: 2px solid #ccc; border-radius: 8px;
        font-family: Arial, sans-serif; font-size: 12px;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <b style="font-size:13px;">Mining Detections</b><br>
        <i style="background:#d32f2f; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>High confidence<br>
        <i style="background:#f57c00; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>Medium confidence<br>
        <i style="background:#fbc02d; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>Low confidence<br>
        <br>
        <b>Change Status</b><br>
        <i style="background:#b71c1c; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>Expanding<br>
        <i style="background:#388e3c; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>Stable<br>
        <i style="background:#1565c0; width:14px; height:14px; display:inline-block; border-radius:2px; margin-right:6px;"></i>Reduced<br>
        <br>
        <b>Equipment (UAV)</b><br>
        <i style="background:#1a237e; width:14px; height:14px; display:inline-block; border-radius:50%; margin-right:6px;"></i>Excavator<br>
        <i style="background:#0d47a1; width:14px; height:14px; display:inline-block; border-radius:50%; margin-right:6px;"></i>Water pump<br>
        <i style="background:#4a148c; width:14px; height:14px; display:inline-block; border-radius:50%; margin-right:6px;"></i>Settling pond<br>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def _add_priority_panel(m, output_dir: Path, date_str: str) -> None:
    """Add a fixed HTML panel showing the top-priority sites."""
    import folium

    # Try to load priority list
    priority_csv = output_dir / f"priority_list_{date_str}.csv"
    fused_csv = output_dir / f"fused_summary_{date_str}.csv"
    csv_path = priority_csv if priority_csv.exists() else (
        fused_csv if fused_csv.exists() else None
    )

    if csv_path is None:
        return

    df = pd.read_csv(csv_path).head(10)
    rows_html = ""
    rank_col = "priority_rank_fused" if "priority_rank_fused" in df.columns else "priority_rank"
    site_col = "site_id" if "site_id" in df.columns else df.columns[0]

    for _, row in df.iterrows():
        rank = row.get(rank_col, "?")
        site = row.get(site_col, "?")
        area = row.get("area_ha", "?")
        try:
            area_str = f"{float(area):.2f} ha"
        except Exception:
            area_str = str(area)
        activity = row.get("activity_fused", row.get("activity", "?"))
        conf = row.get("confidence_fused", row.get("confidence", "?"))
        try:
            conf_str = f"{float(conf):.0%}"
        except Exception:
            conf_str = str(conf)
        act_color = ACTIVITY_COLORS.get(str(activity), "#555")
        rows_html += (
            f"<tr>"
            f"<td style='padding:2px 6px;'><b>#{rank}</b></td>"
            f"<td style='padding:2px 6px;'>{site}</td>"
            f"<td style='padding:2px 6px;'>{area_str}</td>"
            f"<td style='padding:2px 6px;color:{act_color};'>{activity}</td>"
            f"<td style='padding:2px 6px;'>{conf_str}</td>"
            f"</tr>"
        )

    panel_html = f"""
    <div id="priority-panel" style="
        position: fixed; top: 80px; right: 10px; z-index: 1000;
        background-color: white; padding: 10px 14px;
        border: 2px solid #ccc; border-radius: 8px;
        font-family: Arial, sans-serif; font-size: 11px;
        max-width: 420px; max-height: 320px; overflow-y: auto;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
    ">
        <b style="font-size:13px;">🚨 Priority Sites — {date_str}</b>
        <button onclick="document.getElementById('priority-panel').style.display='none';"
            style="float:right; border:none; background:none; cursor:pointer; font-size:14px;">✕</button>
        <table style="margin-top:6px; border-collapse:collapse; width:100%;">
            <thead>
                <tr style="border-bottom:1px solid #eee;">
                    <th style="padding:2px 6px;">Rank</th>
                    <th style="padding:2px 6px;">Site ID</th>
                    <th style="padding:2px 6px;">Area</th>
                    <th style="padding:2px 6px;">Activity</th>
                    <th style="padding:2px 6px;">Conf.</th>
                </tr>
            </thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    m.get_root().html.add_child(folium.Element(panel_html))


# ---------------------------------------------------------------------------
# Main dashboard builder
# ---------------------------------------------------------------------------

@timer
def build_dashboard(config: dict) -> Path:
    """Build and save the interactive HTML dashboard.

    Parameters
    ----------
    config : loaded config dict

    Returns
    -------
    Path to saved HTML file
    """
    try:
        import folium
        from folium.plugins import MiniMap, MeasureControl, Fullscreen
    except ImportError:
        log.error(
            "folium is not installed. Install with: pip install folium"
        )
        raise

    output_dir = Path(config["paths"]["outputs"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dash_cfg = config.get("dashboard", {})
    center = dash_cfg.get("center", [6.2, -0.55])
    zoom = dash_cfg.get("zoom_start", 11)

    log.info("Building interactive dashboard...")

    # ---- Base map ----
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,  # we add tiles manually for layer control
    )

    # Tile layers
    folium.TileLayer(
        "OpenStreetMap", name="OpenStreetMap", control=True
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite",
        control=True,
    ).add_to(m)

    # ---- Plugins ----
    MiniMap(toggle_display=True).add_to(m)
    MeasureControl().add_to(m)
    Fullscreen().add_to(m)

    # ---- Atewa boundary ----
    boundary_path = Path(config["paths"]["boundary"])
    _add_boundary(m, boundary_path)

    # ---- Detection layers (one per available date) ----
    fused_files = sorted(output_dir.glob("fused_detections_[0-9]*.geojson"))
    if not fused_files:
        fused_files = sorted(output_dir.glob("mining_detections_[0-9]*.geojson"))

    latest_date = "unknown"
    for fused_file in fused_files:
        stem = fused_file.stem
        for prefix in ("fused_detections_", "mining_detections_"):
            stem = stem.replace(prefix, "")
        date_str = stem
        latest_date = date_str
        _add_detections_layer(
            m, fused_file,
            layer_name=f"Mining Sites ({date_str})",
            date_str=date_str,
        )

    # ---- Heatmap (latest) ----
    if fused_files:
        _add_heatmap(m, fused_files[-1], "Density Heatmap")

    # ---- UAV equipment ----
    equip_path = output_dir / "equipment_detections.geojson"
    _add_equipment_layer(m, equip_path)

    # ---- Change detection ----
    change_path = output_dir / "change_report.geojson"
    _add_change_layer(m, change_path)

    # ---- Legend and priority panel ----
    _add_legend(m)
    _add_priority_panel(m, output_dir, latest_date)

    # ---- Summary stats title bar ----
    # Count sites across latest detection
    n_sites = 0
    total_ha = 0.0
    if fused_files:
        try:
            gdf = gpd.read_file(fused_files[-1])
            n_sites = len(gdf)
            total_ha = float(gdf.get("area_ha", pd.Series([0])).sum())
        except Exception:
            pass

    title_html = f"""
    <div style="
        position: fixed; top: 10px; left: 50%; transform: translateX(-50%);
        z-index: 1000; background-color: rgba(27, 94, 32, 0.92);
        color: white; padding: 8px 20px; border-radius: 20px;
        font-family: Arial, sans-serif; font-size: 13px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.4);
    ">
        <b>🌿 Atewa Forest Reserve — Illegal Mining Monitor</b>
        &nbsp;|&nbsp; Sites: <b>{n_sites}</b>
        &nbsp;|&nbsp; Total area: <b>{total_ha:.1f} ha</b>
        &nbsp;|&nbsp; Latest: <b>{latest_date}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(title_html))

    # ---- Layer control ----
    folium.LayerControl(collapsed=False).add_to(m)

    # ---- Save ----
    out_path = output_dir / "dashboard.html"
    m.save(str(out_path))
    log.info(f"\n{'='*50}\nDASHBOARD SAVED: {out_path}\n{'='*50}")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build interactive mining dashboard.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    build_dashboard(cfg)
