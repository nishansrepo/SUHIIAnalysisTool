# SUHII Analysis Tool

A Python tool for calculating Surface Urban Heat Island Intensity (SUHII) from satellite imagery.

## What It Does

This tool quantifies the urban heat island effect by comparing Land Surface Temperature (LST) between urban and rural areas. It supports multiple peer-reviewed methodologies for defining urban cores and rural references, making results comparable across studies. [View complete documentation here](./SUHII_Analysis_Tool_Technical_Document.pdf)

**Key Features:**
- Three urban definition methods: LULC-based, NDVI-based, or combined LULC+NDVI
- Four rural reference methods: buffer, halo, three-ring zones, or in-city
- Adaptive elevation filtering to ensure climatic comparability
- Normalized UHI metric for cross-city comparisons
- Uncertainty quantification with standard error estimates

## Installation

```bash
pip install numpy geopandas rasterio scipy matplotlib
```

**Requirements:** Python 3.8+

## Quick Start

1. **Prepare your data** in an input directory:
   - `lst.tif` — Land Surface Temperature raster
   - `lulc.tif` — Land Use/Land Cover classification
   - `ndvi.tif` — NDVI raster
   - `elevation.tif` — Digital Elevation Model
   - `boundary.geojson` — Study area boundary

2. **Create `config.json`:**

```json
{
  "paths": {
    "input_dir": "data",
    "output_dir": "results",
    "lst_file": "lst.tif",
    "lulc_file": "lulc.tif",
    "ndvi_file": "ndvi.tif",
    "dem_file": "elevation.tif",
    "boundary_file": "boundary.geojson"
  },
  "lst_units": "celsius",
  "urban_selection": {
    "method": "lulc_ndvi",
    "urban_classes": [6],
    "water_class": 0,
    "nodata_value": 255,
    "ndvi_max_threshold": 0.3
  },
  "rural_selection": {
    "method": "buffer",
    "buffer_params": {
      "fixed_width_m": 10000
    },
    "vegetation_ndvi_threshold": 0.2,
    "exclude_urban_lulc_classes": true
  },
  "filters": {
    "mask_water": true,
    "use_elevation_correction": true,
    "elevation_params": {
      "initial_tolerance_m": 50,
      "max_tolerance_m": 200,
      "step_m": 25,
      "min_valid_pixels": 100
    }
  },
  "resampling": {
    "lst": "nearest",
    "ndvi": "bilinear",
    "dem": "bilinear"
  },
  "outputs": {
    "generate_debug_plots": true
  }
}
```

3. **Run:**

```bash
python suhii_tool.py
```

Or generate a config template:

```bash
python suhii_tool.py --generate-config config.json
```

## Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `urban_selection.method` | `lulc`, `ndvi`, `lulc_ndvi` | How to define urban pixels. |
| `rural_selection.method` | `buffer`, `halo`, `three_rings`, `incity` | How to define rural reference area. |
| `lst_units` | `celsius`, `kelvin`, `celsius_scaled` | Input LST format. |
| `filters.use_elevation_correction` | `true`/`false` | Filter rural pixels by elevation (±50m default). |

## Outputs

| File | Description |
|------|-------------|
| `results.json` | SUHII value, means, standard errors, methodology used |
| `SUHIIPixelDeviation.tif` | Deviation map (GeoTIFF) |
| `SUHIIPixelDeviation.png` | Deviation map visualization |
| `debug_plots/` | Diagnostic visualizations (if enabled) |

**Example `results.json`:**

```json
{
  "suhii": 3.45,
  "suhii_standard_error": 0.12,
  "urban_mean": 32.50,
  "rural_mean": 29.05,
  "methodology": {
    "urban_method": "lulc_ndvi",
    "rural_method": "buffer",
    "elevation_tolerance_m": 50.0
  }
}
```

## Methods Summary

| Method | Use Case |
|--------|----------|
| **buffer** | Standard approach — fixed ring around city boundary |
| **halo** | Large cities — excludes immediate urban footprint |
| **three_rings** | Multi-city studies — dynamic widths scale with city size |
| **incity** | Isolated cities — uses green spaces within boundary |

## References

- Raj & Yun (2024) — Buffer method, elevation filtering, incity comparison
- Fernandes et al. (2024) — Three-ring zone formulas
- Ahmad et al. (2024) — Normalized UHI metric
