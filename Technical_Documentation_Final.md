# Surface Urban Heat Island Intensity (SUHII) Analysis Tool

## Technical Documentation

**Authors:** Nishan Sah, Sakina Mammadova, Nandita Kannapadi, Gabe Hafemann, Zoe Baker 
**Date:** December 2025  
**License:** 

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Methodology](#3-methodology)
   - 3.1 [Urban Area Definition](#31-urban-area-definition)
   - 3.2 [Rural Reference Selection](#32-rural-reference-selection)
   - 3.3 [Elevation Correction](#33-elevation-correction)
   - 3.4 [SUHII Calculation](#34-suhii-calculation)
4. [Technical Implementation](#4-technical-implementation)
   - 4.1 [Coordinate System Handling](#41-coordinate-system-handling)
   - 4.2 [Raster Alignment and Resampling](#42-raster-alignment-and-resampling)
   - 4.3 [LST Unit Conversion](#43-lst-unit-conversion)
   - 4.4 [Uncertainty Quantification](#44-uncertainty-quantification)
5. [Output Products](#5-output-products)
6. [Validation Considerations](#6-validation-considerations)
7. [Limitations](#7-limitations)
8. [References](#8-references)
9. [Appendix A: User Guide](#appendix-a-user-guide)
10. [Appendix B: Configuration Reference](#appendix-b-configuration-reference)

---

## 1. Introduction

The Surface Urban Heat Island Intensity (SUHII) Analysis Tool is a Python-based framework for quantifying the urban heat island effect using satellite-derived Land Surface Temperature (LST) data. The tool implements multiple peer-reviewed methodologies for defining urban and rural reference areas, enabling researchers to select approaches appropriate for their study context and compare results across methods.

### 1.1 Purpose and Scientific Basis

Urban Heat Islands (UHIs) represent one of the most significant anthropogenic modifications to local climate. The energetic basis of this phenomenon—characterized by increased heat storage in the urban fabric and reduced latent heat flux due to vegetation loss—has been well established in the literature.

This tool calculates Surface UHI Intensity (SUHII) using satellite thermal imagery (LST), which is distinct from Canopy UHI (air temperature). As noted by Zhou et al. (2014), SUHII exhibits distinct spatial patterns driven by surface cover, albedo, and anthropogenic heat, necessitating rigorous spatial definition of "urban" and "rural" baselines.

Quantifying UHI intensity is essential for:
- Urban climate adaptation planning
- Sustainable Development Goal 11 (Sustainable Cities) monitoring
- Heat-health early warning systems
- Urban greening intervention assessment

### 1.2 Design Philosophy

The tool was developed with three guiding principles:

1. **Methodological Transparency**: All algorithms are traceable to peer-reviewed literature with explicit citations.
2. **Flexibility**: Multiple methods are implemented to accommodate different data availability scenarios and research questions.
3. **Reproducibility**: Configuration-driven execution ensures analyses can be exactly replicated.

### 1.3 Scope

This tool calculates **Surface** UHI Intensity (SUHII) using satellite thermal imagery, distinct from **Canopy** UHI Intensity (CUHII) measured by air temperature sensors. SUHII represents the radiative temperature difference between urban and rural land surfaces as observed from space.

---

## 2. Theoretical Background

### 2.1 Defining Urban Heat Island Intensity

The Surface Urban Heat Island Intensity is defined as the difference between mean urban and mean rural land surface temperatures:

$$SUHII = \overline{LST}_{urban} - \overline{LST}_{rural}$$

Where:
- $\overline{LST}_{urban}$ = Mean LST of pixels classified as urban
- $\overline{LST}_{rural}$ = Mean LST of pixels in the rural reference area

This absolute difference formulation is the standard approach used in recent literature, including Raj and Yun (2024) and Fernandes et al. (2024). An alternative normalized formulation is used by Ahmad, Najar, and Ahmad (2024) — see Section 3.4.2.

### 2.2 Challenges in SUHII Estimation

Three fundamental challenges affect SUHII estimation:

#### 2.2.1 Urban Definition Ambiguity

"Urban" can be defined by administrative boundaries, population density, land cover classification, or spectral indices. Each definition yields different urban extents and consequently different SUHII values.

#### 2.2.2 Rural Reference Selection

The magnitude of SUHII is highly sensitive to the definition of the rural reference. Raj and Yun (2024) demonstrated that for highly urbanized cities (urban fraction > 50%), the choice between a surrounding buffer method and an in-city non-urban method can alter SUHII estimates by over 1.0°C. Their study compared buffer-based and in-city methods across five South Korean metropolitan areas.

#### 2.2.3 Confounding Factors

**Elevation Bias:** Mentaschi et al. (2022) and Raj and Yun (2024) highlight that elevation differences induce temperature biases via the environmental lapse rate, necessitating correction or filtering. Raj and Yun (2024) address this by filtering rural pixels to within ±50m of urban average elevation.

**Temporal Mismatch:** LULC products may not match LST acquisition date, introducing classification errors in rapidly changing urban environments.

---

## 3. Methodology

### 3.1 Urban Area Definition

The tool implements three methods for defining urban pixels, selectable via the `urban_selection.method` configuration parameter.

#### 3.1.1 LULC-Based Method (`method: "lulc"`)

Urban pixels are identified using land use/land cover classification products. The user specifies which class codes represent urban/built-up areas.

**Algorithm:**
```
urban_mask = pixel ∈ {urban_classes}
```

**Applicable LULC Products:**

| Product | Urban Class Code | Resolution |
|---------|------------------|------------|
| Dynamic World | 6 (Built Area) | 10m |
| ESA WorldCover | 50 (Built-up) | 10m |
| MODIS MCD12Q1 | 13 (Urban) | 500m |
| Copernicus GLC | 50 (Built-up) | 100m |

**Scientific Rationale for Dynamic World:**

While static products (e.g., ESA WorldCover) are supported, the tool explicitly recommends and supports the Dynamic World dataset as a superior alternative for dynamic urban analysis.

Dynamic World is a near-real-time (NRT) 10m resolution global land use land cover dataset generated using a deep learning model (Fully Convolutional Neural Network) applied to Sentinel-2 imagery (Brown et al. 2022).

The justification for using this advanced product over traditional static maps is threefold:

1. **Methodological Robustness:** Recent literature validates the superiority of machine learning and deep learning approaches for urban LULC classification. Vignesh et al. (2022) demonstrated that Deep Learning models (specifically Long Short-Term Memory Recurrent Neural Networks) applied to multispectral satellite data (Landsat-8) significantly outperform traditional classifiers in capturing complex urban heterogeneity. Similarly, Kumar et al. (2025) validated the efficacy of machine learning (Random Forest) on Sentinel-2 imagery within Google Earth Engine for accurate urban mapping. Dynamic World operationalizes its advanced DL/ML methodologies on a global scale.

2. **Temporal Synchronization:** SUHII analysis requires precise temporal alignment between the thermal image (LST) and the land cover mask. Traditional annual composites may fail to capture rapid urbanization or seasonal phenology changes. Dynamic World allows the generation of an LULC map for the specific date range of the LST acquisition, minimizing temporal mismatch errors.

3. **Probabilistic Definition:** Unlike binary classification products, Dynamic World provides probability scores for each class. This allows the tool to define "Built Area" based on a confidence threshold (e.g., built_probability > 0.5), offering greater sensitivity in mixed-pixel transition zones common in peri-urban areas.

**Literature Examples:**
- Raj and Yun (2024) use Landsat-8 land cover classification to identify urban pixels in their South Korean study
- Mentaschi et al. (2022) use GHSL (Global Human Settlements Layer) with >15% built-up probability threshold in their global analysis

**Advantages:**
- Leverages validated classification products
- Consistent definition across studies using same product
- Includes all built-up pixels regardless of vegetation

**Limitations:**
- Classification accuracy varies by product and region
- May include vegetated urban areas (parks, gardens)
- Dependent on temporal alignment with LST acquisition

#### 3.1.2 NDVI-Based Method (`method: "ndvi"`)

Urban pixels are identified using a Normalized Difference Vegetation Index (NDVI) threshold, based on the assumption that built-up areas have low vegetation cover.

**Algorithm:**
```
urban_mask = NDVI < ndvi_max_threshold
```

**Default Threshold:** 0.3

**Scientific Basis:**

While most studies use LULC products, Ahmad, Najar, and Ahmad (2024) and Keerthi Naidu and Chundeli (2023) demonstrate a strong, statistically significant inverse correlation between LST and NDVI (Pearson $r \approx -0.6$ to $-0.8$ in summer). Ahmad, Najar, and Ahmad (2024) report correlations of r = -0.673 in summer for Delhi. This strong linearity supports the use of low NDVI as a proxy for built-up/impervious surfaces when high-resolution LULC data is unavailable. 

Furthermore, Tempa et al. (2024) confirm the robustness of NDVI as a monitoring tool for urban transitions, showing that significant drops in NDVI (e.g., 75% loss in "very healthy vegetation") directly correspond to built-up expansion in rapidly urbanizing regions.

**Note:** Ahmad, Najar, and Ahmad (2024) analyzed LST-NDVI relationships using established techniques but did not originate the NDVI-based urban definition method itself, which is a common approach in thermal remote sensing literature predating recent studies.

**Advantages:**
- Does not require LULC classification product
- Contemporaneous with LST (if derived from same image)
- Captures impervious surface fraction

**Limitations:**
- **Critical:** Cannot distinguish urban from bare soil, rock, or water
- Threshold selection is subjective and context-dependent
- Seasonal vegetation phenology affects results

**Warning:** This method should only be used when LULC data is unavailable. Results should be interpreted with caution in semi-arid regions where bare soil is prevalent.

#### 3.1.3 Combined LULC + NDVI Method (`method: "lulc_ndvi"`)

This hybrid approach uses LULC classification to identify built-up areas, then applies an NDVI filter to exclude vegetated pixels within those areas (e.g., urban parks, gardens, tree-lined streets).

**Algorithm:**
```
urban_mask = (pixel ∈ {urban_classes}) AND (NDVI < ndvi_max_threshold)
```

**Rationale:** Urban parks and green spaces, while administratively urban, exhibit thermal behavior more similar to rural vegetation than impervious surfaces. Including them in the urban sample would underestimate true SUHII for built infrastructure.

**Scientific Basis:**

Zhou et al. (2014) emphasize that the "urban" thermal signal is driven by impervious surfaces. Mentaschi et al. (2022) exclude water from their urban definition to avoid thermal contamination. Yang and Yao (2022) demonstrate that even within urban boundaries, vegetated patches maintain distinct lower LSTs. This method synthesizes these findings by strictly isolating the impervious fraction of the LULC urban class, ensuring the urban temperature mean is not biased by urban green spaces, which Keerthi Naidu and Chundeli (2023) show can act as cool islands.

**Advantages:**
- Most rigorous isolation of impervious surfaces
- Excludes thermally-anomalous urban vegetation
- Consistent with physical understanding of UHI drivers

**Limitations:**
- Requires both LULC and NDVI inputs
- More aggressive filtering reduces urban sample size
- May exclude legitimate urban pixels with incidental vegetation

**Recommendation:** This is the preferred method when both LULC and NDVI data are available.

#### 3.1.4 Water Body Exclusion

All methods optionally exclude water bodies from the urban mask when `filters.mask_water: true`. Water has distinct thermal properties (high heat capacity, evaporative cooling) that would bias urban LST estimates.

Raj and Yun (2024) and Mentaschi et al. (2022) explicitly exclude water bodies from both urban and rural reference areas in their methodologies.

### 3.2 Rural Reference Selection

The rural reference area defines the "baseline" temperature against which urban temperatures are compared. Four methods are implemented, selectable via `rural_selection.method`.

#### 3.2.1 Fixed Buffer Method (`method: "buffer"`)

**Primary Reference:** Raj and Yun (2024)

A fixed-width annular buffer is created around the urban boundary. Rural pixels are selected from within this buffer based on vegetation and elevation criteria.

**Algorithm:**
```
rural_geometry = Buffer(urban_boundary, width) - urban_boundary
```

**Parameters:**
- `fixed_width_m`: Buffer width in meters (default: 10,000m)

Raj and Yun (2024) use a buffer with "ichimyeonjuk roughly equivalent to the city area" (~10 km) for South Korean metropolitan cities, combined with elevation and land cover filtering. Zhou et al. (2014) also utilized an equal-area buffer approach for 32 major Chinese cities, validating the geometric consistency of annular buffers for comparative analysis.

**Advantages:**
- Simple and widely used
- Easy to replicate across studies
- Computationally efficient

**Limitations:**
- Arbitrary buffer width selection
- Does not account for city size variation
- May include urbanized areas in buffer for polycentric regions

#### 3.2.2 Urban Halo Method (`method: "halo"`)

This method recognizes that the immediate periphery of cities often experiences thermal contamination from the urban heat island "footprint." An inner exclusion zone is skipped before the rural buffer begins.

**Algorithm:**
```
inner_ring = Buffer(urban_boundary, min_distance)
outer_ring = Buffer(urban_boundary, min_distance + width)
rural_geometry = outer_ring - inner_ring
```

**Parameters:**
- `min_distance_from_edge_m`: Inner exclusion distance (default: 0m)
- `fixed_width_m`: Buffer width after exclusion zone

**Rationale:** UHI effects extend beyond administrative boundaries through heat advection and modified boundary layer dynamics. Excluding the immediate urban periphery provides a "cleaner" rural reference.

**Scientific Basis:**

Mentaschi et al. (2022) employ a large (70km) kernel to ensure the reference is taken far beyond the "urban footprint," acknowledging that the thermal impact of a city extends beyond its physical boundary. This tool implements that concept as a user-configurable exclusion zone (`min_distance_from_edge_m`), allowing users to replicate the "far-field" reference approach necessary to avoid thermal contamination from suburban sprawl or advection.

**Note on Terminology:** This approach is sometimes confused with simple kernel-based methods (e.g., Mentaschi et al. 2022 use a kernel approach, not a true halo method with explicit inner exclusion). The distinction is important for methodological transparency.

**Recommendation:** Use 1-3 km inner exclusion for large cities (>500 km²).

#### 3.2.3 Three-Ring Method (`method: "three_rings"`)

**Primary Reference:** Fernandes et al. (2024)

This method creates dynamically-sized buffer zones scaled to city area, enabling consistent comparisons across cities of different sizes. Three concentric zones are defined:

1. **Ua (Urban Adjacent):** Immediate urban periphery
2. **FUa (Future Urban Adjacent):** Transition zone
3. **PUa (Peri-Urban):** Rural reference zone

**Formulas from Fernandes et al. (2024):**

$$W_{Ua} = 0.25 \sqrt{A}$$

$$W_{FUa} = 0.25 \sqrt{A + A_{Ua}}$$

$$W_{PUa} = 1.5\sqrt{A} - W_{FUa} - W_{Ua}$$

Where:
- $A$ = Urban area (km²)
- $A_{Ua}$ = Area of the Ua ring (km²)
- $W$ = Width of each zone (km)

**Parameters:**
- `ring_type`: Which zone to use as rural reference (`ua`, `fua`, or `pua`)

**Advantages:**
- Physically-motivated scaling with city size
- Enables consistent cross-city comparisons
- Captures urban-rural gradient

**Limitations:**
- More complex to implement and explain
- May extend beyond available imagery for small scenes
- Assumes roughly circular city morphology

#### 3.2.4 In-City Method (`method: "incity"`)

**Primary References:** Raj and Yun (2024); Ahmad, Najar, and Ahmad (2024)

Rural reference pixels are selected from within the administrative boundary itself, representing non-urban land cover (parks, urban forests, undeveloped land) inside the city.

Raj and Yun (2024) refer to this as "Method 1: Non-urban areas within city limits" and compare it directly against the buffer approach ("Method 2"). They use Landsat-8 classification to identify non-urban pixels within city boundaries and find it yields lower SUHII values for highly urbanized cities like Seoul.

**Algorithm:**
```
rural_geometry = urban_boundary  # Same as urban geometry
rural_mask = rural_geometry AND (NOT urban_pixels) AND vegetation_criteria
```

**Normalized UHI Calculation:**

When using the incity method, the tool additionally calculates the Normalized UHI following Ahmad, Najar, and Ahmad (2024):

$$UHI_{normalized} = \frac{\overline{LST}_{urban} - \overline{LST}_{study}}{\sigma_{study}}$$

Where:
- $\overline{LST}_{urban}$ = Mean LST of urban-classified pixels
- $\overline{LST}_{study}$ = Mean LST of entire study area (administrative boundary)
- $\sigma_{study}$ = Standard deviation of LST across study area

Ahmad, Najar, and Ahmad (2024) use this normalized formulation in their Delhi study, expressing UHI as standard deviations from the mean rather than absolute temperature difference.

**Advantages:**
- All pixels within same administrative unit
- Captures intra-urban temperature variation
- Normalized UHI enables cross-city comparison regardless of absolute temperatures
- Useful when surrounding rural areas are unavailable or dissimilar

**Limitations:**
- "Rural" pixels may not represent true rural conditions
- Limited sample size in highly urbanized cities
- Confounds UHI with urban green space cooling effect

**Key Finding from Raj and Yun (2024):** Cities with >50% urban land cover show >1°C difference between buffer and incity methods. For less urbanized cities (<40% urban), both methods produce comparable results.

**Use Case:** Most appropriate for studies focused on intra-urban variation or cities where the surrounding landscape is climatically dissimilar (e.g., coastal cities, cities in deserts). The normalized UHI output is particularly valuable for comparative studies across multiple cities.

#### 3.2.5 Rural Pixel Filtering

Regardless of geometry method, rural pixels undergo additional filtering:

1. **Vegetation Filter:** NDVI ≥ `vegetation_ndvi_threshold` (default: 0.2)
2. **Water Exclusion:** Remove water bodies (see §3.2.7 for rationale)
3. **LULC Exclusion:** Remove urban LULC classes from buffer (optional)
4. **Nodata Exclusion:** Remove pixels with invalid LULC values
5. **Distance Filter:** Minimum distance from urban edge (optional)
6. **Elevation Filter:** Within tolerance of urban mean elevation (see §3.3)

#### 3.2.6 Minimum Rural Distance Filter

**Parameter:** `min_rural_distance_m`

For `buffer` and `three_rings` methods, an optional minimum distance filter excludes rural pixels too close to the urban edge. This serves a similar purpose to the `halo` method's inner exclusion zone but operates at the pixel level rather than geometry level.

**Implementation:** Uses Euclidean Distance Transform (EDT) via `scipy.ndimage.distance_transform_edt` to efficiently compute distance from each pixel to the nearest urban pixel.

#### 3.2.7 Water Body Exclusion from Rural Reference

**Parameter:** `filters.mask_water`

Water bodies are excluded from the rural reference area when `mask_water: true`. This filter is critical for accurate SUHII estimation and is applied in addition to urban water exclusion (§3.1.4).

**Scientific Rationale:**

Water bodies exhibit fundamentally different thermal behavior compared to vegetated land surfaces due to three key physical properties:

1. **High Thermal Inertia:** Water has a specific heat capacity approximately 4× greater than soil and vegetation. This causes water surfaces to warm and cool much more slowly than surrounding land, resulting in systematically different diurnal and seasonal temperature patterns.

2. **Evaporative Cooling:** Open water surfaces experience continuous latent heat flux through evaporation, which suppresses surface temperatures relative to vegetated land under equivalent radiative forcing. This cooling effect is independent of vegetation health or land cover type.

3. **Low Surface Emissivity Variation:** Unlike vegetated surfaces where emissivity varies with moisture content and vegetation type, water maintains relatively constant emissivity (~0.98), potentially introducing systematic retrieval differences in LST products.

**Literature Support:**

Raj and Yun (2024) and Mentaschi et al. (2022) explicitly exclude water bodies from both urban and rural reference areas in their methodologies. The exclusion ensures that the rural reference represents the thermal signature of vegetated land surfaces, which is the appropriate baseline for quantifying the urban-rural temperature contrast caused by impervious surface cover.

Including water bodies in the rural reference would artificially depress the rural mean LST (due to evaporative cooling), resulting in inflated SUHII values that do not accurately reflect the true urban heat island effect relative to surrounding vegetation.

**Implementation:**

When LULC data is available, water pixels are identified using the `water_class` parameter (default: 0 for Dynamic World). When LULC is unavailable, the tool uses a conservative approach by excluding pixels with NDVI ≤ 0, as water typically exhibits negative or near-zero NDVI values due to strong absorption in the near-infrared band.

### 3.3 Elevation Correction

**Primary References:** Raj and Yun (2024); Mentaschi et al. (2022)

Elevation differences between urban and rural areas introduce systematic temperature biases due to the environmental lapse rate (~6.5°C per 1000m elevation gain). The tool implements adaptive elevation filtering to ensure climatic comparability.

To account for the environmental lapse rate, it is critical to compare urban and rural pixels at similar elevations. Raj and Yun (2024) enforce a strict filter, excluding rural pixels outside ±50m of the urban average elevation. Mentaschi et al. (2022) define the standard environmental lapse rate of -6.5 K/km. Yang and Yao (2022) further validate this necessity in their global analysis of 346 cities, demonstrating that failing to control for elevation in rural reference selection introduces significant bias, particularly in cities with diverse topography (e.g., coastal vs. inland).

**Alternative Approach:** Mentaschi et al. (2022) use a different methodology—they apply environmental lapse rate correction (-6.5 K/1000m) to adjust all LST values to a common reference elevation before comparing urban and rural temperatures. Both approaches address the same underlying issue but through different mechanisms: filtering (this tool, Raj and Yun) vs. correction (Mentaschi et al.).

#### 3.3.1 Reference Elevation Calculation

The reference elevation is calculated as the mean elevation of urban-classified pixels:

$$\overline{E}_{urban} = \frac{1}{n}\sum_{i=1}^{n} DEM_i \quad \text{where } i \in \text{urban pixels}$$

**Rationale:** Raj and Yun (2024) filter rural pixels to within ±50m of the urban average elevation. Using actual urban pixels (rather than the entire boundary) ensures the elevation reference reflects the built environment.

#### 3.3.2 Adaptive Tolerance Algorithm

This tool implements the Raj and Yun (2024) filtering approach but makes it adaptive. Rural pixels are filtered to those within an elevation tolerance of the urban reference. The algorithm iteratively expands tolerance until a minimum pixel count is reached:

```
current_tolerance = initial_tolerance
while current_tolerance ≤ max_tolerance:
    candidate_pixels = rural_pixels where |DEM - urban_elev| ≤ current_tolerance
    if count(candidate_pixels) ≥ min_valid_pixels:
        accept candidate_pixels
        break
    current_tolerance += step
```

**Parameters:**
- `initial_tolerance_m`: Starting tolerance (default: 50m, per Raj and Yun, 2024)
- `max_tolerance_m`: Maximum tolerance (default: 200m)  
- `step_m`: Tolerance increment (default: 25m)
- `min_valid_pixels`: Minimum required pixels (default: 100)

**Scientific Basis:** Raj and Yun (2024) explicitly state they exclude pixels "outside ±50m of urban average elevation" in their buffer method for South Korean cities. The adaptive approach allows the tool to work in topographically challenging regions where strict thresholds would exclude too many rural pixels (common in mountainous terrain), while still attempting to find pixels within ±50m (default) before relaxing the constraint. The tool reports the final tolerance used for transparency.

#### 3.3.3 Elevation Correction Limitations

The elevation filter assumes temperature varies primarily with elevation. This may not hold when:
- Strong temperature inversions exist
- Coastal effects dominate (sea breeze, marine layer)
- Local topographic effects (cold air pooling) are significant

Raj and Yun (2024) note that coastal cities like Busan show different SUHII patterns due to sea breeze effects, which may not be fully captured by elevation filtering alone.

### 3.4 SUHII Calculation

#### 3.4.1 Primary Metric (Absolute Difference)

The primary output is the scalar SUHII value representing the absolute temperature difference:

$$SUHII = \overline{LST}_{urban} - \overline{LST}_{rural}$$

**Calculation Method:**

Mean Urban LST ($\overline{LST}_{urban}$): Calculated from all pixels in the urban mask using `numpy.nanmean()` to handle missing data appropriately.

Mean Rural LST ($\overline{LST}_{rural}$): Calculated from all pixels in the rural reference mask that passed the elevation filter.

SUHII: $\overline{LST}_{urban} - \overline{LST}_{rural}$

Units are degrees Celsius (°C). This formulation is used by Raj and Yun (2024) and Fernandes et al. (2024).

#### 3.4.2 Normalized UHI (In-City Method Only)

For the `incity` method, an additional normalized metric is calculated following Ahmad, Najar, and Ahmad (2024):

$$UHI_{normalized} = \frac{\overline{LST}_{urban} - \overline{LST}_{study}}{\sigma_{study}}$$

Ahmad, Najar, and Ahmad (2024) report normalized UHI values for Delhi ranging from 8.13 to 10.29 across seasons.

| Metric | Units | Interpretation |
|--------|-------|----------------|
| SUHII (absolute) | °C | "Urban is X degrees warmer than rural" |
| UHI (normalized) | σ (dimensionless) | "Urban is X standard deviations above mean" |

**When to use each:**
- **Absolute SUHII:** When communicating temperature differences to planners, public health officials, or general audiences
- **Normalized UHI:** When comparing UHI intensity across cities with different climates or baseline temperatures

#### 3.4.3 Deviation Maps

To visualize spatial heterogeneity, pixel-wise deviation maps are generated, all referenced to the rural mean:

$$Deviation_i = LST_i - \overline{LST}_{rural}$$

Where $Deviation_i$ represents the heat island intensity at pixel location $i$.

Three spatially-explicit deviation maps are generated:

| Output | Spatial Extent | Use Case |
|--------|----------------|----------|
| `SUHIIUrbanFullDeviation` | Full urban boundary + filtered rural | Visualizing entire urban area |
| `SUHIIPixelDeviation` | Filtered urban + filtered rural only | Analysis of actual sample pixels |
| `SUHIIAll` | Urban boundary + entire buffer geometry | Complete analysis domain |

---

## 4. Technical Implementation

### 4.1 Coordinate System Handling

#### 4.1.1 UTM Projection

All spatial operations are performed in a locally-appropriate Universal Transverse Mercator (UTM) projection to ensure metric accuracy for buffer calculations. The tool automatically detects the correct UTM zone based on the city centroid latitude and longitude.

**Algorithm:**
```python
zone = floor((longitude + 180) / 6) + 1
EPSG = 32600 + zone  # Northern hemisphere
EPSG = 32700 + zone  # Southern hemisphere
```

**Rationale:** Buffer distances specified in meters require a projected coordinate system. UTM provides <0.1% distance distortion within each 6° zone.

#### 4.1.2 Coordinate Flow

```
Input Data (various CRS)
    ↓
Boundary → UTM (geometric operations)
Rasters → UTM (reprojection)
    ↓
All operations in UTM
    ↓
Output GeoTIFFs (UTM)
```

### 4.2 Raster Alignment and Resampling

#### 4.2.1 Grid Alignment

To ensure scientific validity of pixel-wise operations ($LST - \overline{LST}_{rural}$), disparate input datasets (LULC, NDVI, DEM) must be perfectly aligned. All input rasters are aligned to a common grid defined by the LST raster. This ensures pixel-perfect correspondence for array operations.

**Process:**
1. LST raster reprojected to UTM (defines master grid)
2. NDVI, DEM, LULC reprojected and resampled to match LST grid using `rasterio.warp.reproject`
3. All arrays share identical dimensions, transform, and CRS

#### 4.2.2 Resampling Methods

The tool supports configurable resampling methods per data layer:

| Method | Algorithm | Appropriate For |
|--------|-----------|-----------------|
| `nearest` | Nearest neighbor | Categorical data, measured values |
| `bilinear` | Weighted average of 4 neighbors | Smooth continuous fields |

**Configuration:**
```json
"resampling": {
    "lst": "nearest",
    "ndvi": "bilinear",
    "dem": "bilinear"
}
```

**Scientific Rationale for LST Resampling:**

Bilinear interpolation creates weighted average values that were never actually measured. At an urban-rural boundary:

```
Urban pixel (35°C) | Rural pixel (28°C)
        ↓ Bilinear resampling ↓
     Artificial 31.5°C pixel created
```

This artificially smooths the urban-rural temperature gradient, potentially underestimating SUHII magnitude at boundaries. **Nearest neighbor resampling is recommended for LST** to preserve actual radiometric measurements.

LULC data always uses nearest neighbor resampling regardless of configuration, as interpolating categorical class codes is meaningless.

### 4.3 LST Unit Conversion

The tool supports three input LST formats:

| Format | Conversion | Common Sources |
|--------|------------|----------------|
| `celsius` | None | Pre-processed data |
| `kelvin` | LST - 273.15 | MODIS, Landsat Collection 2 |
| `celsius_scaled` | LST / 100 | Some GEE exports |

**Configuration:**
```json
"lst_units": "kelvin"
```

All internal calculations and outputs are in degrees Celsius. The user must specify the correct input unit format in the configuration; the tool applies the appropriate conversion based on this setting.

### 4.4 Uncertainty Quantification

#### 4.4.1 Standard Error Calculation

The tool reports standard error of the SUHII estimate, assuming urban and rural samples are independent:

$$SE_{SUHII} = \sqrt{SE_{urban}^2 + SE_{rural}^2}$$

Where:

$$SE = \frac{\sigma}{\sqrt{n}}$$

#### 4.4.2 Reported Metrics

| Metric | Description |
|--------|-------------|
| `suhii_standard_error` | Standard error of SUHII difference |
| `urban_std` / `rural_std` | Sample standard deviations |
| `urban_valid_pixels` / `rural_valid_pixels` | Non-NaN pixel counts |

#### 4.4.3 Limitations

The standard error calculation assumes:
- Independent, identically distributed samples
- No spatial autocorrelation

These assumptions are violated in practice (adjacent pixels are correlated). The reported SE should be considered a lower bound; true uncertainty is likely higher.

---

## 5. Output Products

### 5.1 Scalar Results (`results.json`)

```json
{
    "suhii": 3.45,
    "suhii_standard_error": 0.12,
    "urban_mean": 32.50,
    "urban_std": 2.10,
    "urban_elevation_m": 1650,
    "urban_pixels_total": 15000,
    "urban_pixels_valid": 14850,
    "rural_mean": 29.05,
    "rural_std": 1.80,
    "rural_elevation_m": 1680,
    "rural_pixels_total": 8000,
    "rural_pixels_valid": 7950,
    "methodology": {
        "urban_method": "lulc_ndvi",
        "rural_method": "buffer",
        "elevation_tolerance_m": 50.0,
        "min_rural_distance_m": 0.0,
        "lst_input_units": "celsius",
        "resampling": {
            "lst": "nearest",
            "ndvi": "bilinear",
            "dem": "bilinear",
            "lulc": "nearest"
        }
    }
}
```

**Additional output for `incity` method only:**

When `rural_method: "incity"` is used, an additional `normalized_uhi` object is included following Ahmad, Najar, and Ahmad (2024):

```json
{
    "suhii": 3.87,
    "normalized_uhi": {
        "uhi_normalized": 1.42,
        "study_area_mean": 30.15,
        "study_area_std": 2.72,
        "formula": "(LST_urban - LST_study_area_mean) / LST_study_area_std",
        "citation": "Ahmad, Najar, and Ahmad (2024)",
        "units": "standard deviations (σ)"
    },
    ...
}
```

**Interpretation:** A `uhi_normalized` value of 1.42 means the urban mean temperature is 1.42 standard deviations above the study area mean. This dimensionless metric enables comparison across cities with different absolute temperature ranges.

### 5.2 Raster Products

| File | Format | Description |
|------|--------|-------------|
| `SUHIIUrbanFullDeviation.tif` | GeoTIFF (Float32) | Deviation map, full urban extent |
| `SUHIIPixelDeviation.tif` | GeoTIFF (Float32) | Deviation map, filtered pixels only |
| `SUHIIAll.tif` | GeoTIFF (Float32) | Deviation map, entire analysis domain |
| `aligned_*.tif` | GeoTIFF | Intermediate aligned rasters |

### 5.3 Visualization Products

| File | Description |
|------|-------------|
| `SUHIIUrbanFullDeviation.png` | Deviation map visualization |
| `SUHIIPixelDeviation.png` | Filtered pixel visualization |
| `SUHIIAll.png` | Full domain visualization |

All visualizations use a divergent colormap (RdBu_r) centered on zero deviation from the rural mean.

### 5.4 Debug Products (optional)

When `outputs.generate_debug_plots: true`:

| File | Description |
|------|-------------|
| `lulc_check.png` | LULC classification verification |
| `mask_urban_final.png` | Final urban pixel selection |
| `mask_rural_final.png` | Final rural pixel selection |
| `buffer_geometry.png` | Buffer/rural geometry visualization |
| `final_analysis_area.png` | Combined urban + rural mask |
| `distance_from_urban.png` | Distance raster (if distance filter used) |
| `analysis.log` | Detailed processing log |

---

## 6. Validation Considerations

### 6.1 Comparison with Literature Values

SUHII values should be contextualized against published studies in the study area.

### 6.2 Ground Validation

Where possible, validate satellite-derived SUHII against:
- Meteorological station air temperature differences
- Vehicle traverse measurements
- Flux tower observations

Raj and Yun (2024) validated MODIS LST against ASOS (Automated Surface Observing System) meteorological stations and found night-time RMSE < 3°C and daytime RMSE > 4°C, attributable to spatial heterogeneity.

### 6.3 Sensitivity Analysis

Recommended sensitivity tests:
1. **Method comparison:** Run all rural methods, report range
2. **Threshold sensitivity:** Vary NDVI thresholds ±0.1
3. **Buffer width:** Test 5km, 10km, 15km buffers
4. **Elevation tolerance:** Compare 50m vs. 100m vs. no filter

Raj and Yun (2024) found that method choice matters most for highly urbanized cities (>50% urban land cover), where buffer and incity methods can differ by >1°C.

### 6.4 Input Data Quality Checks

Users should validate inputs before analysis:

- **Cloud Contamination:** LST rasters should be cloud-masked. The tool handles NaN values but cannot detect clouds labeled as valid cold pixels.
- **LULC Accuracy:** If using LULC method, verify the "Built" class ID matches your raster (e.g., Class 6 for Dynamic World).
- **Projection:** Ensure the input city boundary is accurate and has a defined CRS.
- **Temporal Alignment:** Confirm LULC and LST acquisition dates are reasonably synchronized.

---

## 7. Limitations

### 7.1 Methodological Limitations

1. **Single-image analysis:** Tool processes one LST image; temporal averaging requires external preprocessing or iterative runs
2. **Administrative boundary dependence:** Results tied to boundary definition
3. **Homogeneity assumption:** Assumes urban/rural categories are internally homogeneous

### 7.2 Data Limitations

1. **Cloud contamination:** Thermal imagery requires cloud-free conditions
2. **Temporal mismatch:** LULC products may not match LST acquisition date
3. **Resolution effects:** Coarser resolution smooths temperature extremes

### 7.3 Physical Limitations

1. **Surface vs. air temperature:** SUHII ≠ CUHII; interpretation differs
2. **Emissivity effects:** LST retrieval assumes emissivity; urban materials vary
3. **View angle effects:** Off-nadir observations may bias LST

### 7.4 Elevation Correction Limitations

The current version filters by elevation but does not apply a lapse rate correction factor (e.g., adding $6.5 \times \Delta h$) to the rural temperature. This is a conservative approach (discarding data rather than modifying it).

---

## 8. References

Ahmad, Bilal, Mohammad Bareeq Najar, and Shamshad Ahmad. 2024. "Analysis of LST, NDVI, and UHI Patterns for Urban Climate Using Landsat-9 Satellite Data in Delhi." *Journal of Atmospheric and Solar-Terrestrial Physics* 265 (December): 106359. https://doi.org/10.1016/j.jastp.2024.106359.

Brown, Christopher F., Steven P. Brumby, Brookie Guzder-Williams, et al. 2022. "Dynamic World, Near Real-Time Global 10 m Land Use Land Cover Mapping." *Scientific Data* 9 (1): 251. https://doi.org/10.1038/s41597-022-01307-4.

Fernandes, Rodrigo, Antonio Ferreira, Victor Nascimento, Marcos Freitas, and Jean Ometto. 2024. "Urban Heat Island Assessment in the Northeastern State Capitals in Brazil Using Sentinel-3 SLSTR Satellite Data." *Sustainability* 16 (11): 4764. https://doi.org/10.3390/su16114764.

Keerthi Naidu, Bhogadi Naga, and Faiz Ahmed Chundeli. 2023. "Assessing LULC Changes and LST through NDVI and NDBI Spatial Indicators: A Case of Bengaluru, India." *GeoJournal* 88 (4): 4335–50. https://doi.org/10.1007/s10708-023-10862-1.

Kumar, Ajay, Pushpendra Sharma, Narayan Vyas, Amit Sharma, Rohit Maheshwari, and Pushan Kumar Dutta. 2025. "Land Use Land Cover Mapping Using Random Forest and Multispectral Satellite Imagery in Google Earth Engine." *2025 Third International Conference on Networks, Multimedia and Information Technology (NMITCON)*, August 1, 1–5. https://doi.org/10.1109/NMITCON65824.2025.11188013.

Mentaschi, Lorenzo, Grégory Duveiller, Grazia Zulian, et al. 2022. "Global Long-Term Mapping of Surface Temperature Shows Intensified Intra-City Urban Heat Island Extremes." *Global Environmental Change* 72 (January): 102441. https://doi.org/10.1016/j.gloenvcha.2021.102441.

Raj, Sarath, and Geun Young Yun. 2024. "Influence of Selection of Rural Reference Area for Quantifying the Surface Urban Heat Islands Intensity in Major South Korean Cities." *Architectural Science Review* 67 (4): 345–56. https://doi.org/10.1080/00038628.2023.2290740.

Tempa, Karma, Masengo Ilunga, Abhishek Agarwal, and Tashi. 2024. "Utilizing Sentinel-2 Satellite Imagery for LULC and NDVI Change Dynamics for Gelephu, Bhutan." *Applied Sciences* 14 (4): 1578. https://doi.org/10.3390/app14041578.

Vignesh, T., Thyagharajan K. K., R. Beaulah Jeyavathana, and Prasanna Kumar R. 2022. "Land Use and Land Cover Classification Using Landsat-8 Multispectral Remote Sensing Images and Long Short-Term Memory-Recurrent Neural Network." 070001. https://doi.org/10.1063/5.0113197.

Yang, Xiaoshan, and Lingye Yao. 2022. "Reexamining the Relationship between Surface Urban Heat Island Intensity and Annual Precipitation: Effects of Reference Rural Land Cover." *Urban Climate* 41 (January): 101074. https://doi.org/10.1016/j.uclim.2021.101074.

Zhou, Decheng, Shuqing Zhao, Shuguang Liu, Liangxia Zhang, and Chao Zhu. 2014. "Surface Urban Heat Island in China's 32 Major Cities: Spatial Patterns and Drivers." *Remote Sensing of Environment* 152 (September): 51–61. https://doi.org/10.1016/j.rse.2014.05.017.

---

## Appendix A: User Guide

### A.1 System Requirements

**Python Version:** 3.8+

**Required Packages:**
```
numpy>=1.20
geopandas>=0.10
rasterio>=1.2
scipy>=1.7
matplotlib>=3.4
```

**Installation:**
```bash
pip install numpy geopandas rasterio scipy matplotlib
```

Or using a requirements file:
```bash
pip install -r requirements.txt
```

**System Requirements:**
- RAM: Minimum 8GB (16GB+ recommended for large cities)
- Disk: ~500MB free space per analysis
- OS: Linux, macOS, or Windows with Python 3.8+

### A.2 Input Data Requirements

| Input | Format | Requirements | Notes |
|-------|--------|--------------|-------|
| LST | GeoTIFF | Single band, any CRS, Float32/Float64 | Must be cloud-masked |
| LULC | GeoTIFF | Integer classes, any CRS | Required for `lulc` or `lulc_ndvi` methods |
| NDVI | GeoTIFF | Float [-1, 1], any CRS | Required for all methods |
| DEM | GeoTIFF | Float (meters), any CRS | Required if elevation correction enabled |
| Boundary | GeoJSON/Shapefile | Polygon geometry, any CRS | Must encompass urban area |

**Data Preparation Notes:**
- All rasters will be reprojected to UTM; input CRS must be defined
- LST can be in Celsius, Kelvin, or scaled Celsius (configure `lst_units`)
- Boundary should encompass entire urban area of interest
- Recommended resolution: 10-100m for urban studies

### A.3 Quick Start

1. **Prepare input data** in a directory (e.g., `./data/`)

2. **Create configuration file** (`config.json`):
```json
{
    "paths": {
        "input_dir": "./data",
        "output_dir": "./results",
        "lst_file": "LST.tif",
        "lulc_file": "LULC.tif",
        "ndvi_file": "NDVI.tif",
        "dem_file": "DEM.tif",
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
            "fixed_width_m": 10000,
            "min_distance_from_edge_m": 0,
            "ring_type": "ua",
            "min_rural_distance_m": 0
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

3. **Run analysis:**
```bash
python suhii_tool.py
```

4. **Review outputs** in `./results/`:
   - `results.json` - Quantitative results
   - `SUHIIPixelDeviation.tif` - Deviation map (GeoTIFF)
   - `SUHIIPixelDeviation.png` - Deviation map (visualization)
   - `analysis.log` - Processing log
   - `debug_plots/` - Quality control visualizations

### A.4 Generating Config Template

To create a fully-documented configuration template:

```bash
python suhii_tool.py --generate-config my_config.json
```

This will create a configuration file with all parameters and inline documentation.

---

## Appendix B: Configuration Reference

### B.1 Complete Parameter List

#### Paths Section
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_dir` | string | Yes | Directory containing input files |
| `output_dir` | string | Yes | Directory for output products |
| `lst_file` | string | Yes | LST raster filename |
| `lulc_file` | string | Conditional | LULC raster filename (required for `lulc` or `lulc_ndvi` methods) |
| `ndvi_file` | string | Yes | NDVI raster filename |
| `dem_file` | string | Conditional | DEM raster filename (required if `use_elevation_correction: true`) |
| `boundary_file` | string | Yes | Boundary vector filename (GeoJSON or Shapefile) |

#### LST Units
| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `lst_units` | string | `"celsius"` | `"celsius"`, `"kelvin"`, `"celsius_scaled"` | Input LST unit format |

#### Urban Selection
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `method` | string | - | Yes | `"lulc"`, `"ndvi"`, `"lulc_ndvi"` |
| `urban_classes` | array[int] | - | Conditional | LULC class codes for urban (required for `lulc` methods) |
| `water_class` | int | 0 | No | LULC class code for water |
| `nodata_value` | int | 255 | No | LULC nodata value |
| `ndvi_max_threshold` | float | 0.3 | Conditional | Max NDVI for urban pixels (required for `ndvi` methods) |

**Method Selection Guide:**
- `lulc`: Use when you have reliable LULC data and want simple classification
- `ndvi`: Use when LULC unavailable; **WARNING:** may include bare soil
- `lulc_ndvi`: **RECOMMENDED** - Most rigorous, excludes urban parks

#### Rural Selection
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `method` | string | - | Yes | `"buffer"`, `"halo"`, `"three_rings"`, `"incity"` |
| `fixed_width_m` | float | 10000 | Conditional | Buffer width for `buffer`/`halo` methods |
| `min_distance_from_edge_m` | float | 0 | Conditional | Inner exclusion for `halo` method |
| `ring_type` | string | `"ua"` | Conditional | Zone selection for `three_rings`: `"ua"`, `"fua"`, or `"pua"` |
| `min_rural_distance_m` | float | 0 | No | Pixel-level distance filter (all methods) |
| `vegetation_ndvi_threshold` | float | 0.2 | No | Min NDVI for rural pixels |
| `exclude_urban_lulc_classes` | bool | true | No | Exclude urban LULC from rural |

**Method Selection Guide:**
- `buffer`: Most common, easy to replicate (Raj and Yun 2024)
- `halo`: Use for large cities to avoid thermal footprint contamination
- `three_rings`: Best for multi-city comparisons (Fernandes et al. 2024)
- `incity`: Use for intra-urban analysis or isolated cities (Ahmad, Najar, and Ahmad 2024)

#### Filters
| Parameter | Type | Default | Required | Description |
|-----------|------|---------|----------|-------------|
| `mask_water` | bool | true | No | Exclude water bodies from all masks |
| `use_elevation_correction` | bool | true | No | Apply elevation filter to rural pixels |
| `initial_tolerance_m` | float | 50 | No | Starting elevation tolerance (Raj and Yun 2024: ±50m) |
| `max_tolerance_m` | float | 200 | No | Maximum elevation tolerance |
| `step_m` | float | 25 | No | Tolerance increment |
| `min_valid_pixels` | int | 100 | No | Minimum rural pixel count |

**Elevation Correction Recommendations:**
- Flat terrain: Can disable (`use_elevation_correction: false`)
- Moderate terrain: Use defaults (±50m initial)
- Mountainous: Increase `max_tolerance_m` to 500m

#### Resampling
| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `lst` | string | `"nearest"` | `"nearest"`, `"bilinear"` | LST resampling method |
| `ndvi` | string | `"bilinear"` | `"nearest"`, `"bilinear"` | NDVI resampling method |
| `dem` | string | `"bilinear"` | `"nearest"`, `"bilinear"` | DEM resampling method |

**Note:** LULC always uses `"nearest"` (categorical data).

**Recommendation:** Keep LST as `"nearest"` to preserve measured values.

#### Outputs
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `generate_debug_plots` | bool | true | Create diagnostic visualizations in `debug_plots/` subdirectory |

### B.2 Example Configurations

#### Configuration 1: Simple Buffer Method (Recommended for Most Users)
```json
{
    "paths": {
        "input_dir": "./data",
        "output_dir": "./results",
        "lst_file": "LST.tif",
        "lulc_file": "LULC.tif",
        "ndvi_file": "NDVI.tif",
        "dem_file": "DEM.tif",
        "boundary_file": "boundary.geojson"
    },
    "lst_units": "kelvin",
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

#### Configuration 2: In-City Method for Comparative Study
```json
{
    "urban_selection": {
        "method": "lulc",
        "urban_classes": [6]
    },
    "rural_selection": {
        "method": "incity",
        "vegetation_ndvi_threshold": 0.3
    },
    "filters": {
        "mask_water": true,
        "use_elevation_correction": false
    }
}
```

#### Configuration 3: Three-Rings for Multi-City Comparison
```json
{
    "urban_selection": {
        "method": "lulc_ndvi",
        "urban_classes": [6],
        "ndvi_max_threshold": 0.3
    },
    "rural_selection": {
        "method": "three_rings",
        "buffer_params": {
            "ring_type": "pua"
        },
        "vegetation_ndvi_threshold": 0.2
    }
}
```

#### Configuration 4: Mountainous Terrain (Relaxed Elevation Constraints)
```json
{
    "filters": {
        "use_elevation_correction": true,
        "elevation_params": {
            "initial_tolerance_m": 50,
            "max_tolerance_m": 500,
            "step_m": 50,
            "min_valid_pixels": 50
        }
    }
}
```

---

*End of Technical Documentation*