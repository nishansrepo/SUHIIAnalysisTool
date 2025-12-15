"""
Adaptive Surface Urban Heat Island Intensity (SUHII) Analysis Tool

A comprehensive, academically rigorous framework for SUHII calculation allowing
flexible methodology selection based on recent literature.

This tool calculates SUHII ($SUHII = LST_{urban} - LST_{rural}$) by spatially aligning
satellite imagery and defining urban/rural zones using various peer-reviewed 
methodologies.

Methodologies Implemented:
1. Urban Definition:
   - LULC-based: Uses classification rasters (e.g., Dynamic World).
   - NDVI-based thresholding: Uses vegetation indices when LULC is unavailable.
   - LULC + NDVI combined: Uses LULC with NDVI filtering to exclude parks.
   
2. Rural Reference Selection:
   - Fixed Buffer (Raj & Yun, 2024): A static ring around the city with elevation filtering.
   - Urban Halo: Excludes the immediate "urban footprint" before buffer.
   - Three-Ring Method (Fernandes et al., 2024): Calculates dynamic buffer widths 
     (Ua, FUa, PUa) based on the square root of the urban area size.
   - In-City Non-Urban (Raj & Yun, 2024): Uses vegetated spaces within the administrative boundary.

3. Correction Factors:
   - Adaptive Elevation Filtering (Raj & Yun, 2024): Uses ±50m threshold from urban
     mean elevation to ensure rural reference pixels are climatically comparable.

4. Additional Metrics:
   - Normalized UHI (Ahmad et al., 2024): For incity method, calculates 
     (LST_urban - LST_mean) / SD for cross-city comparisons.

Primary References:
- Ahmad et al. (2024): Normalized UHI formula, LST-NDVI correlations
- Raj & Yun (2024): Buffer vs incity comparison, ±50m elevation filtering
- Fernandes et al. (2024): Three-ring zone formulas (Ua/FUa/PUa)

Author: Nishan Sah, Carnegie Mellon University
Date: 2025
License: MIT
"""

import os
import json
import logging
import math
import numpy as np
import geopandas as gpd
import rasterio
import rasterio.features
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================

@dataclass
class AnalysisConfig:
    """
    Data class to validate and hold runtime configuration parameters.
    
    Parses the JSON configuration file and maps values to typed attributes
    to ensure type safety throughout the analysis pipeline.
    """
    # Paths
    input_dir: str
    output_dir: str
    lst_file: str
    lulc_file: Optional[str]
    ndvi_file: str
    dem_file: str
    boundary_file: str
    
    # LST Configuration
    lst_units: str  # 'celsius', 'kelvin', or 'celsius_scaled' (C * 100)
    
    # Urban Selection
    urban_method: str  # 'lulc', 'ndvi', or 'lulc_ndvi'
    urban_classes: List[int]
    water_class: int
    nodata_value: int
    urban_ndvi_max: float
    
    # Rural Selection
    rural_method: str  # 'buffer', 'halo', 'three_rings', 'incity'
    buffer_width: float
    halo_min_distance: float
    ring_type: str  # 'ua', 'fua', 'pua' (for three_rings method)
    rural_ndvi_min: float
    exclude_urban_lulc: bool  # If True, excludes urban LULC classes from rural mask
    min_rural_distance: float  # Minimum distance from urban edge for rural pixels (meters)
    
    # Filters
    mask_water: bool
    use_elevation: bool
    elev_init_tol: float
    elev_max_tol: float
    elev_step: float
    min_pixels: int
    
    # Resampling Methods
    resample_lst: str  # 'nearest' or 'bilinear'
    resample_ndvi: str
    resample_dem: str
    
    # Outputs
    debug_plots: bool

    @classmethod
    def from_json(cls, json_path: str):
        """
        Factory method to create a configuration object from a JSON file.
        
        Args:
            json_path (str): Path to the config.json file.
            
        Returns:
            AnalysisConfig: Populated configuration object.
        """
        with open(json_path, 'r') as f:
            cfg = json.load(f)
        
        return cls(
            input_dir=cfg['paths']['input_dir'],
            output_dir=cfg['paths']['output_dir'],
            lst_file=cfg['paths']['lst_file'],
            lulc_file=cfg['paths'].get('lulc_file'),  # Can be null
            ndvi_file=cfg['paths']['ndvi_file'],
            dem_file=cfg['paths']['dem_file'],
            boundary_file=cfg['paths']['boundary_file'],
            
            lst_units=cfg.get('lst_units', 'celsius'),  # Default to celsius for backward compatibility
            
            urban_method=cfg['urban_selection']['method'],
            urban_classes=cfg['urban_selection'].get('urban_classes', []),
            water_class=cfg['urban_selection'].get('water_class', 0),
            nodata_value=cfg['urban_selection'].get('nodata_value', 255),
            urban_ndvi_max=cfg['urban_selection']['ndvi_max_threshold'],
            
            rural_method=cfg['rural_selection']['method'],
            buffer_width=cfg['rural_selection']['buffer_params'].get('fixed_width_m', 10000),
            halo_min_distance=cfg['rural_selection']['buffer_params'].get('min_distance_from_edge_m', 0),
            ring_type=cfg['rural_selection']['buffer_params'].get('ring_type', 'ua'),
            rural_ndvi_min=cfg['rural_selection']['vegetation_ndvi_threshold'],
            exclude_urban_lulc=cfg['rural_selection'].get('exclude_urban_lulc_classes', True),
            min_rural_distance=cfg['rural_selection']['buffer_params'].get('min_rural_distance_m', 0),
            
            mask_water=cfg['filters']['mask_water'],
            use_elevation=cfg['filters']['use_elevation_correction'],
            elev_init_tol=cfg['filters']['elevation_params']['initial_tolerance_m'],
            elev_max_tol=cfg['filters']['elevation_params']['max_tolerance_m'],
            elev_step=cfg['filters']['elevation_params']['step_m'],
            min_pixels=cfg['filters']['elevation_params']['min_valid_pixels'],
            
            resample_lst=cfg.get('resampling', {}).get('lst', 'nearest'),
            resample_ndvi=cfg.get('resampling', {}).get('ndvi', 'bilinear'),
            resample_dem=cfg.get('resampling', {}).get('dem', 'bilinear'),
            
            debug_plots=cfg['outputs']['generate_debug_plots']
        )

# =============================================================================
# ANALYZER CLASS
# =============================================================================

class SUHIIAnalyzer:
    """
    Main controller for Surface Urban Heat Island Intensity (SUHII) analysis.
    
    This class orchestrates the loading of geospatial data, alignment of rasters,
    geometric operations for urban/rural masking, and statistical calculation of SUHII.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the analyzer with a configuration file.
        
        Args:
            config_path (str): Path to the JSON configuration file.
        """
        self.cfg = AnalysisConfig.from_json(config_path)
        self._setup_directories()
        self._setup_logging()
        
    def _setup_directories(self):
        """Create necessary output directories based on configuration."""
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        if self.cfg.debug_plots:
            self.debug_dir = os.path.join(self.cfg.output_dir, 'debug_plots')
            os.makedirs(self.debug_dir, exist_ok=True)

    def _setup_logging(self):
        """Configure logging to both file and console."""
        logging.basicConfig(
            filename=os.path.join(self.cfg.output_dir, 'analysis.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filemode='w'
        )
        # Add stdout handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger().addHandler(console)
        self.logger = logging.getLogger(__name__)
        self.logger.info("SUHII Analysis Initialized with user configuration.")

    def _clean_array(self, arr):
        """Replace -inf and inf with NaN. Returns a copy to avoid side effects."""
        cleaned = arr.astype(np.float32).copy()
        cleaned[~np.isfinite(cleaned)] = np.nan
        return cleaned

    def _convert_lst_units(self, lst: np.ndarray) -> np.ndarray:
        """
        Convert LST to Celsius based on configured input units.
        
        Supported input formats:
        - 'celsius': No conversion needed
        - 'kelvin': Subtract 273.15 (common for MODIS, Landsat Collection 2)
        - 'celsius_scaled': Divide by 100 (some GEE exports use C * 100)
        
        Args:
            lst (np.ndarray): Input LST array in configured units.
            
        Returns:
            np.ndarray: LST array in Celsius.
        """
        if self.cfg.lst_units == 'celsius':
            self.logger.info("LST units: Celsius (no conversion)")
            return lst
        elif self.cfg.lst_units == 'kelvin':
            self.logger.info("LST units: Kelvin -> Converting to Celsius (subtracting 273.15)")
            return lst - 273.15
        elif self.cfg.lst_units == 'celsius_scaled':
            self.logger.info("LST units: Celsius scaled (x100) -> Converting to Celsius (dividing by 100)")
            return lst / 100.0
        else:
            self.logger.warning(f"Unknown LST units '{self.cfg.lst_units}', assuming Celsius")
            return lst

    # -------------------------------------------------------------------------
    # GEOSPATIAL UTILS
    # -------------------------------------------------------------------------

    def _estimate_utm_crs(self, lat: float, lon: float) -> str:
        """
        Estimate the appropriate Universal Transverse Mercator (UTM) CRS.
        
        UTM divides the Earth into 60 zones of 6 degrees longitude. Calculating
        buffers in a geographic CRS (degrees) creates distortion; this ensures
        calculations happen in a metric projected coordinate system.
        
        Args:
            lat (float): Latitude of the centroid.
            lon (float): Longitude of the centroid.
            
        Returns:
            str: EPSG code for the UTM zone (e.g., 'EPSG:32618').
        """
        zone = math.floor((lon + 180) / 6) + 1
        epsg = 32600 + zone if lat >= 0 else 32700 + zone
        return f"EPSG:{int(epsg)}"

    def _align_raster(self, master_path: str, slave_path: str, out_path: str, 
                      resample_method: str = 'bilinear') -> str:
        """
        Pixel-perfectly align a slave raster to a master raster's grid.
        
        Essential for pixel-wise arithmetic (e.g., LST - Rural_Mean). Ensures that
        pixel [0,0] in the LST raster corresponds exactly to pixel [0,0] in the 
        LULC, NDVI, and DEM rasters.
        
        Resampling methods:
        - 'nearest': Preserves original values, best for categorical data (LULC) and
          measured values (LST) where artificial interpolation is undesirable.
        - 'bilinear': Weighted average of 4 nearest pixels, creates smoother results
          for continuous data (NDVI, DEM) but introduces artificial values.
        
        Scientific note: For LST, nearest neighbor is recommended to preserve actual
        radiometric measurements. Bilinear interpolation at urban-rural boundaries
        can create artificial temperature values that were never measured.
        
        Args:
            master_path (str): Path to the reference raster (usually LST).
            slave_path (str): Path to the raster to be reprojected/resampled.
            out_path (str): Output path for the aligned raster.
            resample_method (str): 'nearest' or 'bilinear'. Default 'bilinear'.
                                   
        Returns:
            str: Path to the generated aligned raster.
        """
        # Map string to rasterio Resampling enum
        if resample_method == 'nearest':
            resampling = Resampling.nearest
            dtype = 'float32'  # Default for continuous
            nodata = np.nan
        elif resample_method == 'bilinear':
            resampling = Resampling.bilinear
            dtype = 'float32'
            nodata = np.nan
        elif resample_method == 'nearest_categorical':
            # Special case for categorical data (LULC)
            resampling = Resampling.nearest
            dtype = 'uint8'
            nodata = self.cfg.nodata_value
        else:
            self.logger.warning(f"Unknown resample method '{resample_method}', defaulting to bilinear")
            resampling = Resampling.bilinear
            dtype = 'float32'
            nodata = np.nan
        
        with rasterio.open(master_path) as master:
            kwargs = master.meta.copy()
            dst_crs = master.crs
            dst_transform = master.transform
            
        with rasterio.open(slave_path) as slave:
            kwargs.update(dtype=dtype, nodata=nodata)
                
            with rasterio.open(out_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(slave, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=slave.transform,
                    src_crs=slave.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )
        
        self.logger.info(f"  Aligned {os.path.basename(slave_path)} using {resample_method} resampling")
        return out_path

    def _create_distance_raster(self, urban_mask: np.ndarray, transform) -> np.ndarray:
        """
        Create a raster of distances from urban boundary edge.
        
        Uses Euclidean Distance Transform (EDT) algorithm for efficient computation
        of distances across the entire raster. Essential for hybrid rural selection
        that excludes pixels too close to the urban heat island footprint.
        
        Args:
            urban_mask (np.ndarray): Boolean mask where True = urban pixels.
            transform: Rasterio affine transform for pixel size calculation.
            
        Returns:
            np.ndarray: Distance raster in meters. 0 = inside urban, >0 = outside.
        """
        # Calculate Euclidean distance transform (in pixels)
        # For each non-urban pixel, computes distance to nearest urban pixel
        distance_pixels = distance_transform_edt(~urban_mask)
        
        # Convert pixel distances to meters
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        pixel_size = np.mean([pixel_size_x, pixel_size_y])
        
        distance_meters = distance_pixels * pixel_size
        
        self.logger.info(f"Distance raster created: pixel size = {pixel_size:.2f}m, "
                        f"max distance = {np.max(distance_meters):.0f}m")
        
        return distance_meters

    # -------------------------------------------------------------------------
    # SELECTION METHODOLOGIES
    # -------------------------------------------------------------------------

    def _generate_lulc_visualization(self, lulc_arr: np.ndarray, filename: str, title: str):
        """
        Generate a standardized LULC map highlighting Urban and Water pixels.
        
        This debug plot allows users to visually verify that their configured 
        `urban_classes` and `water_class` match the actual data in their LULC raster.
        
        Args:
            lulc_arr (np.ndarray): The LULC data array.
            filename (str): Output filename.
            title (str): Title for the plot.
        """
        if not self.cfg.debug_plots or lulc_arr is None: return
        
        # Create a display array
        disp = np.zeros_like(lulc_arr, dtype=int)  # 0 = Other
        
        # Flag Water
        if self.cfg.mask_water:
            disp[lulc_arr == self.cfg.water_class] = 1  # Blue
            
        # Flag Urban
        mask_urban = np.isin(lulc_arr, self.cfg.urban_classes)
        disp[mask_urban] = 2  # Red
        
        # Mask nodata
        disp = np.ma.masked_where(lulc_arr == self.cfg.nodata_value, disp)
        
        cmap = mcolors.ListedColormap(['#e0e0e0', '#419bdf', '#c4281b'])  # Gray, Blue, Red
        bounds = [-0.5, 0.5, 1.5, 2.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(disp, cmap=cmap, norm=norm)
        legend_elements = [
            Patch(facecolor='#c4281b', label='Urban Target'),
            Patch(facecolor='#419bdf', label='Water'),
            Patch(facecolor='#e0e0e0', label='Other')
        ]
        plt.legend(handles=legend_elements)
        plt.title(title)
        plt.axis('off')
        plt.savefig(os.path.join(self.debug_dir, filename), dpi=150, bbox_inches='tight')
        plt.close()

    def _define_urban_mask(self, lulc: Optional[np.ndarray], ndvi: np.ndarray) -> np.ndarray:
        """
        Define the urban core pixels based on the selected methodology.
        
        Methodologies:
        1. 'lulc': Uses classification codes only (e.g., Dynamic World class 6).
           Pure LULC-based selection.
        2. 'ndvi': Uses an inverted NDVI threshold only.
           Warning: Can inadvertently select bare soil as urban.
        3. 'lulc_ndvi': Combined approach using LULC classification
           with NDVI filtering to exclude vegetated pixels (parks, urban green spaces)
           within built-up areas. Most rigorous for isolating impervious surfaces.
           
        Args:
            lulc (np.ndarray): Land Use Land Cover array. Required for 'lulc' and 'lulc_ndvi'.
            ndvi (np.ndarray): Normalized Difference Vegetation Index array.
            
        Returns:
            np.ndarray: Boolean mask where True indicates an Urban Core pixel.
        """
        self.logger.info(f"Defining Urban Mask using method: {self.cfg.urban_method}")
        
        if self.cfg.urban_method == 'lulc':
            # Pure LULC-based
            if lulc is None:
                raise ValueError("Method 'lulc' selected but no LULC file provided.")
            
            urban_mask = np.isin(lulc, self.cfg.urban_classes)
            self.logger.info(f"  LULC-only: Selected classes {self.cfg.urban_classes}")
                
        elif self.cfg.urban_method == 'ndvi':
            # Pure NDVI-based (Ahmad et al., 2024 simplified)
            # Warning: This risks including bare soil
            urban_mask = (ndvi < self.cfg.urban_ndvi_max) & np.isfinite(ndvi)
            self.logger.warning(f"  NDVI-only (threshold < {self.cfg.urban_ndvi_max}): "
                              "Warning - may include bare soil as urban.")
            
        elif self.cfg.urban_method == 'lulc_ndvi':
            # Combined LULC + NDVI
            # Most rigorous: LULC defines built-up, NDVI excludes parks/green spaces
            if lulc is None:
                raise ValueError("Method 'lulc_ndvi' selected but no LULC file provided.")
            
            # Base: Is in urban LULC classes
            lulc_urban = np.isin(lulc, self.cfg.urban_classes)
            
            # Filter: Remove high-NDVI pixels (parks, urban vegetation)
            low_ndvi = (ndvi < self.cfg.urban_ndvi_max) & np.isfinite(ndvi)
            
            urban_mask = lulc_urban & low_ndvi
            self.logger.info(f"  LULC + NDVI combined: Classes {self.cfg.urban_classes} "
                           f"AND NDVI < {self.cfg.urban_ndvi_max}")
            
        else:
            raise ValueError(f"Unknown urban method: {self.cfg.urban_method}. "
                           "Valid options: 'lulc', 'ndvi', 'lulc_ndvi'")

        # Always mask water if requested and LULC available
        if self.cfg.mask_water and lulc is not None:
            is_water = (lulc == self.cfg.water_class)
            urban_mask = urban_mask & (~is_water)
            
        return urban_mask

    def _define_rural_geometry(self, urban_geom, utm_crs):
        r"""
        Calculate the geometry for the rural reference zone based on the config.
        
        Methodologies:
        1. 'incity' (Raj & Yun, 2024): The rural reference is defined as vegetated
           pixels *inside* the administrative boundary. Raj & Yun call this 
           "Method 1: Non-urban areas within city limits".
        2. 'buffer' (Raj & Yun, 2024): A fixed width buffer around the city boundary
           with optional elevation and land cover filtering.
        3. 'halo': Skips an immediate 'Urban Footprint' zone before starting the 
           buffer to avoid heat contamination from UHI advection effects.
        4. 'three_rings' (Fernandes et al., 2024): Uses dynamic buffer widths scaled
           by the square root of the city area ($A$) to create comparable zones 
           (Ua, FUa, PUa) for cities of different sizes.
           
           Formulas from Fernandes et al. (2024):
           - $W_{Ua} = 0.25\sqrt{A}$ (Urban Adjacent)
           - $W_{FUa} = 0.25\sqrt{A_{Wa}}$ where $A_{Wa} = A + A_{Ua}$ (Future Urban Adjacent)
           - $W_{PUa} = 1.5\sqrt{A} - W_{FUa} - W_{Ua}$ (Peri-Urban)
           
        Args:
            urban_geom (shapely.geometry): The urban boundary.
            utm_crs (str): The projected CRS used for calculation (for logging).
            
        Returns:
            shapely.geometry: The geometry representing the rural search area.
        """
        method = self.cfg.rural_method
        self.logger.info(f"Calculating Rural Reference Geometry using method: {method}")
        
        urban_area_km2 = urban_geom.area / 1e6
        
        if method == 'incity':
            # Raj & Yun (2024) "Method 1": Non-urban pixels INSIDE the boundary
            self.logger.info("  Method: In-City Analysis (Reference is non-built space inside boundary)")
            return urban_geom
            
        elif method == 'buffer':
            # Raj & Yun (2024): Fixed buffer around city
            buff_dist = self.cfg.buffer_width
            self.logger.info(f"  Method: Fixed Buffer ({buff_dist}m)")
            return urban_geom.buffer(buff_dist).difference(urban_geom)
            
        elif method == 'halo':
            # Skip 'urban footprint' zone, then buffer
            min_dist = self.cfg.halo_min_distance
            max_dist = min_dist + self.cfg.buffer_width
            self.logger.info(f"  Method: Urban Halo (Skip {min_dist}m, Width {self.cfg.buffer_width}m)")
            
            inner_ring = urban_geom.buffer(min_dist)
            outer_ring = urban_geom.buffer(max_dist)
            return outer_ring.difference(inner_ring)
            
        elif method == 'three_rings':
            # Fernandes et al. (2024) three-tier classification
            # Formulas produce widths in KM when area is in KM². Convert to meters.
            
            # Width of Urban Adjacent (Ua) zone
            w_ua_km = 0.25 * math.sqrt(urban_area_km2)
            w_ua_m = w_ua_km * 1000
            
            # Geometry of Ua (Urban Adjacent) - ring around urban boundary
            geom_ua = urban_geom.buffer(w_ua_m).difference(urban_geom)
            
            # Cumulative area for FUa calculation: A_Wa = urban_area + Ua_ring_area
            # (Fernandes et al. use cumulative enclosed area)
            area_ua_ring_km2 = geom_ua.area / 1e6
            cumulative_area_wa = urban_area_km2 + area_ua_ring_km2
            
            # Width of Future Urban Adjacent (FUa) zone
            w_fua_km = 0.25 * math.sqrt(cumulative_area_wa)
            w_fua_m = w_fua_km * 1000
            
            # Geometry of FUa - buffer from outer edge of Ua
            geom_ua_union = urban_geom.buffer(w_ua_m)
            geom_fua = geom_ua_union.buffer(w_fua_m).difference(geom_ua_union)
            
            # Width of Peri-Urban (PUa) zone
            w_pua_km = (1.5 * math.sqrt(urban_area_km2)) - w_fua_km - w_ua_km
            w_pua_m = w_pua_km * 1000
            
            # Geometry of PUa - buffer from outer edge of FUa
            geom_fua_union = geom_ua_union.buffer(w_fua_m)
            geom_pua = geom_fua_union.buffer(w_pua_m).difference(geom_fua_union)
            
            self.logger.info(f"  Method: Three Rings. Selected: {self.cfg.ring_type.upper()}")
            self.logger.info(f"  Calculated Widths -> Ua: {w_ua_m:.0f}m, FUa: {w_fua_m:.0f}m, PUa: {w_pua_m:.0f}m")
            
            if self.cfg.ring_type == 'ua': return geom_ua
            elif self.cfg.ring_type == 'fua': return geom_fua
            elif self.cfg.ring_type == 'pua': return geom_pua
            else: raise ValueError("Invalid ring_type. Choose 'ua', 'fua', or 'pua'")
            
        else:
            raise ValueError(f"Unknown rural method: {method}")

    def _save_final_deviation_maps(self, deviation_urban_full: np.ndarray, 
                                     deviation_pixel: np.ndarray,
                                     deviation_all: np.ndarray,
                                     profile: dict, suhii_val: float):
        """
        Save the SUHII Pixel Deviation maps in both GeoTIFF and PNG formats.
        
        Three outputs are generated, all with deviations calculated from the 
        mean of the final filtered rural mask:
        
        1. SUHIIUrbanFullDeviation: Full urban boundary + final rural mask
        2. SUHIIPixelDeviation: Urban core (filtered) + final rural mask  
        3. SUHIIAll: Entire buffer + full urban boundary
        
        Args:
            deviation_urban_full (np.ndarray): Full urban + final rural mask deviations.
            deviation_pixel (np.ndarray): Filtered urban + final rural mask deviations.
            deviation_all (np.ndarray): Full analysis universe deviations.
            profile (dict): Rasterio profile for the GeoTIFF.
            suhii_val (float): The calculated SUHII scalar value (for the plot title).
        """
        prof = profile.copy()
        prof.update(dtype='float32', nodata=np.nan, compress='deflate')
        
        # Clean deviation maps before saving
        clean_urban_full = self._clean_array(deviation_urban_full)
        clean_pixel = self._clean_array(deviation_pixel)
        clean_all = self._clean_array(deviation_all)
        
        # 1. SUHIIUrbanFullDeviation: Full urban geometry + final rural mask
        out_tif_1 = os.path.join(self.cfg.output_dir, "SUHIIUrbanFullDeviation.tif")
        with rasterio.open(out_tif_1, 'w', **prof) as dst:
            dst.write(clean_urban_full.astype('float32'), 1)
        self.logger.info(f"Saved: {out_tif_1}")
        
        out_png_1 = os.path.join(self.cfg.output_dir, "SUHIIUrbanFullDeviation.png")
        self._save_deviation_png(clean_urban_full, out_png_1, suhii_val,
                                 "SUHII Urban Full Deviation\n(Full Urban + Filtered Rural Mask)")
        
        # 2. SUHIIPixelDeviation: Urban core (filtered) + final rural mask
        out_tif_2 = os.path.join(self.cfg.output_dir, "SUHIIPixelDeviation.tif")
        with rasterio.open(out_tif_2, 'w', **prof) as dst:
            dst.write(clean_pixel.astype('float32'), 1)
        self.logger.info(f"Saved: {out_tif_2}")
        
        out_png_2 = os.path.join(self.cfg.output_dir, "SUHIIPixelDeviation.png")
        self._save_deviation_png(clean_pixel, out_png_2, suhii_val,
                                 "SUHII Pixel Deviation\n(Filtered Urban + Filtered Rural Mask)")
        
        # 3. SUHIIAll: Entire buffer + full urban boundary
        out_tif_3 = os.path.join(self.cfg.output_dir, "SUHIIAll.tif")
        with rasterio.open(out_tif_3, 'w', **prof) as dst:
            dst.write(clean_all.astype('float32'), 1)
        self.logger.info(f"Saved: {out_tif_3}")
        
        out_png_3 = os.path.join(self.cfg.output_dir, "SUHIIAll.png")
        self._save_deviation_png(clean_all, out_png_3, suhii_val,
                                 "SUHII All\n(Full Urban + Entire Buffer)")
    
    def _save_deviation_png(self, deviation_data: np.ndarray, out_path: str, 
                            suhii_val: float, title_prefix: str):
        """
        Helper to save a deviation map as PNG with consistent styling.
        
        Args:
            deviation_data (np.ndarray): Cleaned deviation data array.
            out_path (str): Output file path.
            suhii_val (float): SUHII value for title.
            title_prefix (str): Title prefix for the plot.
        """
        plt.figure(figsize=(12, 10))
        
        # Determine color scale
        valid_mask = np.isfinite(deviation_data)
        valid_data = deviation_data[valid_mask]
        
        if valid_data.size > 0:
            vmax = np.percentile(np.abs(valid_data), 98)
            vmin = -vmax
        else:
            vmin, vmax = -5, 5

        plt.imshow(deviation_data, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        cbar = plt.colorbar(shrink=0.8)
        cbar.set_label("Deviation from Rural Mean (°C)")
        
        title_val = f"{suhii_val:.2f}" if np.isfinite(suhii_val) else "NaN"
        plt.title(f"{title_prefix}\nGlobal SUHII: {title_val} °C")
        plt.axis('off')
        
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Saved Deviation Image: {out_path}")

    def _calculate_normalized_uhi(self, lst: np.ndarray, urban_mask: np.ndarray, 
                                   study_area_mask: np.ndarray) -> Dict:
        """
        Calculate Normalized Urban Heat Island Intensity (Ahmad et al., 2024).
        
        This method expresses UHI as the number of standard deviations the urban
        mean deviates from the study area mean, enabling scale-independent 
        comparisons across cities with different baseline temperatures.
        
        Formula: UHI_normalized = (LST_urban - LST_mean) / σ
        
        Where:
        - LST_urban = Mean LST of urban pixels
        - LST_mean = Mean LST of entire study area (urban boundary for incity method)
        - σ = Standard deviation of LST across study area
        
        Args:
            lst (np.ndarray): Land Surface Temperature array.
            urban_mask (np.ndarray): Boolean mask of urban pixels.
            study_area_mask (np.ndarray): Boolean mask of entire study area.
            
        Returns:
            Dict: Dictionary containing normalized UHI metrics.
        """
        # Extract study area pixels
        study_pixels = lst[study_area_mask]
        study_valid = study_pixels[np.isfinite(study_pixels)]
        
        # Extract urban pixels
        urban_pixels = lst[urban_mask]
        urban_valid = urban_pixels[np.isfinite(urban_pixels)]
        
        if len(study_valid) < 2 or len(urban_valid) == 0:
            return {
                "uhi_normalized": None,
                "study_area_mean": None,
                "study_area_std": None
            }
        
        # Calculate study area statistics
        study_mean = float(np.mean(study_valid))
        study_std = float(np.std(study_valid))
        
        # Calculate urban mean
        urban_mean = float(np.mean(urban_valid))
        
        # Normalized UHI (Ahmad et al., 2024)
        if study_std > 0:
            uhi_normalized = (urban_mean - study_mean) / study_std
        else:
            uhi_normalized = None
            
        self.logger.info(f"Normalized UHI (Ahmad et al., 2024): {uhi_normalized:.4f} σ" 
                        if uhi_normalized is not None else "Normalized UHI: N/A")
        
        return {
            "uhi_normalized": float(uhi_normalized) if uhi_normalized is not None else None,
            "study_area_mean": study_mean,
            "study_area_std": study_std
        }

    def _calculate_uncertainty_metrics(self, urban_pixels: np.ndarray, rural_pixels: np.ndarray) -> Dict:
        """
        Calculate uncertainty metrics for SUHII estimation.
        
        Computes standard deviation, standard error, and valid pixel counts
        for both urban and rural samples. Standard error of SUHII difference
        is calculated assuming independent samples.
        
        Args:
            urban_pixels (np.ndarray): LST values for urban pixels.
            rural_pixels (np.ndarray): LST values for rural pixels.
            
        Returns:
            Dict: Dictionary containing uncertainty metrics.
        """
        # Filter to finite values only
        urban_valid = urban_pixels[np.isfinite(urban_pixels)]
        rural_valid = rural_pixels[np.isfinite(rural_pixels)]
        
        n_urban = len(urban_valid)
        n_rural = len(rural_valid)
        
        # Calculate standard deviations
        urban_std = float(np.std(urban_valid)) if n_urban > 1 else np.nan
        rural_std = float(np.std(rural_valid)) if n_rural > 1 else np.nan
        
        # Calculate standard errors
        urban_se = urban_std / np.sqrt(n_urban) if n_urban > 0 else np.nan
        rural_se = rural_std / np.sqrt(n_rural) if n_rural > 0 else np.nan
        
        # Standard error of the difference (SUHII)
        # SE_diff = sqrt(SE_urban^2 + SE_rural^2) assuming independence
        if np.isfinite(urban_se) and np.isfinite(rural_se):
            suhii_se = float(np.sqrt(urban_se**2 + rural_se**2))
        else:
            suhii_se = np.nan
        
        return {
            "urban_std": urban_std,
            "urban_se": float(urban_se) if np.isfinite(urban_se) else None,
            "urban_valid_pixels": n_urban,
            "rural_std": rural_std,
            "rural_se": float(rural_se) if np.isfinite(rural_se) else None,
            "rural_valid_pixels": n_rural,
            "suhii_se": suhii_se if np.isfinite(suhii_se) else None
        }

    # -------------------------------------------------------------------------
    # MAIN WORKFLOW
    # -------------------------------------------------------------------------

    def run_analysis(self):
        """
        Execute the full SUHII analysis pipeline.
        
        Steps:
        1. Process inputs and define UTM CRS.
        2. Align all rasters (LST, NDVI, DEM, LULC) to the same grid.
        3. Convert LST units to Celsius.
        4. Define Urban Mask based on LULC/NDVI/combined method.
        5. Define Rural Mask based on Geometry + Vegetation + Distance + Elevation.
        6. Calculate SUHII and uncertainty metrics.
        7. Generate Deviation Maps.
        8. Export results.
        """
        # 1. Prepare Paths
        lst_in = os.path.join(self.cfg.input_dir, self.cfg.lst_file)
        ndvi_in = os.path.join(self.cfg.input_dir, self.cfg.ndvi_file)
        dem_in = os.path.join(self.cfg.input_dir, self.cfg.dem_file)
        bound_in = os.path.join(self.cfg.input_dir, self.cfg.boundary_file)
        
        # 2. Process Boundary
        gdf = gpd.read_file(bound_in)
        centroid = gdf.geometry.iloc[0].centroid
        utm_crs = self._estimate_utm_crs(centroid.y, centroid.x)
        gdf_utm = gdf.to_crs(utm_crs)
        urban_geom = gdf_utm.geometry.iloc[0]
        
        # 3. Align Rasters
        self.logger.info("Aligning rasters...")
        lst_utm = os.path.join(self.cfg.output_dir, "aligned_lst.tif")
        
        # Map LST resampling method to rasterio enum
        if self.cfg.resample_lst == 'nearest':
            lst_resampling = Resampling.nearest
        else:
            lst_resampling = Resampling.bilinear
        
        # Reproject master (LST)
        with rasterio.open(lst_in) as src:
            transform, width, height = calculate_default_transform(
                src.crs, utm_crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({'crs': utm_crs, 'transform': transform, 'width': width, 'height': height})
            with rasterio.open(lst_utm, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=utm_crs,
                    resampling=lst_resampling)
        self.logger.info(f"  Aligned LST using {self.cfg.resample_lst} resampling")

        # Align others to LST grid with configured resampling methods
        self.logger.info(f"Resampling methods - LST: {self.cfg.resample_lst}, NDVI: {self.cfg.resample_ndvi}, DEM: {self.cfg.resample_dem}")
        ndvi_utm = self._align_raster(lst_utm, ndvi_in, os.path.join(self.cfg.output_dir, "aligned_ndvi.tif"),
                                      resample_method=self.cfg.resample_ndvi)
        dem_utm = self._align_raster(lst_utm, dem_in, os.path.join(self.cfg.output_dir, "aligned_dem.tif"),
                                     resample_method=self.cfg.resample_dem)
        
        lulc_utm = None
        if self.cfg.lulc_file:
            lulc_in = os.path.join(self.cfg.input_dir, self.cfg.lulc_file)
            lulc_utm = self._align_raster(lst_utm, lulc_in, 
                                         os.path.join(self.cfg.output_dir, "aligned_lulc.tif"), 
                                         resample_method='nearest_categorical')

        # 4. Load Data Arrays and Clean Them
        with rasterio.open(lst_utm) as src: 
            lst_raw = self._clean_array(src.read(1))
            profile = src.profile
            transform = src.transform
            shape = src.shape

        # Convert LST units to Celsius
        lst = self._convert_lst_units(lst_raw)

        with rasterio.open(ndvi_utm) as src: ndvi = self._clean_array(src.read(1))
        with rasterio.open(dem_utm) as src: dem = self._clean_array(src.read(1))

        lulc = None
        if lulc_utm:
            with rasterio.open(lulc_utm) as src: lulc = src.read(1)
            self._generate_lulc_visualization(lulc, "lulc_check.png", "LULC Classification Check")

        # 5. Define Urban Mask
        # Create geometry mask first
        geom_mask = rasterio.features.geometry_mask([urban_geom], transform=transform, invert=True, out_shape=shape)
        
        # Apply classification logic
        pixel_urban_mask = self._define_urban_mask(lulc, ndvi)
        
        # Combine: Must be inside geometry AND match class criteria
        final_urban_mask = geom_mask & pixel_urban_mask
        
        # Extract Urban Stats
        urban_pixels = lst[final_urban_mask]
        # Use nanmean to handle potential NaNs (from cleaning)
        urban_mean = np.nanmean(urban_pixels) if urban_pixels.size > 0 else np.nan
        
        # Elevation reference: Mean of urban-classified pixels (Raj & Yun, 2024 methodology)
        # This ensures elevation filtering is based on the actual built environment
        urban_elev = np.nanmean(dem[final_urban_mask]) if final_urban_mask.any() else np.nan
        
        self.logger.info(f"Urban Mean LST: {urban_mean:.2f} °C (Elevation: {urban_elev:.0f}m)")
        
        if self.cfg.debug_plots:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.where(final_urban_mask, lst, np.nan), cmap='RdYlBu_r')
            plt.colorbar(label='LST (°C)')
            plt.title("Selected Urban Pixels")
            plt.savefig(os.path.join(self.debug_dir, "mask_urban_final.png"))
            plt.close()

        # 6. Define Rural Mask
        rural_geom = self._define_rural_geometry(urban_geom, utm_crs)
        
        # Get rural geometry mask
        rural_geom_mask = rasterio.features.geometry_mask([rural_geom], transform=transform, invert=True, out_shape=shape)
        
        # Debug plot: Buffer/Rural geometry created
        if self.cfg.debug_plots:
            plt.figure(figsize=(10, 10))
            buffer_viz = np.zeros(shape, dtype=np.float32)
            buffer_viz[geom_mask] = 2  # Urban = 2
            buffer_viz[rural_geom_mask] = 1  # Buffer = 1
            buffer_viz[buffer_viz == 0] = np.nan  # Outside = NaN
            
            cmap = mcolors.ListedColormap(['#88b053', '#c4281b'])  # Green for buffer, Red for urban
            plt.imshow(buffer_viz, cmap=cmap, vmin=1, vmax=2)
            legend_elements = [
                Patch(facecolor='#c4281b', label='Urban Boundary'),
                Patch(facecolor='#88b053', label=f'Rural Buffer ({self.cfg.rural_method})')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title(f"Buffer Geometry Created\nMethod: {self.cfg.rural_method}")
            plt.axis('off')
            plt.savefig(os.path.join(self.debug_dir, "buffer_geometry.png"), dpi=150, bbox_inches='tight')
            plt.close()
        
        # Filter 1: Vegetation Threshold (Strict NDVI)
        # Check finite first to handle NaNs in NDVI
        rural_pixel_mask = np.isfinite(ndvi) & (ndvi >= self.cfg.rural_ndvi_min)
        
        # Filter 2: Mask Water (if LULC available)
        if self.cfg.mask_water and lulc is not None:
            rural_pixel_mask = rural_pixel_mask & (lulc != self.cfg.water_class)
        # Or if no LULC, assume negative NDVI is water
        elif self.cfg.mask_water and lulc is None:
            rural_pixel_mask = rural_pixel_mask & (ndvi > 0)

        # Filter 3: Exclude Urban LULC Classes and Nodata from Rural Mask
        # For 'incity' method: Always required (rural must be non-built within boundary)
        # For other methods: Optional but recommended when LULC is available
        if lulc is not None:
            # Always exclude nodata pixels - they have no valid land cover info
            is_valid_lulc = (lulc != self.cfg.nodata_value)
            rural_pixel_mask = rural_pixel_mask & is_valid_lulc
            self.logger.info(f"  Excluded LULC nodata pixels (value: {self.cfg.nodata_value})")
            
            if self.cfg.rural_method == 'incity' or self.cfg.exclude_urban_lulc:
                rural_pixel_mask = rural_pixel_mask & (~np.isin(lulc, self.cfg.urban_classes))
                self.logger.info(f"  Applied LULC urban class exclusion (classes: {self.cfg.urban_classes})")
        
        # Filter 4: Distance from Urban Edge (for buffer-based methods)
        # This ensures rural pixels are beyond the UHI "footprint" zone
        # Note: 'halo' method already handles this via geometry (inner ring exclusion)
        if self.cfg.min_rural_distance > 0 and self.cfg.rural_method in ['buffer', 'three_rings']:
            distance_raster = self._create_distance_raster(geom_mask, transform)
            is_far_enough = distance_raster >= self.cfg.min_rural_distance
            rural_pixel_mask = rural_pixel_mask & is_far_enough
            self.logger.info(f"  Applied distance filter: >= {self.cfg.min_rural_distance}m from urban edge")
            
            # Debug plot: distance filter
            if self.cfg.debug_plots:
                dist_viz = np.where(rural_geom_mask, distance_raster, np.nan)
                plt.figure(figsize=(10, 10))
                plt.imshow(dist_viz, cmap='viridis')
                plt.colorbar(label='Distance from Urban (m)')
                plt.title(f"Distance from Urban Edge\nThreshold: {self.cfg.min_rural_distance}m")
                plt.axis('off')
                plt.savefig(os.path.join(self.debug_dir, "distance_from_urban.png"), dpi=150, bbox_inches='tight')
                plt.close()
        
        # Combine Geometry + Pixel criteria
        base_rural_mask = rural_geom_mask & rural_pixel_mask
        
        # Filter 5: Adaptive Elevation Filtering (Raj & Yun, 2024)
        # Iteratively expands elevation tolerance until minimum pixel count is reached
        final_rural_mask = base_rural_mask
        elev_tol = 0
        
        if self.cfg.use_elevation and np.isfinite(urban_elev):
            self.logger.info("Applying Adaptive Elevation Filter (Raj & Yun, 2024 methodology)...")
            current_tol = self.cfg.elev_init_tol
            
            while current_tol <= self.cfg.elev_max_tol:
                elev_diff = np.abs(dem - urban_elev)
                # Ensure we only check valid elevation pixels
                temp_mask = base_rural_mask & (elev_diff <= current_tol) & np.isfinite(elev_diff)
                count = np.count_nonzero(temp_mask)
                
                self.logger.info(f"  Tolerance ±{current_tol}m: {count} pixels")
                
                if count >= self.cfg.min_pixels:
                    final_rural_mask = temp_mask
                    elev_tol = current_tol
                    break
                current_tol += self.cfg.elev_step
            else:
                # Loop completed without finding sufficient pixels
                elev_tol = self.cfg.elev_max_tol
                final_rural_mask = base_rural_mask & (np.abs(dem - urban_elev) <= elev_tol) & np.isfinite(dem)
                final_count = np.count_nonzero(final_rural_mask)
                self.logger.warning(f"  Max elevation tolerance ({elev_tol}m) reached with {final_count} pixels "
                                  f"(min required: {self.cfg.min_pixels})")

        # Extract Rural Stats
        rural_pixels = lst[final_rural_mask]
        # Check for valid pixels before mean
        if rural_pixels.size > 0 and np.any(np.isfinite(rural_pixels)):
            rural_mean = np.nanmean(rural_pixels)
        else:
            rural_mean = np.nan
        
        rural_elev = np.nanmean(dem[final_rural_mask]) if final_rural_mask.any() else np.nan
        self.logger.info(f"Rural Mean LST: {rural_mean:.2f} °C (Elevation: {rural_elev:.0f}m, Tolerance: ±{elev_tol}m)")

        if self.cfg.debug_plots:
            plt.figure(figsize=(10, 10))
            plt.imshow(np.where(final_rural_mask, lst, np.nan), cmap='RdYlBu_r')
            plt.colorbar(label='LST (°C)')
            plt.title(f"Selected Rural Pixels ({self.cfg.rural_method})")
            plt.savefig(os.path.join(self.debug_dir, "mask_rural_final.png"))
            plt.close()
            
            # Debug plot: Final Analysis Area (Urban + Final Rural Mask)
            plt.figure(figsize=(12, 10))
            analysis_area = np.full(shape, np.nan, dtype=np.float32)
            analysis_area[final_rural_mask] = 1  # Rural = 1
            analysis_area[final_urban_mask] = 2  # Urban = 2
            
            cmap = mcolors.ListedColormap(['#88b053', '#c4281b'])  # Green for rural, Red for urban
            plt.imshow(analysis_area, cmap=cmap, vmin=1, vmax=2)
            legend_elements = [
                Patch(facecolor='#c4281b', label=f'Urban Core ({np.count_nonzero(final_urban_mask):,} pixels)'),
                Patch(facecolor='#88b053', label=f'Filtered Rural ({np.count_nonzero(final_rural_mask):,} pixels)')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title(f"Final Analysis Area\nUrban + Filtered Rural Mask")
            plt.axis('off')
            plt.savefig(os.path.join(self.debug_dir, "final_analysis_area.png"), dpi=150, bbox_inches='tight')
            plt.close()

        # 7. Final SUHII
        suhii = urban_mean - rural_mean
        if np.isfinite(suhii):
            self.logger.info(f"FINAL SUHII: {suhii:.4f} °C")
        else:
            self.logger.warning("FINAL SUHII: NaN (Insufficient data)")
        
        # Calculate uncertainty metrics
        uncertainty = self._calculate_uncertainty_metrics(urban_pixels, rural_pixels)
        
        if uncertainty['suhii_se'] is not None:
            self.logger.info(f"SUHII Standard Error: ±{uncertainty['suhii_se']:.4f} °C")
        
        # Calculate Normalized UHI for incity method (Ahmad et al., 2024)
        normalized_uhi = None
        if self.cfg.rural_method == 'incity':
            self.logger.info("Calculating Normalized UHI (Ahmad et al., 2024 methodology)...")
            normalized_uhi = self._calculate_normalized_uhi(lst, final_urban_mask, geom_mask)
        
        # 8. Deviation Maps
        # All deviations are calculated from the mean of the final filtered rural mask
        # Create 3 outputs with different spatial extents
        
        if np.isfinite(rural_mean):
            # 1. SUHIIUrbanFullDeviation: Full urban geometry + final rural mask
            #    (urban area without NDVI filtering + filtered rural reference)
            urban_full_mask = geom_mask | final_rural_mask
            deviation_urban_full = np.where(urban_full_mask, lst - rural_mean, np.nan)
            
            # 2. SUHIIPixelDeviation: Urban core (filtered) + final rural mask
            #    (only the pixels actually used in SUHII calculation)
            pixel_analysis_mask = final_urban_mask | final_rural_mask
            deviation_pixel = np.where(pixel_analysis_mask, lst - rural_mean, np.nan)
            
            # 3. SUHIIAll: Entire buffer geometry + full urban boundary
            #    (complete analysis universe)
            all_mask = geom_mask | rural_geom_mask
            deviation_all = np.where(all_mask, lst - rural_mean, np.nan)
        else:
            deviation_urban_full = np.full_like(lst, np.nan)
            deviation_pixel = np.full_like(lst, np.nan)
            deviation_all = np.full_like(lst, np.nan)
        
        # Save all deviation maps
        self._save_final_deviation_maps(deviation_urban_full, deviation_pixel, deviation_all, profile, suhii)
                
        # Save JSON results with full uncertainty metrics
        results = {
            "suhii": float(suhii) if np.isfinite(suhii) else None,
            "suhii_standard_error": uncertainty['suhii_se'],
            "urban_mean": float(urban_mean) if np.isfinite(urban_mean) else None,
            "urban_std": uncertainty['urban_std'] if np.isfinite(uncertainty['urban_std']) else None,
            "urban_elevation_m": float(urban_elev) if np.isfinite(urban_elev) else None,
            "urban_pixels_total": int(np.count_nonzero(final_urban_mask)),
            "urban_pixels_valid": uncertainty['urban_valid_pixels'],
            "rural_mean": float(rural_mean) if np.isfinite(rural_mean) else None,
            "rural_std": uncertainty['rural_std'] if np.isfinite(uncertainty['rural_std']) else None,
            "rural_elevation_m": float(rural_elev) if np.isfinite(rural_elev) else None,
            "rural_pixels_total": int(np.count_nonzero(final_rural_mask)),
            "rural_pixels_valid": uncertainty['rural_valid_pixels'],
            "methodology": {
                "urban_method": self.cfg.urban_method,
                "rural_method": self.cfg.rural_method,
                "elevation_tolerance_m": float(elev_tol),
                "min_rural_distance_m": float(self.cfg.min_rural_distance),
                "lst_input_units": self.cfg.lst_units,
                "resampling": {
                    "lst": self.cfg.resample_lst,
                    "ndvi": self.cfg.resample_ndvi,
                    "dem": self.cfg.resample_dem,
                    "lulc": "nearest"
                }
            }
        }
        
        # Add normalized UHI metrics for incity method (Ahmad et al., 2024)
        if normalized_uhi is not None:
            results["normalized_uhi"] = {
                "uhi_normalized": normalized_uhi['uhi_normalized'],
                "study_area_mean": normalized_uhi['study_area_mean'],
                "study_area_std": normalized_uhi['study_area_std'],
                "formula": "(LST_urban - LST_study_area_mean) / LST_study_area_std",
                "citation": "Ahmad et al. (2024)",
                "units": "standard deviations (σ)"
            }
        
        results_path = os.path.join(self.cfg.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        self.logger.info(f"Saved results: {results_path}")


def generate_config_template(output_path: str = "config_template.json"):
    """
    Generate a template configuration file with all available options documented.
    
    Args:
        output_path (str): Path to save the template JSON file.
    """
    template = {
        "_comment": "SUHII Analysis Configuration Template - v6",
        "paths": {
            "input_dir": "./data/input",
            "output_dir": "./data/output",
            "lst_file": "LST.tif",
            "lulc_file": "LULC.tif",
            "ndvi_file": "NDVI.tif",
            "dem_file": "DEM.tif",
            "boundary_file": "boundary.geojson"
        },
        "lst_units": "celsius",
        "_lst_units_options": ["celsius", "kelvin", "celsius_scaled"],
        "urban_selection": {
            "method": "lulc_ndvi",
            "_method_options": ["lulc", "ndvi", "lulc_ndvi"],
            "_method_descriptions": {
                "lulc": "Pure LULC classification only",
                "ndvi": "NDVI threshold only - may include bare soil",
                "lulc_ndvi": "LULC + NDVI filter to exclude parks - RECOMMENDED"
            },
            "urban_classes": [6],
            "_urban_classes_note": "Dynamic World: 6=Built Area. ESA WorldCover: 50=Built-up",
            "water_class": 0,
            "nodata_value": 255,
            "ndvi_max_threshold": 0.3,
            "_ndvi_max_note": "Pixels with NDVI >= this value are excluded from urban (removes parks)"
        },
        "rural_selection": {
            "method": "buffer",
            "_method_options": ["buffer", "halo", "three_rings", "incity"],
            "buffer_params": {
                "fixed_width_m": 10000,
                "min_distance_from_edge_m": 0,
                "_min_distance_note": "For 'halo' method: distance to skip before buffer starts",
                "ring_type": "ua",
                "_ring_type_options": ["ua", "fua", "pua"],
                "min_rural_distance_m": 0,
                "_min_rural_distance_note": "For buffer/three_rings: exclude pixels closer than this to urban edge"
            },
            "vegetation_ndvi_threshold": 0.2,
            "_vegetation_note": "Rural pixels must have NDVI >= this value",
            "exclude_urban_lulc_classes": True
        },
        "filters": {
            "mask_water": True,
            "use_elevation_correction": True,
            "elevation_params": {
                "initial_tolerance_m": 50,
                "max_tolerance_m": 200,
                "step_m": 25,
                "min_valid_pixels": 100,
                "_note": "Raj & Yun (2024) use ±50m threshold"
            }
        },
        "outputs": {
            "generate_debug_plots": True
        },
        "resampling": {
            "lst": "nearest",
            "ndvi": "bilinear",
            "dem": "bilinear",
            "_options": ["nearest", "bilinear"],
            "_note": "LST: 'nearest' recommended to preserve measured radiometric values. "
                     "Bilinear creates artificial values at urban-rural boundaries. "
                     "LULC always uses nearest (categorical data)."
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=4)
    print(f"Config template saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--generate-config":
        output = sys.argv[2] if len(sys.argv) > 2 else "config_template.json"
        generate_config_template(output)
    elif os.path.exists("config.json"):
        analyzer = SUHIIAnalyzer("config.json")
        analyzer.run_analysis()
    else:
        print("Config file not found. Options:")
        print("  1. Create config.json based on the template")
        print("  2. Run with --generate-config to create a template:")
        print("     python suhii_tool.py --generate-config [output_path]")