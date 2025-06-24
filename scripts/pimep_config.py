"""
PIMEP Configuration Module
==========================

Configuration settings, mappings, and utilities for Pi-MEP data analysis.

Usage:
------
# At the beginning of your Jupyter notebook:
from pimep_config import PIMEPConfig

# Initialize config
config = PIMEPConfig()

# Access mappings
print(config.get_satellite_name('smos-l2-v700'))
print(config.get_insitu_name('argo'))
print(config.get_region_name('GO'))

# Access all mappings
satellites = config.satellite_map
regions = config.region_map
insitu = config.insitu_map
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt

class PIMEPConfig:
    """
    PIMEP Configuration class containing all mappings and settings.
    """
    
    def __init__(self):
        """Initialize PIMEP configuration with all mappings and settings."""
        
        # Core data mappings
        self.satellite_map = {
            "smos-l2-v700": "SMOS SSS L2 v700 (ESA)",
            "smap-l2-rss-v6": "SMAP SSS L2 v6 (RSS)",
            "smap-l2-rss-v6-40km": "SMAP SSS L2 v6 - 40 km (RSS)",
            "smap-l2-jpl-v5.0": "SMAP SSS L2 v5.0 (JPL)",
            "aquarius-l2-or-v5": "Aquarius SSS L2 OR v5 (NASA-GSFC)",
            "aquarius-l2-jpl-v5": "Aquarius SSS L2 CAP v5 (JPL)",       
            "smos-l2.5-v700": "SMOS SSS L2-AVERAGED v700 (ESA)",
            "smap-l2.5-rss-v6": "SMAP SSS L2-AVERAGED v6 (RSS)",
            "smap-l2.5-jpl-v5.0": "SMAP SSS L2-AVERAGED v5.0 (JPL)",
            "aquarius-l2.5-or-v5": "Aquarius SSS L2-AVERAGED OR v5 (NASA-GSFC)",
            "aquarius-l2.5-jpl-v5": "Aquarius SSS L2-AVERAGED CAP v5 (JPL)",    
            "smos-l3-cmems-v334-l2q": "SMOS SSS L2Q v334 (CMEMS-CATDS-CPDC)",
            "smos-l3-catds-cpdc-v332-9d": "SMOS SSS L3 v332 - 9 Days (CATDS-CPDC)",
            "smos-l3-catds-cpdc-v335-10d-25km": "SMOS SSS L3 v335 - 10 Days - 25 km (CATDS-CPDC)",
            "smos-l3-catds-cpdc-v335-1m-25km": "SMOS SSS L3 v335 - Monthly - 25 km (CATDS-CPDC)",
            "smos-l3-catds-locean-v9-9d": "SMOS SSS L3 v9 - 9 Days (CATDS-CEC-LOCEAN)",
            "smos-l3-catds-locean-v9-18d": "SMOS SSS L3 v9 - 18 Days (CATDS-CEC-LOCEAN)",
            "smos-l3-catds-locean-v10-9d": "SMOS SSS L3 v10 - 9 Days (CATDS-CEC-LOCEAN)",
            "smos-l3-catds-locean-v10-18d": "SMOS SSS L3 v10 - 18 Days (CATDS-CEC-LOCEAN)",
            "smos-l3-bec-v2-9d": "SMOS SSS L3 v2 - 9 Days (BEC)",
            "smap-l3-rss-v6-8dr": "SMAP SSS L3 v6 - 8-Day running (RSS)",
            "smap-l3-rss-v6-1m": "SMAP SSS L3 v6 - Monthly (RSS)",    
            "smap-l3-jpl-v5.0-8dr": "SMAP SSS L3 v5.0 - 8-Day running (JPL)",
            "smap-l3-jpl-v5.0-1m": "SMAP SSS L3 v5.0 - Monthly (JPL)",    
            "aquarius-l3-or-v5-7dr": "Aquarius SSS L3 OR v5 - 7-Day running (NASA-GSFC)",
            "aquarius-l3-or-v5-1m": "Aquarius SSS L3 OR v5 - Monthly (NASA-GSFC)",
            "aquarius-l3-or-v5-7dr-rain-mask": "Aquarius SSS L3 OR v5 rain-flagged - 7-Day running (NASA-GSFC)",
            "aquarius-l3-or-v5-1m-rain-mask": "Aquarius SSS L3 OR v5 rain-flagged - Monthly (NASA-GSFC)",
            "aquarius-l3-jpl-v5-7dr": "Aquarius SSS L3 CAP v5 - 7-Day running (JPL)",
            "aquarius-l3-jpl-v5-1m": "Aquarius SSS L3 CAP v5 - Monthly (JPL)",      
            "cci-l4-esa-merged-oi-v5.5-7dr": "CCI SSS L4 Merged-OI v5.5 - 7-day running (ESA)",
            "cci-l4-esa-merged-oi-v5.5-30dr": "CCI SSS L4 Merged-OI v5.5 - 30-day running (ESA)",
            "smos-l4-cmems-catds-lops-oi-v346-1w": "SMOS SSS L4 OI v346 - Weekly (CMEMS-CATDS-LOPS)",
            "smos-l4-cmems-cnr-v1-1d": "SMOS SSS L4 v1 - Daily (CMEMS-CNR)",
            "smos-l4-cmems-cnr-v1-1m": "SMOS SSS L4 v1 - Monthly (CMEMS-CNR)",
            "smos-l4-bec-v2-1d": "SMOS SSS L4 v2 - Daily (BEC)",
            "smap-l4-esr-oi-v2-7d": "SMAP SSS L4 OI v2 - 7 Days (ESR)",
            "smap-l4-esr-oi-v2-1m": "SMAP SSS L4 OI v2 - Monthly (ESR)", 
            "smap-l4-esr-oi-v3-1d": "SMAP SSS L4 OI v3 - Daily (ESR)",
            "smap-l4-esr-oi-v3-1m": "SMAP SSS L4 OI v3 - Monthly (ESR)",
            "aquarius-l4-iprc-v5-1w": "Aquarius SSS L4 OI v5 - Weekly (IPRC)",
            "aquarius-l4-iprc-v5-1m": "Aquarius SSS L4 OI v5 - Monthly (IPRC)", 
            "smos-l3-catds-locean-arctic-v2-9d": "SMOS L3 CATDS LOCEAN Arctic v2 9-day",
            "smos-l3-catds-locean-arctic-v2-18d": "SMOS L3 CATDS LOCEAN Arctic v2 18-day",
            "smos-l3-bec-med-atl-oa-v2-9d": "SMOS L3 BEC Mediterranean-Atlantic OA v2 9-day",
            "smos-l3-bec-arctic-v4-9d": "SMOS L3 BEC Arctic v4 9-day",
            "smos-l4-bec-baltic-v1-1d": "SMOS L4 BEC Baltic v1 Daily"
        }
        
        self.insitu_map = {
            "argo": "Argo profilers",
            "mammal": "Marine mammals",
            "drifter": "Surface drifters",
            "saildrone": "Saildrones",
            "tsg-legos-dm": "TSG (LEGOS-DM)",
            "tsg-legos-pacific": "TSG (LEGOS-PACIFIC)",    
            "tsg-gosud-research-vessel": "TSG (GOSUD-Research vessels)",    
            "tsg-gosud-sailing-ship": "TSG (GOSUD-Sailing ships)",
            "tsg-samos": "TSG (SAMOS)",
            "tsg-csic-utm": "TSG (CSIC-UTM)",
            "tsg-polarstern": "TSG (Polarstern)",
            "tsg-german-research-vessel": "TSG (German-Research vessels)",
            "tsg-ncei-0170743": "TSG (NCEI-0170743)",
            "tsg-amundsen": "TSG (Amundsen)",
            "tsg-lauge-koch": "TSG (Lauge-Koch)",
            "tsg-legos-survostral": "TSG (LEGOS-Survostral)",
            "tsg-legos-survostral-adelie": "TSG (LEGOS-Survostral adélie)",
            "saildrone-spurs2": "Saildrone (SPURS 2)",
            "snake": "Salinity Snake (SPURS 2)",
            "waveglider": "Waveglider (SPURS 2)",
            "seaglider": "Seaglider (SPURS 2)",
            "eurec4a": "EUREC4A",
            "ices": "ICES",
            "sassie": "SASSIE"
        }
        
        self.region_map = {
            "GO": "Global Ocean",
            "MLL-45": "Mid-Low Latitudes 45N-45S",
            "EO-10": "Equatorial Ocean 10N-10S",
            "AO": "Atlantic Ocean",
            "NAO": "North Atlantic Ocean",
            "SAO": "South Atlantic Ocean",
            "TAO": "Tropical Atlantic Ocean",
            "PO": "Pacific Ocean",
            "NPO": "North Pacific Ocean",
            "SPO": "South Pacific Ocean",
            "TPO": "Tropical Pacific Ocean",
            "IO": "Indian Ocean",
            "BoB": "Bay of Bengal",
            "SCS": "South China Sea",
            "SoJ": "Sea of Japan",
            "SO": "Southern Ocean",
            "ARCO": "Arctic Ocean",
            "BS": "Black Sea",
            "MS": "Mediterranean Sea",
            "SPURS1": "SPURS 1",
            "SPURS2": "SPURS 2",
            "SMOS-C-800-60": "SMOS coastal zone (<800km) 60N-60S",
            "OTT": "OTT zone",
            "GoM": "Gulf of Mexico",
            "AORP": "Amazon & Orinoco river plumes",
            "CRP": "Congo river plume",
            "MRP": "Mississippi river plume",
            "GBRP": "Ganga & Brahmaputra river plumes",
            "RFFF": "Roaring forties/Furious Fifties",
            "GS": "Gulf Stream"
        }
        
        # Plotting settings
        self.plot_settings = {
            'figure_size_default': (12, 8),
            'figure_size_three_panel': (18, 6),
            'figure_size_comparison': (10, 8),
            'dpi': 300,
            'font_size_large': 18,
            'font_size_medium': 14,
            'font_size_small': 12,
            'font_size_tiny': 10,
            'colormap_density': 'viridis',
            'colormap_argo_delayed': 'black',
            'colormap_argo_realtime': '#0A73E8',
            'alpha_default': 0.8,
            'point_size_default': 15,
            'point_size_small': 8,
            'line_width_default': 1,
            'grid_alpha': 0.3
        }
        # Core in situ variables
        core_vars = {
            'DATE', 'LATITUDE', 'LONGITUDE', 'SSS', 'SST', 'SSS_DEPTH', 'DELAYED_MODE',
            'PLATFORM_NUMBER', 'CYCLE_NUMBER', 'MLD'
        }
        
        # Satellite product variables
        satellite_vars = {
            'DATE_Satellite_product', 'LATITUDE_Satellite_product', 'LONGITUDE_Satellite_product', 
            'SSS_Satellite_product', 'SST_Satellite_product', 'Spatial_lags', 'Time_lags'
        }
        
        # Environmental variables
        env_vars = {
            'DISTANCE_TO_COAST', 'DISTANCE_TO_ICE_EDGE', 'ROSSBY_RADIUS', 'BATHYMETRY_ETOPO1',
            'SEA_ICE_CONCENTRATION', 'SLA'
        }
        
        # Wind variables
        wind_vars = {
            'WIND_SPEED_ASCAT_daily', 'WIND_STRESS_X_CMEMS_6h', 'WIND_STRESS_Y_CMEMS_6h', 
            'WIND_SPEED_CMEMS_6h', 'WIND_SPEED_MAXSS_1h', 'WIND_SPEED_CCMP_6h'
        }
        
        # Precipitation variables
        precip_vars = {
            'RAIN_RATE_CMORPH_3h', 'RAIN_RATE_IMERG_30min', 'EVAPORATION_OAFLUX'
        }
        
        # SSS climatology variables
        sss_clim_vars = {
            'SSS_ISAS', 'SSS_PCTVAR_ISAS', 'SSS_ISAS17', 'SSS_PCTVAR_ISAS17', 'SSS_ISAS20', 'SSS_PCTVAR_ISAS20',
            'SSS_WOA18', 'SSS_STD_WOA18', 'SSS_WOA23', 'SSS_STD_WOA23', 'SSS_WOA23_025', 'SSS_STD_WOA23_025',
            'SSS_SCRIPPS', 'SSS_IPRC', 'SSS_SSD_CMEMS', 'SSS_EN4', 'SSS_UNCERTAINTY_EN4', 'SSS_GLORYS'
        }
        
        # Ocean color variables
        color_vars = {
            'CDM_GLOBCOLOUR', 'CHL1_GLOBCOLOUR'
        }
        
        # Current variables
        current_vars = {
            'U_CMEMS_GLOBCURRENT', 'V_CMEMS_GLOBCURRENT', 'U_OSCAR_CURRENT', 'V_OSCAR_CURRENT'
        }
        
        # SST variables
        sst_vars = {
            'SST_AVHRR', 'SST_OSTIA', 'SST_CMC', 'SST_RSS'
        }
        
        # ERA5 variables
        era5_vars = {
            'ERA5_SHWW', 'ERA5_SWH', 'ERA5_PP1D', 'ERA5_MPWW', 'ERA5_U10', 'ERA5_V10', 
            'ERA5_RELATIVE_HUMIDITY', 'ERA5_SST', 'ERA5_BOUNDARY_LAYER_HEIGHT', 
            'ERA5_2m_TEMPERATURE', 'ERA5_MEAN_SEA_LEVEL_PRESSURE'
        }
        
        # Model comparison variables
        model_vars = {
            'SSS_ECCO_RMSD_Aquarius_Argo', 'SSS_ECCO_RMSD_SMAP_Argo', 'SSS_ECCO_RMSD_SMOS_Argo', 
            'SSS_GLORYS_AT_ARGO_DEPTH',
            'SSS_MEAN_GLORYS_AT_Satellite_product_resolution',
            'SSS_STD_GLORYS_AT_Satellite_product_resolution',
            'SSS_MEAN_GLORYS_AT_Satellite_product_temporal_resolution',
            'SSS_STD_GLORYS_AT_Satellite_product_temporal_resolution',
            'SSS_MEAN_GLORYS_AT_Satellite_product_spatial_resolution',
            'SSS_STD_GLORYS_AT_Satellite_product_spatial_resolution'
        }
        
        # Index variables
        index_vars = {'PIMEP_INDEX'}
        
        # Combine all variables
        self.coloc_variables = (core_vars | satellite_vars | env_vars | wind_vars | 
                               precip_vars | sss_clim_vars | color_vars | current_vars | 
                               sst_vars | era5_vars | model_vars | index_vars)
        
        # Also store categorized versions for easy access
        self.variable_categories = {
            'core': core_vars,
            'satellite': satellite_vars,
            'environmental': env_vars,
            'wind': wind_vars,
            'precipitation': precip_vars,
            'sss_climatology': sss_clim_vars,
            'ocean_color': color_vars,
            'currents': current_vars,
            'sst': sst_vars,
            'era5': era5_vars,
            'model_comparison': model_vars,
            'indices': index_vars
        }
        mdb_report_figure_name = {
            'DeltaSSS-vs-Depth.png','DeltaSSS-vs-Dist2coast.png','DeltaSSS-vs-RR.png','DeltaSSS-vs-SIC.png',
            'DeltaSSS-vs-sss.png','DeltaSSS-vs-sst.png','DeltaSSS-vs-ws.png','Dt-distribution.png',
            'Dx-distribution.png','Histogram-SSS-SAT-minus-INSITU-C1.png','Histogram-SSS-SAT-minus-INSITU-C2.png',
            'Histogram-SSS-SAT-minus-INSITU-C3.png','Histogram-SSS-SAT-minus-INSITU-C4.png',
            'Histogram-SSS-SAT-minus-INSITU-C5.png','Histogram-SSS-SAT-minus-INSITU-C6.png',
            'Number-of-SSS-vs-dist2coast.png','Number-of-SSS-vs-time.png','PRES-distribution.png',
            'Scatterplot-SSSdensity-Latband-0-20.png','Scatterplot-SSSdensity-Latband-0-80.png',
            'Scatterplot-SSSdensity-Latband-20-40.png','Scatterplot-SSSdensity-Latband-40-60.png',
            'Scatterplot-SSSdensity.png','Scatterplot-SSSdiff-vs-Time-C1.png',
            'Scatterplot-SSSdiff-vs-Time-Latband-0-20.png','Scatterplot-SSSdiff-vs-Time-Latband-0-80.png',
            'Scatterplot-SSSdiff-vs-Time-Latband-20-40.png','Scatterplot-SSSdiff-vs-Time-Latband-40-60.png',
            'Scatterplot-SSSdiff-vs-Time.png','SSS-INSITU-distribution.png','SSS-SAT-distribution.png',
            'Time-Mean-PRES.png','Time-Mean-SSS-INSITU.png','Time-Mean-SSS-SAT-minus-INSITU-C1.png',
            'Time-Mean-SSS-SAT-minus-INSITU-C2.png','Time-Mean-SSS-SAT-minus-INSITU-C3.png',
            'Time-Mean-SSS-SAT-minus-INSITU-C4.png','Time-Mean-SSS-SAT-minus-INSITU-C5.png',
            'Time-Mean-SSS-SAT-minus-INSITU-C6.png','Time-Mean-SSS-SAT-minus-INSITU.png',
            'Time-Mean-SSS-SAT.png','Time-Number-SSS.png','Time-STD-SSS-INSITU.png',
            'Time-STD-SSS-SAT-minus-INSITU.png','Time-STD-SSS-SAT.png','Zonally-averaged-time-mean-SSS.png'
        } 
        self.figure_names = {
            'mdb_report': mdb_report_figure_name
        }
        
        # File paths and naming conventions
        self.file_settings = {
            'figure_prefix': 'pimep-mdb-figure',
            'figure_extension': '.png',
            'output_directory': '../output/figures',
            'data_directory': '../data',
            'temp_directory': '../temp'
        }
        self.url = {
            'pimep_https': 'https://pimep.ifremer.fr/diffusion/',
            'pimep_analyses_mdb-database_https': 'https://pimep.ifremer.fr/diffusion/analyses/mdb-database/',
            'pimep_analyses_spectra_https': 'https://pimep.ifremer.fr/diffusion/analyses/spectra/',
            'pimep_analyses_triple-collocation_https': 'https://pimep.ifremer.fr/diffusion/analyses/triple-collocation/',
            'pimep_data_https': 'https://pimep.ifremer.fr/diffusion/data/',
            'pimep_data_concat_https': 'https://pimep.ifremer.fr/diffusion/data_concat/',
            'pimep_region_masks':'https://pimep.ifremer.fr/diffusion/mask/',
            'pimep_ftp': 'ftp://ftp.ifremer.fr/ifremer/cersat/pimep/diffusion/',
            'pimep_analyses_ftp': 'ftp://pimep.ifremer.fr/diffusion/analyses/',
            'pimep_data_ftp': 'ftp://pimep.ifremer.fr/diffusion/data/'
        }
        
        # Analysis settings
        self.analysis_settings = {
            'max_points_scatter': 50000,
            'max_points_stats': 10000,
            'percentile_limits': [2, 98],  # For robust axis limits
            'histogram_bins_sss': 100,
            'histogram_bins_time': 50,
            'outlier_iqr_factor': 1.5,
            'correlation_threshold': 0.7,
            'time_range_start': '2010-01-01',
            'time_range_end': '2025-12-31'
        }
    
    # Getter methods for mappings
    def get_satellite_name(self, satellite_id: str) -> str:
        """Get full satellite name from ID."""
        return self.satellite_map.get(satellite_id, satellite_id)
    
    def get_insitu_name(self, insitu_id: str) -> str:
        """Get full in-situ instrument name from ID."""
        return self.insitu_map.get(insitu_id, insitu_id)
    
    def get_region_name(self, region_id: str) -> str:
        """Get full region name from ID."""
        return self.region_map.get(region_id, region_id)
    
    # List methods
    def list_satellites(self) -> List[str]:
        """Get list of all satellite IDs."""
        return list(self.satellite_map.keys())
    
    def list_insitu(self) -> List[str]:
        """Get list of all in-situ instrument IDs."""
        return list(self.insitu_map.keys())
    
    def list_regions(self) -> List[str]:
        """Get list of all region IDs."""
        return list(self.region_map.keys())
    
    # Search methods
    def search_satellites(self, search_term: str) -> Dict[str, str]:
        """Search satellites by name or ID."""
        search_term = search_term.lower()
        return {k: v for k, v in self.satellite_map.items() 
                if search_term in k.lower() or search_term in v.lower()}
    
    def search_insitu(self, search_term: str) -> Dict[str, str]:
        """Search in-situ instruments by name or ID."""
        search_term = search_term.lower()
        return {k: v for k, v in self.insitu_map.items() 
                if search_term in k.lower() or search_term in v.lower()}
    
    def search_regions(self, search_term: str) -> Dict[str, str]:
        """Search regions by name or ID."""
        search_term = search_term.lower()
        return {k: v for k, v in self.region_map.items() 
                if search_term in k.lower() or search_term in v.lower()}
    
    # File naming methods
    def generate_filename(self, region_id: str, sat_id: str, insitu_id: str, 
                         plot_type: str, extension: str = None) -> str:
        """Generate standardized filename."""
        if extension is None:
            extension = self.file_settings['figure_extension']
        
        prefix = self.file_settings['figure_prefix']
        return f"{prefix}_{region_id}_{sat_id}_{insitu_id}_{plot_type}{extension}"
    
    def get_figure_path(self, filename: str, output_dir: str = None) -> Path:
        """Get full figure path."""
        if output_dir is None:
            output_dir = self.file_settings['output_directory']
        return Path(output_dir) / filename
    
    # Plotting helper methods
    def setup_matplotlib_defaults(self):
        """Setup default matplotlib settings for PIMEP plots."""
        plt.rcParams.update({
            'figure.figsize': self.plot_settings['figure_size_default'],
            'figure.dpi': self.plot_settings['dpi'],
            'font.size': self.plot_settings['font_size_medium'],
            'axes.labelsize': self.plot_settings['font_size_medium'],
            'axes.titlesize': self.plot_settings['font_size_large'],
            'xtick.labelsize': self.plot_settings['font_size_small'],
            'ytick.labelsize': self.plot_settings['font_size_small'],
            'legend.fontsize': self.plot_settings['font_size_small'],
            'grid.alpha': self.plot_settings['grid_alpha'],
            'savefig.dpi': self.plot_settings['dpi'],
            'savefig.bbox': 'tight'
        })
    
    # Validation methods
    def validate_satellite_id(self, satellite_id: str) -> bool:
        """Check if satellite ID is valid."""
        return satellite_id in self.satellite_map
    
    def validate_insitu_id(self, insitu_id: str) -> bool:
        """Check if in-situ ID is valid."""
        return insitu_id in self.insitu_map
    
    def validate_region_id(self, region_id: str) -> bool:
        """Check if region ID is valid."""
        return region_id in self.region_map
    
    # Configuration export/import
    def export_config(self, filepath: Union[str, Path]):
        """Export configuration to JSON file."""
        config_dict = {
            'satellite_map': self.satellite_map,
            'insitu_map': self.insitu_map,
            'region_map': self.region_map,
            'plot_settings': self.plot_settings,
            'file_settings': self.file_settings,
            'analysis_settings': self.analysis_settings
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def from_config_file(cls, filepath: Union[str, Path]):
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        instance = cls()
        instance.satellite_map = config_dict.get('satellite_map', instance.satellite_map)
        instance.insitu_map = config_dict.get('insitu_map', instance.insitu_map)
        instance.region_map = config_dict.get('region_map', instance.region_map)
        instance.plot_settings = config_dict.get('plot_settings', instance.plot_settings)
        instance.file_settings = config_dict.get('file_settings', instance.file_settings)
        instance.analysis_settings = config_dict.get('analysis_settings', instance.analysis_settings)
        
        return instance
    
    # Convenience methods for common operations
    def get_argo_colors(self) -> tuple:
        """Get standard colors for Argo delayed/realtime modes."""
        return (self.plot_settings['colormap_argo_delayed'], 
                self.plot_settings['colormap_argo_realtime'])
    
    def get_figure_size(self, plot_type: str = 'default') -> tuple:
        """Get figure size for different plot types."""
        size_map = {
            'default': 'figure_size_default',
            'three_panel': 'figure_size_three_panel',
            'comparison': 'figure_size_comparison'
        }
        size_key = size_map.get(plot_type, 'figure_size_default')
        return self.plot_settings[size_key]
    
    def __repr__(self) -> str:
        """String representation of config object."""
        return (f"PIMEPConfig(\n"
                f"  Satellites: {len(self.satellite_map)} entries\n"
                f"  In-situ: {len(self.insitu_map)} entries\n"
                f"  Regions: {len(self.region_map)} entries\n"
                f")")


# Global instance for easy importing
pimep_config = PIMEPConfig()

# Convenience functions for direct import
def get_satellite_name(satellite_id: str) -> str:
    """Get satellite name from global config."""
    return pimep_config.get_satellite_name(satellite_id)

def get_insitu_name(insitu_id: str) -> str:
    """Get in-situ name from global config."""
    return pimep_config.get_insitu_name(insitu_id)

def get_region_name(region_id: str) -> str:
    """Get region name from global config."""
    return pimep_config.get_region_name(region_id)

def setup_pimep_plotting():
    """Setup matplotlib defaults for PIMEP."""
    pimep_config.setup_matplotlib_defaults()


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    config = PIMEPConfig()
    
    print("=== PIMEP Configuration Example ===")
    print(f"Satellite: {config.get_satellite_name('smos-l2-v700')}")
    print(f"In-situ: {config.get_insitu_name('argo')}")
    print(f"Region: {config.get_region_name('GO')}")
    
    print(f"\nSearch SMOS satellites:")
    smos_sats = config.search_satellites('smos')
    for k, v in list(smos_sats.items())[:3]:
        print(f"  {k}: {v}")
    
    print(f"\nGenerate filename:")
    filename = config.generate_filename('ATL', 'SMOS', 'argo', 'SSS-comparison')
    print(f"  {filename}")
    
    print(f"\nConfiguration summary:")
    print(config)
