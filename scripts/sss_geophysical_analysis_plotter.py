import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlotConfig:
    """Configuration for different plot sections"""
    x_label: str
    dx: float
    dtick: float  # tick interval for x-axis
    bin_range: Tuple[float, float, float]  # start, stop, step
    ax2_xlim: Tuple[float, float]
    ax3_xlim: Tuple[float, float]
    filename_suffix: str

class SSSAnalysisPlotter:
    """Class for plotting SSS (Sea Surface Salinity) analysis comparisons"""
    
    # Plot configurations for different geophysical conditions
    PLOT_CONFIGS = {
        'sss': PlotConfig('SSS', 0.2, 1, (30.1, 40, 0.2), (30, 39), (-1, 1), 'sss'),
        'sst': PlotConfig('SST [deg C]', 1, 2, (-1.5, 32, 1), (-2, 32), (-1, 1), 'sst'),
        'wind_speed_ascat': PlotConfig('Wind Speed [m/s]', 1, 2, (0.5, 25, 1), (0, 25), (-1, 1), 'ws'),
        'rain_rate_cmorph': PlotConfig('Rain Rate [mm/h]', 2, 2, (0, 40, 2), (0, 40), (-1, 1), 'RR'),
        'distance_to_coast': PlotConfig('Distance to coasts [km]', 50, 250, (25, 3000, 50), (0, 2500), (-1, 1), 'Dist2coast'),
        'depth': PlotConfig('Depth measurement [m]', 1, 1, (0.5, 10, 1), (0, 10), (-1, 1), 'Depth'),
        'sea_ice_concentration': PlotConfig('Sea Ice Fraction [%]', 10, 10, (5, 100, 10), (0, 100), (-1, 1), 'SIC'),
        'distance_to_ice_edge': PlotConfig('Distance to ice edge [km]', 50, 250, (25, 3000, 50), (0, 2500), (-1, 1), 'Dist2iceedge'),
    }
    
    def __init__(self, pathout: str, fig_id: str, sat_name: str, insitu_name: str, dpi: int = 150):
        """
        Initialize the plotter with output configuration
        
        Args:
            pathout: Output path for figures
            fig_id: Figure identifier for naming
            sat_name: Satellite data name
            insitu_name: In-situ data name
            dpi: Figure resolution
        """
        self.pathout = Path(pathout)
        self.pathout.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist
        self.fig_id = fig_id
        self.sat_name = sat_name
        self.insitu_name = insitu_name
        self.dpi = dpi
    
    def extract_geophysical_data(self, ana: Dict[str, Any], indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract and prepare geophysical data from analysis dictionary
        
        Args:
            ana: Analysis data dictionary or dataset object
            indices: Data indices to extract (boolean mask or integer indices)
            
        Returns:
            Dictionary of extracted geophysical variables
        """
        # Handle different data access patterns (dict vs object attributes)
        def safe_get_data(key, fallback_keys=None):
            """Safely extract data from ana with fallback options"""
            fallback_keys = fallback_keys or []
            
            # Try main key first
            for k in [key] + fallback_keys:
                if hasattr(ana, k):
                    data = getattr(ana, k)
                    if hasattr(data, '__len__') and len(data) > 0:
                        return data
                elif isinstance(ana, dict) and k in ana:
                    data = ana[k]
                    if hasattr(data, '__len__') and len(data) > 0:
                        return data
            
            # Return NaN array if not found
            reference_data = None
            if hasattr(ana, 'SSS'):
                reference_data = ana.SSS
            elif hasattr(ana, 'sss'):
                reference_data = ana.sss
            elif isinstance(ana, dict) and 'sss' in ana:
                reference_data = ana['sss']
            
            if reference_data is not None:
                return np.full_like(reference_data, np.nan)
            else:
                return np.array([])
        
        # Handle boolean mask vs integer indices
        if indices.dtype == bool:
            def extract_with_mask(data):
                return data[indices] if hasattr(data, '__len__') and len(data) > 0 else np.array([])
        else:
            def extract_with_mask(data):
                return data[indices] if hasattr(data, '__len__') and len(data) > 0 else np.array([])
        
        # Extract data with multiple possible key names
        data = {
            'sst': extract_with_mask(safe_get_data('sst', ['SST'])),
            'rain_rate_cmorph': extract_with_mask(safe_get_data('rain_rate_cmorph', ['RAIN_RATE_CMORPH_3h'])),
            'wind_speed_ascat': extract_with_mask(safe_get_data('wind_speed_ascat', ['WIND_SPEED_ASCAT_daily'])),
            'distance_to_coast': extract_with_mask(safe_get_data('distance_to_coast', ['DISTANCE_TO_COAST'])),
            'depth': extract_with_mask(safe_get_data('depth', ['depth', 'SSS_DEPTH'])),
            'sea_ice_concentration': extract_with_mask(safe_get_data('sea_ice_concentration', ['SEA_ICE_CONCENTRATION'])),
            'distance_to_ice_edge': extract_with_mask(safe_get_data('distance_to_ice_edge', ['DISTANCE_TO_ICE_EDGE'])),
            'sss_isas': extract_with_mask(safe_get_data('sss_isas', ['SSS_ISAS'])),
            'rain_rate_imerg': extract_with_mask(safe_get_data('rain_rate_imerg', ['RAIN_RATE_IMERG_30min'])),
            'wind_speed_ccmp': extract_with_mask(safe_get_data('wind_speed_ccmp', ['WIND_SPEED_CCMP_6h'])),
            'sst_era5': extract_with_mask(safe_get_data('sst_era5', ['ERA5_SST'])),
            'sst_cmc': extract_with_mask(safe_get_data('sst_cmc', ['SST_CMC'])),
            'sst_avhrr': extract_with_mask(safe_get_data('sst_avhrr', ['SST_AVHRR'])),
        }
        
        # Convert ERA5 temperature from Kelvin to Celsius if needed
        if len(data['sst_era5']) > 0 and np.nanmean(data['sst_era5']) > 100:
            data['sst_era5'] = data['sst_era5'] - 273.15
        
        return data
    
    def calculate_binned_statistics(self, x_data: np.ndarray, delta_sss: np.ndarray, 
                                  bin_centers: np.ndarray, dx: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate median and standard deviation for binned data
        
        Args:
            x_data: X-axis data
            delta_sss: SSS difference data
            bin_centers: Bin center values
            dx: Bin width
            
        Returns:
            Tuple of (medians, standard_deviations)
        """
        medians = np.full(len(bin_centers), np.nan)
        stds = np.full(len(bin_centers), np.nan)
        
        for i, bin_center in enumerate(bin_centers):
            # Find data within bin
            in_bin = np.abs(x_data - bin_center) <= dx / 2
            bin_data = delta_sss[in_bin]
            bin_data = bin_data[~np.isnan(bin_data)]
            
            if bin_data.size > 0:
                medians[i] = np.median(bin_data)
                stds[i] = np.std(bin_data)
        
        return medians, stds
    
    def create_subplot_layout(self, fig=None, figsize=(12, 8)) -> Tuple[plt.Axes, plt.Axes, plt.Axes]:
        """Create the three-panel subplot layout"""
        if fig is None:
            plt.clf()
            plt.figure(figsize=figsize)
            
            # Top histogram
            ax1 = plt.axes([0.13, 0.84, 0.7, 0.0673])
            
            # Main scatter plot
            ax2 = plt.axes([0.13, 0.11, 0.7, 0.72])
            
            # Right histogram
            ax3 = plt.axes([0.84, 0.11, 0.035, 0.72])
        else:
            # For grid layout, axes are created externally
            ax1, ax2, ax3 = None, None, None
        
        return ax1, ax2, ax3
    
    def plot_top_histogram(self, ax: plt.Axes, x_data: np.ndarray, bin_centers: np.ndarray):
        """Plot the top histogram showing data distribution"""
        if len(x_data) == 0:
            return
            
        counts, bin_edges, _ = ax.hist(x_data, bins=bin_centers, alpha=0)
        normalized_counts = counts / np.max(counts) if np.max(counts) > 0 else counts
        
        ax.bar(bin_edges[:-1], normalized_counts, width=np.diff(bin_edges)[0] * 0.7, color='k')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.set_xlim(bin_centers[0], bin_centers[-1])
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(left=False) 
    
    def plot_main_scatter(self, ax: plt.Axes, x_data: np.ndarray, delta_sss: np.ndarray,
                         bin_centers: np.ndarray, medians: np.ndarray, stds: np.ndarray,
                         config: PlotConfig, data_name: str, fontsize: int = 15):
        """Plot the main scatter plot with error bars"""
        if len(x_data) == 0:
            return
            
        # Scatter plot
        ax.plot(x_data, delta_sss, '.', markersize=4, color=[0.5, 0.5, 0.5], alpha=0.4)
        
        # Error bars
        ax.errorbar(bin_centers, medians, yerr=stds, color='k', linewidth=2, capsize=3)
        
        # Reference line at zero
        ax.plot(bin_centers, np.zeros_like(bin_centers), '--b', linewidth=2, alpha=0.7)
        
        # Median line
        valid_mask = ~np.isnan(medians)
        ax.plot(bin_centers[valid_mask], medians[valid_mask], 'kd-', linewidth=2, markersize=6)
        
        # Labels and limits
        ax.set_xlabel(f'{config.x_label}', fontsize=fontsize)
        ax.set_ylabel(f'Δ SSS ({self.sat_name} - {self.insitu_name})', fontsize=fontsize)
        ax.set_xlim(config.ax2_xlim)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(config.ax2_xlim[0], config.ax2_xlim[1], config.dtick))  
        ax.set_yticks(np.arange(-1, 1, 0.2))
        ax.grid(True, alpha=0.3)
        major_tick_params = {'axis': 'both', 'which': 'major', 'labelsize': fontsize}
        ax.tick_params(**major_tick_params)
    
    def plot_right_histogram(self, ax: plt.Axes, delta_sss: np.ndarray, config: PlotConfig):
        """Plot the right histogram showing delta SSS distribution"""
        valid_data = delta_sss[~np.isnan(delta_sss)]
        
        if len(valid_data) > 0:
            bins = np.arange(-3, 3.1, 0.1)
            counts, bin_edges, _ = ax.hist(valid_data, bins=bins, alpha=0, orientation='horizontal')
            normalized_counts = counts / np.max(counts) if np.max(counts) > 0 else counts
            
            ax.barh(bin_edges[:-1], normalized_counts, height=np.diff(bin_edges)[0] * 0.7, color='k')
            
            # Median line
            median_val = np.nanmedian(valid_data)
            ax.plot([0, 1], [median_val, median_val], 'r', linewidth=2)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(config.ax3_xlim)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(left=False) 
    
    def plot_single_section(self, x_data: np.ndarray, delta_sss: np.ndarray, 
                           config: PlotConfig, data_name: str, save_fig: bool = True):
        """
        Plot a single analysis section
        
        Args:
            x_data: X-axis data
            delta_sss: SSS difference data
            config: Plot configuration
            data_name: Name for the data being plotted
            save_fig: Whether to save the figure
        """
        # Check if we have valid data
        if np.all(np.isnan(delta_sss)) or len(x_data) == 0:
            if save_fig:
                plt.clf()
                plt.text(0.5, 0.5, 'NO DATA', fontsize=18, fontweight='bold', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                output_filename = self.pathout / f"{self.fig_id}_DeltaSSS-vs-{config.filename_suffix}.png"
                plt.savefig(output_filename, dpi=self.dpi, bbox_inches='tight')
                logger.info(f"Saved plot (NO DATA): {output_filename}")
            return
        
        # Create bin centers
        bin_centers = np.arange(*config.bin_range)
        
        # Calculate binned statistics
        medians, stds = self.calculate_binned_statistics(x_data, delta_sss, bin_centers, config.dx)
        
        # Create subplot layout
        ax1, ax2, ax3 = self.create_subplot_layout()
        
        # Plot each panel
        self.plot_top_histogram(ax1, x_data, bin_centers)
        self.plot_main_scatter(ax2, x_data, delta_sss, bin_centers, medians, stds, config, data_name)
        self.plot_right_histogram(ax3, delta_sss, config)
        
        # Save figure
        if save_fig:
            output_filename = self.pathout / f"{self.fig_id}_DeltaSSS-vs-{config.filename_suffix}.png"
            plt.savefig(output_filename, dpi=self.dpi, bbox_inches='tight')
            plt.close()  
            logger.info(f"Saved plot: {output_filename}")
        else:
            plt.show()
    
    def plot_multiple_sections_grid(self, sections: List[Tuple], delta_sss: np.ndarray, 
                                   ncols: int = 3, save_fig: bool = True, show_fig: bool = True):
        """
        Plot multiple 3-panel sections in a grid layout.
        
        Args:
            sections: List of tuples (x_data, config, data_name)
            delta_sss: Precomputed delta SSS, masked to open ocean
            ncols: Number of columns in the grid
            save_fig: Whether to save the figure
            show_fig: Whether to show the figure
        """
        if not sections:
            logger.warning("No sections to plot")
            return
        
        nrows = int(np.ceil(len(sections) / ncols))
        fig = plt.figure(figsize=(ncols * 6, nrows * 5))
        gs = gridspec.GridSpec(nrows * 3, ncols * 4, figure=fig, hspace=0.4, wspace=0.3)

        for i, (x_data, config, data_name) in enumerate(sections):
            if len(x_data) == 0 or np.all(np.isnan(x_data)):
                continue
                
            bin_centers = np.arange(*config.bin_range)
            medians, stds = self.calculate_binned_statistics(x_data, delta_sss, bin_centers, config.dx)

            row = (i // ncols) * 3
            col = i % ncols

            # Create the three sub-axes for this variable
            ax_top = fig.add_subplot(gs[row, col*4:(col+1)*4-1])
            ax_main = fig.add_subplot(gs[row + 1:row + 3, col*4:(col+1)*4-1])
            ax_right = fig.add_subplot(gs[row + 1:row + 3, (col+1)*4-1])

            # Top histogram
            self.plot_top_histogram(ax_top, x_data, bin_centers)
            ax_top.set_title(config.x_label, fontsize=12, fontweight='bold')

            # Main scatter
            self.plot_main_scatter(ax_main, x_data, delta_sss, bin_centers, medians, stds, 
                                 config, data_name, fontsize=10)

            # Right histogram
            self.plot_right_histogram(ax_right, delta_sss, config)

        fig.suptitle(f'Delta SSS vs Geophysical Parameters\n({self.sat_name} - {self.insitu_name})', 
                    fontsize=16, fontweight='bold')
        
        if save_fig:
            output_path = self.pathout / f"{self.fig_id}_grid_all_panels.png"
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved grid plot: {output_path}")
        
        if show_fig:
            plt.show()
        
        if not show_fig:
            plt.close(fig)
    
    def get_available_sections(self, geo_data: Dict[str, np.ndarray], 
                              sss_insitu_filtered: np.ndarray) -> List[Tuple]:
        """
        Get list of available sections based on data availability
        
        Args:
            geo_data: Dictionary of geophysical data
            sss_insitu_filtered: Filtered in-situ SSS data
            
        Returns:
            List of tuples (x_data, config, data_name)
        """
        sections = []
        
        # Standard plots
        plot_mappings = [
            (geo_data['sst'], 'sst', f'{self.insitu_name} SST'),
            (sss_insitu_filtered, 'sss', f'{self.insitu_name} SSS'),
            (geo_data['wind_speed_ascat'], 'wind_speed_ascat', 'Wind Speed'),
            (geo_data['rain_rate_cmorph'], 'rain_rate_cmorph', 'Rain Rate'),
            (geo_data['distance_to_coast'], 'distance_to_coast', 'Distance to coasts'),
            (geo_data['sea_ice_concentration'], 'sea_ice_concentration', 'Sea Ice Fraction'),
            (geo_data['distance_to_ice_edge'], 'distance_to_ice_edge', 'Distance to ice edge'),
        ]
        
        # Add depth plot if data exists
        if len(geo_data['depth']) > 0:
            plot_mappings.append((geo_data['depth'], 'depth', 'Depth measurement'))
        
        # Add standard sections
        for x_data, config_key, data_name in plot_mappings:
            if config_key in self.PLOT_CONFIGS and len(x_data) > 0 and not np.all(np.isnan(x_data)):
                sections.append((x_data, self.PLOT_CONFIGS[config_key], data_name))
        
        # Additional dataset plots with custom configurations
        additional_plots = [
            (geo_data['sss_isas'], 'ISAS SSS', 'sssISAS', 'sss'),
            (geo_data['sst_cmc'], 'CMC SST [deg C]', 'sstCMC', 'sst'),
            (geo_data['sst_era5'], 'ERA5 SST [deg C]', 'sstERA5', 'sst'),
            (geo_data['sst_avhrr'], 'AVHRR SST [deg C]', 'sstAVHRR', 'sst'),
            (geo_data['wind_speed_ccmp'], 'CCMP 6h Wind Speed [m/s]', 'wsCCMP', 'wind_speed_ascat'),
            (geo_data['rain_rate_imerg'], 'IMERG Rain Rate [mm/h]', 'RR-IMERG', 'rain_rate_cmorph'),
        ]
        
        # Add additional sections with custom configurations
        for x_data, x_label, suffix, base_config_key in additional_plots:
            if len(x_data) > 0 and not np.all(np.isnan(x_data)) and base_config_key in self.PLOT_CONFIGS:
                base_config = self.PLOT_CONFIGS[base_config_key]
                config = PlotConfig(x_label, base_config.dx, base_config.dtick, 
                                  base_config.bin_range, base_config.ax2_xlim, 
                                  base_config.ax3_xlim, suffix)
                sections.append((x_data, config, x_label))
        
        return sections
    
    def plot_all_sections(self, sss_sat: np.ndarray, sss_insitu: np.ndarray, 
                         ana: Dict[str, Any], indices: np.ndarray, 
                         mode: str = 'individual', ncols: int = 3):
        """
        Plot all geophysical condition sections
        
        Args:
            sss_sat: Satellite SSS data
            sss_insitu: In-situ SSS data
            ana: Analysis data dictionary or dataset object
            indices: Data indices to use (boolean mask or integer indices)
            mode: 'individual' for separate plots, 'grid' for grid layout, 'both' for both
            ncols: Number of columns for grid layout
        """
        # Handle boolean mask for indices
        if indices.dtype == bool:
            sss_sat_filtered = sss_sat[indices]
            sss_insitu_filtered = sss_insitu[indices]
        else:
            sss_sat_filtered = sss_sat[indices]
            sss_insitu_filtered = sss_insitu[indices]
        
        # Extract geophysical data
        geo_data = self.extract_geophysical_data(ana, indices)
        
        # Handle cases where dist2coast might not exist or be empty
        if len(geo_data['distance_to_coast']) == 0:
            # If no distance to coast data, use all data
            open_ocean_mask = np.ones(len(sss_sat_filtered), dtype=bool)
            logger.warning("No distance to coast data found, using all data points")
        else:
            # Filter out coastal data (distance to coast < 0)
            coastal_mask = geo_data['distance_to_coast'] < 0
            open_ocean_mask = ~coastal_mask
        
        # Calculate SSS difference for open ocean only
        delta_sss = sss_sat_filtered[open_ocean_mask] - sss_insitu_filtered[open_ocean_mask]
        
        # Apply open ocean mask to geophysical data
        for key in geo_data:
            if len(geo_data[key]) == len(open_ocean_mask):
                geo_data[key] = geo_data[key][open_ocean_mask]
        
        # Get available sections
        sections = self.get_available_sections(geo_data, sss_insitu_filtered[open_ocean_mask])
        
        if mode in ['individual', 'both']:
            # Plot individual sections
            for x_data, config, data_name in sections:
                self.plot_single_section(x_data, delta_sss, config, data_name, save_fig=True)
        
        if mode in ['grid', 'both']:
            # Plot grid layout
            self.plot_multiple_sections_grid(sections, delta_sss, ncols=ncols, 
                                           save_fig=True, show_fig=True)

def plot_deltaSSS_sorted_by_geophys_cond(sss_sat: np.ndarray, sss_insitu: np.ndarray, 
                                        ana: Dict[str, Any], ind: np.ndarray, 
                                        pathout: str, fig_id: str, SAT_name: str, 
                                        insitu_database_name: str, h1: int = 300,
                                        mode: str = 'individual', ncols: int = 3):
    """
    Main function to plot Delta SSS sorted by geophysical conditions
    
    Args:
        sss_sat: Satellite SSS data
        sss_insitu: In-situ SSS data
        ana: Analysis data dictionary or dataset object containing geophysical variables
        ind: Indices to use for analysis (boolean mask or integer array)
        pathout: Output path for figures
        fig_id: Figure identifier
        SAT_name: Satellite dataset name
        insitu_database_name: In-situ dataset name
        h1: Figure DPI resolution
        mode: 'individual' for separate plots, 'grid' for grid layout, 'both' for both
        ncols: Number of columns for grid layout (default: 3)
    """
    plotter = SSSAnalysisPlotter(pathout, fig_id, SAT_name, insitu_database_name, h1)
    plotter.plot_all_sections(sss_sat, sss_insitu, ana, ind, mode=mode, ncols=ncols)
