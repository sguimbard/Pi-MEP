import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from numba import jit, prange
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings

# Grid configuration lookup table for faster access
GRID_CONFIGS = {
    'GLOBAL': {
        'Quarter_degree': (0.25, -179.875, 180, -89.875, 90),
        'Half_degree': (0.5, -179.75, 180, -89.75, 90),
        'One_degree': (1.0, -179.5, 180, -89.5, 90),
        'One_degree1': (1.0, -180, 181, -90, 91),
        'OneHalf_degree': (1.5, -179.25, 180, -89.25, 90),
        'Two_degree': (2.0, -179, 180, -89, 90),
        'Five_degree': (5.0, -177.5, 180, -87.5, 90)
    },
    'SPURS': {
        'base_bounds': (22.125, 25.875, -40.875, -35.125),
        'resolutions': {
            'Quarter_degree': 0.25, 'Half_degree': 0.5, 'One_degree': 1.0,
            'OneHalf_degree': 1.5, 'Two_degree': 2.0, 'Five_degree': 5.0
        }
    }
}

COLOR_LIMITS = {
    'SSS': [37, 38],
    'SWH': [0, 6],
    'WS': [2, 12]
}

@jit(nopython=True, parallel=True)
def _fast_grid_stats(x, y, z, weights, lon_edges, lat_edges):
    """
    Numba-optimized function for computing grid statistics
    """
    nlon, nlat = len(lon_edges) - 1, len(lat_edges) - 1
    n_data = len(x)
    
    # Pre-allocate output arrays
    counts = np.zeros((nlon, nlat), dtype=np.int32)
    sums = np.zeros((nlon, nlat))
    sum_squares = np.zeros((nlon, nlat))
    weighted_sums = np.zeros((nlon, nlat))
    weight_sums = np.zeros((nlon, nlat))
    
    # Vectorized binning
    for i in prange(n_data):
        if np.isnan(x[i]) or np.isnan(y[i]) or np.isnan(z[i]):
            continue
            
        # Find longitude bin
        lon_idx = -1
        for j in range(nlon):
            if lon_edges[j] <= x[i] < lon_edges[j + 1]:
                lon_idx = j
                break
        
        # Find latitude bin  
        lat_idx = -1
        for k in range(nlat):
            if lat_edges[k] <= y[i] < lat_edges[k + 1]:
                lat_idx = k
                break
        
        if lon_idx >= 0 and lat_idx >= 0:
            counts[lon_idx, lat_idx] += 1
            sums[lon_idx, lat_idx] += z[i]
            sum_squares[lon_idx, lat_idx] += z[i] * z[i]
            
            if len(weights) > 1 and not np.isnan(weights[i]):
                weighted_sums[lon_idx, lat_idx] += z[i] * weights[i]
                weight_sums[lon_idx, lat_idx] += weights[i]
    
    return counts, sums, sum_squares, weighted_sums, weight_sums

def _create_grid_coords(opt_region, space_res, x=None, y=None, lat_int=None, lon_int=None):
    """
    Optimized grid coordinate creation
    """
    opt_region = opt_region.upper()
    
    if opt_region == 'GLOBAL':
        if space_res not in GRID_CONFIGS['GLOBAL']:
            raise ValueError(f"Unknown resolution for GLOBAL: {space_res}")
        
        res, lon_start, lon_end, lat_start, lat_end = GRID_CONFIGS['GLOBAL'][space_res]
        grid_lon = np.arange(lon_start, lon_end, res)
        grid_lat = np.arange(lat_start, lat_end, res)
        
    elif opt_region == 'SPURS':
        if space_res not in GRID_CONFIGS['SPURS']['resolutions']:
            raise ValueError(f"Unknown resolution for SPURS: {space_res}")
            
        res = GRID_CONFIGS['SPURS']['resolutions'][space_res]
        lat0, lat1, lon0, lon1 = GRID_CONFIGS['SPURS']['base_bounds']
        
        if space_res == 'Five_degree':
            lat0, lon0 = -lat0, -lon0
            
        grid_lon = np.arange(lon0, lon1 + res/2, res)
        grid_lat = np.arange(lat0, lat1 + res/2, res)
        
    elif opt_region == 'LOCAL':
        if x is None or y is None:
            raise ValueError("x and y coordinates required for LOCAL region")
            
        # Use percentiles for more robust bounds
        lat0, lat1 = np.floor(np.nanmin(y)), np.ceil(np.nanmax(y))
        lon0, lon1 = np.floor(np.nanmin(x)), np.ceil(np.nanmax(x))
        
        res_map = {'Quarter_degree': 0.25, 'Half_degree': 0.5, 'One_degree': 1.0,
                   'OneHalf_degree': 1.5, 'Two_degree': 2.0, 'Five_degree': 5.0}
        
        if space_res not in res_map:
            raise ValueError(f"Unknown resolution: {space_res}")
            
        res = res_map[space_res]
        
        if space_res == 'Quarter_degree':
            grid_lon = np.arange(lon0 + res/2, lon1, res)
            grid_lat = np.arange(lat0 + res/2, lat1, res)
        elif space_res == 'Five_degree':
            grid_lon = np.arange(-lon0, lon1 + res/2, res)
            grid_lat = np.arange(-lat0, lat1 + res/2, res)
        else:
            grid_lon = np.arange(lon0, lon1 + res/2, res)
            grid_lat = np.arange(lat0, lat1 + res/2, res)
            
    elif opt_region == 'LOCAL1':
        if lat_int is None or lon_int is None:
            raise ValueError("lat_int and lon_int must be provided for LOCAL1 option")
            
        lat0, lat1 = np.min(lat_int), np.max(lat_int)
        lon0, lon1 = np.min(lon_int), np.max(lon_int)
        
        if space_res == 'Quarter_degree':
            res = 0.25
            grid_lon = np.arange(lon0 + res/2, lon1, res)
            grid_lat = np.arange(lat0 + res/2, lat1, res)
        else:
            raise NotImplementedError(f"LOCAL1 only supports Quarter_degree resolution")
    else:
        raise ValueError(f"Unknown region option: {opt_region}")
    
    return grid_lon, grid_lat

def make_griddata_stat(x, y, z, space_res, opt_plot=0, opt_region='LOCAL', 
                                weight=1, opt_variable='SSS', output_filename='output', 
                                lat_int=None, lon_int=None, use_scipy=True):
    """
        
    Parameters:
    -----------
    x, y, z : array-like
        Input coordinates and values
    space_res : str
        Spatial resolution ('Quarter_degree', 'Half_degree', etc.)
    opt_plot : int, default 0
        Plotting option
    opt_region : str, default 'LOCAL'
        Region option ('GLOBAL', 'SPURS', 'LOCAL', 'LOCAL1')
    weight : array-like or scalar, default 1
        Weights for weighted mean
    opt_variable : str, default 'SSS'
        Variable name
    output_filename : str, default 'output'
        Output filename
    lat_int, lon_int : array-like, optional
        Latitude and longitude intervals for LOCAL1 option
    use_scipy : bool, default True
        Use scipy.stats.binned_statistic_2d for even faster computation
        
    Returns:
    --------
    dict : Dictionary containing gridded statistics
    """
    
    # Convert to numpy arrays and ensure they're contiguous for numba
    x = np.ascontiguousarray(np.array(x, dtype=np.float64))
    y = np.ascontiguousarray(np.array(y, dtype=np.float64))
    z = np.ascontiguousarray(np.array(z, dtype=np.float64))
    
    # Handle weights
    if np.isscalar(weight) or len(np.atleast_1d(weight)) == 1:
        weights = np.ones_like(z)
        use_weights = False
    else:
        weights = np.ascontiguousarray(np.array(weight, dtype=np.float64))
        use_weights = True
        if len(weights) != len(z):
            raise ValueError("Weight array must have same length as data")
    
    # Create grid coordinates
    grid_lon, grid_lat = _create_grid_coords(opt_region, space_res, x, y, lat_int, lon_int)
    
    # Create meshgrid for output
    GridLon, GridLat = np.meshgrid(grid_lon, grid_lat)
    
    if use_scipy and len(x) > 1000:  # Use scipy for large datasets
        # Create bin edges
        dlon = np.diff(grid_lon)[0] if len(grid_lon) > 1 else 1.0
        dlat = np.diff(grid_lat)[0] if len(grid_lat) > 1 else 1.0
        
        lon_edges = np.concatenate([grid_lon - dlon/2, [grid_lon[-1] + dlon/2]])
        lat_edges = np.concatenate([grid_lat - dlat/2, [grid_lat[-1] + dlat/2]])
        
        # Remove NaN values for scipy processing
        valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x_clean, y_clean, z_clean = x[valid_mask], y[valid_mask], z[valid_mask]
        
        if len(x_clean) == 0:
            # Handle case with no valid data
            shape = (len(grid_lat), len(grid_lon))
            return {
                'LAT': GridLat, 'LON': GridLon,
                'ND': np.zeros(shape), 'MEAN': np.full(shape, np.nan),
                'STD': np.full(shape, np.nan), 'Space_res': space_res,
                'opt_region': opt_region
            }
        
        # Use scipy for fast binning
        mean_grid, _, _, _ = binned_statistic_2d(
            x_clean, y_clean, z_clean, statistic='mean', 
            bins=[lon_edges, lat_edges])
        
        count_grid, _, _, _ = binned_statistic_2d(
            x_clean, y_clean, z_clean, statistic='count',
            bins=[lon_edges, lat_edges])
        
        std_grid, _, _, _ = binned_statistic_2d(
            x_clean, y_clean, z_clean, statistic='std',
            bins=[lon_edges, lat_edges])
        
        # Transpose to match expected format
        mean_grid  = mean_grid.T
        count_grid = count_grid.T
        std_grid   = std_grid.T
        
        # Handle weighted mean if needed
        wmean_grid = None
        if use_weights:
            weights_clean = weights[valid_mask]
            
            # Custom weighted mean calculation
            def weighted_mean(values, weights=weights_clean):
                if len(values) == 0:
                    return np.nan
                # Get indices of current bin values
                return np.average(values, weights=weights[:len(values)])
            
            try:
                wmean_grid, _, _, _ = binned_statistic_2d(
                    x_clean, y_clean, z_clean, statistic=weighted_mean,
                    bins=[lon_edges, lat_edges])
                wmean_grid = wmean_grid.T
            except:
                # Fallback to unweighted if weighted computation fails
                wmean_grid = mean_grid.copy()
        
    else:
        # Use optimized numba implementation for smaller datasets or when scipy fails
        dlon = np.diff(grid_lon)[0] if len(grid_lon) > 1 else 1.0
        dlat = np.diff(grid_lat)[0] if len(grid_lat) > 1 else 1.0
        
        lon_edges = np.concatenate([grid_lon - dlon/2, [grid_lon[-1] + dlon/2]])
        lat_edges = np.concatenate([grid_lat - dlat/2, [grid_lat[-1] + dlat/2]])
        
        # Use numba-optimized function
        counts, sums, sum_squares, weighted_sums, weight_sums = _fast_grid_stats(
            x, y, z, weights, lon_edges, lat_edges)
        
        # Compute statistics
        count_grid = counts.T.astype(float)
        mean_grid = np.full_like(count_grid, np.nan)
        std_grid = np.full_like(count_grid, np.nan)
        wmean_grid = np.full_like(count_grid, np.nan) if use_weights else None
        
        # Vectorized computation of means and stds
        valid_counts = counts > 0
        mean_grid[valid_counts.T] = (sums / counts)[valid_counts].T
        
        # Compute standard deviation using the computational formula
        variance = np.zeros_like(sums)
        valid_for_std = counts > 1
        variance[valid_for_std] = (
            (sum_squares[valid_for_std] - sums[valid_for_std]**2 / counts[valid_for_std]) 
            / (counts[valid_for_std] - 1)
        )
        std_grid[valid_for_std.T] = np.sqrt(variance[valid_for_std]).T
        
        # Weighted mean
        if use_weights:
            valid_weights = weight_sums > 0
            wmean_grid[valid_weights.T] = (weighted_sums / weight_sums)[valid_weights].T
    
    # Prepare output dictionary
    output = {
        'LAT': GridLat,
        'LON': GridLon,
        'ND': count_grid,
        'MEAN': mean_grid,
        'STD': std_grid,
        'Space_res': space_res,
        'opt_region': opt_region
    }
    
    # Add weighted mean if computed
    if wmean_grid is not None:
        output['WMEAN'] = wmean_grid
    
    # Optimized plotting
    if opt_plot == 1:
        _create_plots(output, opt_variable)
    
    return output

def _create_plots(output, opt_variable):
    """
    Plotting function
    """
    # Get color limits
    opt_caxis = COLOR_LIMITS.get(opt_variable, 
                                [np.nanpercentile(output['MEAN'], 1), 
                                 np.nanpercentile(output['MEAN'], 95)])
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
    for ax in axes.flat:
        ax.add_feature(cfeature.LAND, facecolor='lightgrey', zorder=2)
    
    # Use more efficient plotting parameters
    plot_kwargs = {'shading': 'auto', 'rasterized': True}
    
    # Data density plot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        im1 = axes[0,0].pcolormesh(
            output['LON'], output['LAT'], output['ND'], 
            vmin=0, vmax=0.1*np.nanmax(output['ND']), transform=ccrs.PlateCarree(),
            cmap='jet', **plot_kwargs)
        axes[0,0].coastlines(resolution='110m')
        gl = axes[0,0].gridlines(draw_labels=True)
        gl.right_labels = False
        gl.top_labels = False
    axes[0,0].set_title('L2 DATA DENSITY MAP')
    plt.colorbar(im1, ax=axes[0,0], shrink=0.6, pad=0.02)
    
    # Mean plot
    im2 = axes[0,1].pcolormesh(
        output['LON'], output['LAT'], output['MEAN'], transform=ccrs.PlateCarree(),
        vmin=32, vmax=opt_caxis[1], 
        cmap='jet', **plot_kwargs)
    axes[0,1].coastlines(resolution='110m')
    gl = axes[0,1].gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    axes[0,1].set_title('L3 BIN AVERAGED MAP')
    plt.colorbar(im2, ax=axes[0,1], shrink=0.6, pad=0.02)

    # STD plot
    im3 = axes[1,0].pcolormesh(
        output['LON'], output['LAT'], output['STD'], 
        vmin=0, vmax=np.nanpercentile(output['STD'], 95), transform=ccrs.PlateCarree(),
        cmap='jet', **plot_kwargs)
    axes[1,0].coastlines(resolution='110m')
    gl = axes[1,0].gridlines(draw_labels=True)
    gl.right_labels = False
    gl.top_labels = False
    axes[1,0].set_title('L3 BIN STD MAP')
    plt.colorbar(im3, ax=axes[1,0], shrink=0.6, pad=0.02)

    # Remove empty subplot
    axes[1,1].remove()
    
    plt.tight_layout()
    plt.show()

def gridbin(x, y, z, grid_lon, grid_lat, func='mean', weights=None):
    """
    Highly optimized version using scipy.stats.binned_statistic_2d
    """
    # Create bin edges efficiently
    dlon = np.diff(grid_lon)[0] if len(grid_lon) > 1 else 1.0
    dlat = np.diff(grid_lat)[0] if len(grid_lat) > 1 else 1.0
    
    lon_edges = np.concatenate([grid_lon - dlon/2, [grid_lon[-1] + dlon/2]])
    lat_edges = np.concatenate([grid_lat - dlat/2, [grid_lat[-1] + dlat/2]])
    
    # Remove NaN values
    valid_mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    if not np.any(valid_mask):
        return np.full((len(grid_lat), len(grid_lon)), np.nan), \
               np.zeros((len(grid_lat), len(grid_lon)))
    
    x_clean, y_clean, z_clean = x[valid_mask], y[valid_mask], z[valid_mask]
    
    # Handle different statistics
    if isinstance(func, str):
        if func == 'mean':
            stat_func = 'mean'
        elif func == 'std':
            stat_func = 'std'
        elif func == 'count':
            stat_func = 'count'
        else:
            stat_func = func
    else:
        stat_func = func
    
    # Compute statistics
    stat, _, _, _ = binned_statistic_2d(
        x_clean, y_clean, z_clean, statistic=stat_func, 
        bins=[lon_edges, lat_edges])
    
    count, _, _, _ = binned_statistic_2d(
        x_clean, y_clean, z_clean, statistic='count', 
        bins=[lon_edges, lat_edges])
    
    return stat.T, count.T