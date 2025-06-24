import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import warnings
from grid_stats import make_griddata_stat
warnings.filterwarnings('ignore')

def plot_conditional_analyses(mdb, pathfig, regindbasinmask, sat_product, insitu_product, 
                            insitu_database, threshold, res, lat_int, lon_int, h1, 
                            opt_plot=True, opt_insitu=False):
    """
    Python translation of MATLAB plot_conditional_analyses function
    
    Parameters:
    - mdb: Dictionary containing matchup database
    - pathfig: Path for figure output
    - regindbasinmask: Region/basin mask object with 'id' attribute
    - sat_product: Satellite product name
    - insitu_product: In-situ product name
    - insitu_database: In-situ database name
    - threshold: Dictionary with threshold values (RR, U10, etc.)
    - res: Resolution
    - lat_int: Latitude interval [min, max]
    - lon_int: Longitude interval [min, max]
    - h1: Handle/figure parameter
    - opt_plot: Boolean, whether to create plots (default True)
    - opt_insitu: Boolean, whether to use in-situ mode (default False)
    """
    
    # Set default parameters
    if opt_insitu:
        fign = f'pimep-insitu-figure_{insitu_product}'
        sat_name = 'ISAS'
    else:
        fign = f'pimep-mdb-figure_{regindbasinmask.id}_{sat_product}_{insitu_product}'
        sat_name = 'SAT'
    
    res_str = str(res)
    
    # Extract all data (no filtering by ind)
    if len(mdb['DATE'].shape) == 1:
        ana = {
            'date': mdb['DATE'],
            'lon': mdb['LONGITUDE'],
            'lat': mdb['LATITUDE'], 
            'sss': mdb['SSS'],
            'sst': mdb['SST'],
            'isas_sss': mdb['SSS_ISAS'],
            'isas_sss_pctvar': mdb['SSS_PCTVAR_ISAS'],
            'dist2coast': mdb['DISTANCE_TO_COAST'],
            'RR': mdb['RAIN_RATE_IMERG_30min'],
            'u': mdb['WIND_SPEED_CCMP_6h'],
            'woa23_std': mdb['SSS_STD_WOA23']
        }
        
        # Optional fields
        if 'MLD' in mdb:
            ana['MLD'] = mdb['MLD']
        if 'BLT' in mdb:
            ana['BLT'] = mdb['BLT']
    
    # Set satellite SSS data
    if opt_insitu:
        ana['sss_sat'] = mdb['isas_sss']
    else:
        ana['sss_sat'] = mdb['SSS_Satellite_product']
    
    # Handle Argo data mode if applicable
    if insitu_product == 'argo':
        if 'DELAYED_MODE' in mdb:
            ana['DELAYED_MODE'] = mdb['DELAYED_MODE'].astype(bool)
        else:
            # Assuming DATA_MODE is a string array
            dm_flags = np.array([1 if mode == 'D' else 0 for mode in mdb['DATA_MODE']])
            ana['DELAYED_MODE'] = dm_flags.astype(bool)
    
    # Get year range for titles
    if len(ana['DATE']) > 0 and not np.all(np.isnan(ana['DATE'])):
        # Assuming date is in datetime format or numeric
        yearmin = str(int(np.nanmin(ana['DATE'])))[:4] if hasattr(ana['DATE'], 'year') else ''
        yearmax = str(int(np.nanmax(ana['DATE'])))[:4] if hasattr(ana['DATE'], 'year') else ''
    else:
        yearmin = ''
        yearmax = ''
    
    # Calculate difference: satellite - in-situ
    y = ana['sss_sat'] - ana['sss']
    
    # Define conditions
    cond = {}
    
    # Condition C1: Distance to coast >= 800km, wind 3-12 m/s, no rain, SST > 5°C
    cond['C1'] = ((ana['dist2coast'] >= 800) & 
                  (ana['u'] >= 3) & (ana['u'] <= 12) & 
                  (ana['RR'] == 0) & 
                  (ana['sst'] > 5) & 
                  (~np.isnan(ana['sss_sat'])))
    y_C1 = y[cond['C1']]
    
    # Condition C2: Wind 3-12 m/s, no rain
    cond['C2'] = ((ana['u'] >= 3) & (ana['u'] <= 12) & 
                  (ana['RR'] == 0) & 
                  (~np.isnan(ana['sss_sat'])))
    y_C2 = y[cond['C2']]
    
    # Condition C3: High rain, low wind
    cond['C3'] = ((ana['RR'] > threshold['RR']) & 
                  (ana['u'] < threshold['U10']) & 
                  (~np.isnan(ana['sss_sat'])))
    y_C3 = y[cond['C3']]
    
    # Condition C4: Mixed layer depth (if available)
    if 'MLD' in ana:
        cond['C4'] = ((ana['MLD'] > -20) & 
                      (~np.isnan(ana['sss_sat'])))
        y_C4 = y[cond['C4']]
    
    # Condition C5: Low WOA23 standard deviation
    cond['C5'] = ((ana['woa23_std'] < 0.2) & 
                  (~np.isnan(ana['sss_sat'])))
    y_C5 = y[cond['C5']]
    
    # Condition C6: High WOA23 standard deviation
    cond['C6'] = ((ana['woa23_std'] > 0.2) & 
                  (~np.isnan(ana['sss_sat'])))
    y_C6 = y[cond['C6']]
    
    # Distance to coast conditions
    cond['C7a'] = ((ana['dist2coast'] <= 150) & (~np.isnan(ana['sss_sat'])))
    cond['C7b'] = ((ana['dist2coast'] <= 800) & (ana['dist2coast'] > 150) & (~np.isnan(ana['sss_sat'])))
    cond['C7c'] = ((ana['dist2coast'] > 800) & (~np.isnan(ana['sss_sat'])))
    
    # SST conditions  
    cond['C8a'] = ((ana['sst'] <= 5) & (~np.isnan(ana['sss_sat'])))
    cond['C8b'] = ((ana['sst'] <= 15) & (ana['sst'] > 5) & (~np.isnan(ana['sss_sat'])))
    cond['C8c'] = ((ana['sst'] > 15) & (~np.isnan(ana['sss_sat'])))
    
    # SSS conditions
    cond['C9a'] = ((ana['sss'] <= 33) & (~np.isnan(ana['sss_sat'])))
    cond['C9b'] = ((ana['sss'] <= 37) & (ana['sss'] > 33) & (~np.isnan(ana['sss_sat'])))
    cond['C9c'] = ((ana['sss'] > 37) & (~np.isnan(ana['sss_sat'])))
    
    if opt_plot:
        # Plotting section
        clim = [-1.5, 1.5]
        
        # Create grid for mapping (simplified version)
        xxvec3 = np.arange(lon_int[0], lon_int[1], res)
        yyvec3 = np.arange(lat_int[0], lat_int[1], res)
        LON3, LAT3 = np.meshgrid(xxvec3, yyvec3)
        
        # Plot conditions 1-6 and subconditions
        conditions_to_plot = ['C1', 'C2', 'C3', 'C5', 'C6']
        if 'MLD' in ana:
            conditions_to_plot.append('C4')
        
        for condition in conditions_to_plot:
            # Grid the data (simplified - would need actual gridding function)
            Time_MEAN = ffgridrms(ana['lon'][cond[condition]], 
                                       ana['lat'][cond[condition]], 
                                       y[cond[condition]], 
                                       res, res, lon_int[0], lat_int[0], 
                                       lon_int[1], lat_int[1])
            
            # Create map (would need actual mapping function)
            figname = f"{pathfig}{fign}_Time-Mean-SSS-{sat_name}-minus-INSITU-{condition}"
            
            result1 = make_griddata_stat(
                x=ana['lon'][cond[condition]], y=ana['lat'][cond[condition]], z=y[cond[condition]]
                space_res='One_degree',
                opt_region='GLOBAL',
                opt_variable='SSS',
                weight=1,
                opt_plot=0,
                use_scipy=True  # Force scipy method for large dataset
            )
            
            make_map(result1['LON'], result1['LAT'], result1['MEAN'], clim, figname, 
                          f'Temporal mean of ({sat_name} - In situ) for {condition} in {res_str}°×{res_str}° boxes over {yearmin}-{yearmax}')
            
            # Create histogram
            figname = f"{pathfig}{fign}_Histogram-SSS-{sat_name}-minus-INSITU-{condition}"
            y_cond = y[cond[condition]]
            make_hist(y_cond[~np.isnan(y_cond)], -1.5, 1.5, 0.1, figname,
                           f'{sat_name} - {insitu_database} ({condition})')
    
    # Create CSV output directory
    csv_dir = Path(pathfig).parent / 'csvfile'
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate and write statistics
    # SAT - insitu
    if 'MLD' in ana:
        stat = calcul_stat(ana['sss_sat'], ana['sss'], cond, opt=True)
        write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_{insitu_product}.csv', stat, opt=True)
    else:
        stat = calcul_stat(ana['sss_sat'], ana['sss'], cond, opt=False)
        write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_{insitu_product}.csv', stat, opt=False)
    
    # SAT - ISAS
    isas = ana['isas_sss'].copy()
    isas[ana['isas_sss_pctvar'] > 80] = np.nan
    
    if 'MLD' in ana:
        stat = calcul_stat(ana['sss_sat'], isas, cond, opt=True)
        write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_ISAS.csv', stat, opt=True)
    else:
        stat = calcul_stat(ana['sss_sat'], isas, cond, opt=False)
        write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_ISAS.csv', stat, opt=False)
    
    # Argo DM statistics if applicable
    if insitu_product == 'argo':
        if 'MLD' in ana:
            stat = calcul_stat_argo_dm(ana, ana['sss_sat'], ana['sss'], cond)
            write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_{insitu_product}_DM.csv', stat, opt=True)
            
            stat = calcul_stat_argo_dm(ana, ana['sss_sat'], isas, cond)
            write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_ISAS_DM.csv', stat, opt=True)
        else:
            stat = calcul_stat_argo_dm(ana, ana['sss_sat'], ana['sss'], cond)
            write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_{insitu_product}_DM.csv', stat, opt=False)
            
            stat = calcul_stat_argo_dm(ana, ana['sss_sat'], isas, cond)
            write_csv(csv_dir / f'stats_{regindbasinmask.id}_{sat_product}_ISAS_DM.csv', stat, opt=False)
    
    return cond

def write_csv(filename, stat, opt=False):
    """Write statistics to CSV file"""
    
    with open(filename, 'w') as fid:
        # Header
        fid.write('cond,Nb,Median,Mean,STD,RMS,IQR,R2,RSTD\n')
        
        # All data
        fid.write(f'all,{stat["SSS_nb"]:.0f},{stat["SSSanomedian"]:.2f},{stat["SSSanomean"]:.2f},'
                 f'{stat["SSSanostd"]:.2f},{stat["SSSanorms"]:.2f},{stat["SSSanoiqr"]:.2f},'
                 f'{stat["SSSanor2"]:.3f},{stat["SSSanostd1"]:.2f}\n')
        
        # Conditions
        conditions = ['C1', 'C2', 'C3']
        if opt:
            conditions.append('C4')
        conditions.extend(['C5', 'C6', 'C7a', 'C7b', 'C7c', 'C8a', 'C8b', 'C8c', 'C9a', 'C9b', 'C9c'])
        
        for cond in conditions:
            fid.write(f'{cond},{stat[f"SSS_nb_{cond}"]:.0f},{stat[f"SSSanomedian_{cond}"]:.2f},'
                     f'{stat[f"SSSanomean_{cond}"]:.2f},{stat[f"SSSanostd_{cond}"]:.2f},'
                     f'{stat[f"SSSanorms_{cond}"]:.2f},{stat[f"SSSanoiqr_{cond}"]:.2f},'
                     f'{stat[f"SSSanor2_{cond}"]:.3f},{stat[f"SSSanostd1_{cond}"]:.2f}\n')

def calcul_stat(sat, insitu, cond, opt=False):
    """Calculate statistics for all conditions"""
    
    y = sat - insitu
    
    stat = {}
    
    # Overall statistics
    stat['SSS_nb'] = np.sum(~np.isnan(y))
    stat['SSSanomean'] = np.round(np.nanmean(y), 2)
    stat['SSSanomedian'] = np.round(np.nanmedian(y), 2)
    stat['SSSanostd'] = np.nanstd(y)
    stat['SSSanorms'] = np.sqrt(np.nanmean(y * np.conj(y)))
    stat['SSSanoiqr'] = np.percentile(y[~np.isnan(y)], 75) - np.percentile(y[~np.isnan(y)], 25)
    stat['SSSanostd1'] = std_robust(y)
    stat['SSSanor2'] = r_square(sat, insitu)
    
    # Statistics for each condition
    conditions = ['C1', 'C2', 'C3']
    if opt:
        conditions.append('C4')
    conditions.extend(['C5', 'C6', 'C7a', 'C7b', 'C7c', 'C8a', 'C8b', 'C8c', 'C9a', 'C9b', 'C9c'])
    
    for condition in conditions:
        if condition in cond:
            stat = calc_stat(stat, sat[cond[condition]], insitu[cond[condition]], f'_{condition}')
    
    return stat

def calcul_stat_argo_dm(ana, sat, insitu, cond, opt=True):
    """Calculate statistics for Argo delayed mode data"""
    
    y = sat - insitu
    dm_mask = ana['dm']
    
    stat = {}
    
    # Overall statistics for DM data
    stat['SSS_nb'] = np.sum(~np.isnan(y) & dm_mask)
    stat['SSSanomean'] = np.round(np.nanmean(y[dm_mask]), 2)
    stat['SSSanomedian'] = np.round(np.nanmedian(y[dm_mask]), 2)
    stat['SSSanostd'] = np.nanstd(y[dm_mask])
    stat['SSSanorms'] = np.sqrt(np.nanmean((y[dm_mask]) * np.conj(y[dm_mask])))
    stat['SSSanoiqr'] = np.percentile(y[dm_mask & ~np.isnan(y)], 75) - np.percentile(y[dm_mask & ~np.isnan(y)], 25)
    stat['SSSanostd1'] = std_robust(y[dm_mask])
    stat['SSSanor2'] = r_square(sat[dm_mask], insitu[dm_mask])
    
    # Statistics for each condition with DM mask
    conditions = ['C1', 'C2', 'C3']
    if opt:
        conditions.append('C4')
    conditions.extend(['C5', 'C6', 'C7a', 'C7b', 'C7c', 'C8a', 'C8b', 'C8c', 'C9a', 'C9b', 'C9c'])
    
    for condition in conditions:
        if condition in cond:
            combined_mask = cond[condition] & dm_mask
            stat = calc_stat(stat, sat[combined_mask], insitu[combined_mask], f'_{condition}')
    
    return stat

def calc_stat(stat, sat, insitu, suffix):
    """Calculate statistics for a specific condition"""
    
    y = sat - insitu
    
    stat[f'SSS_nb{suffix}'] = np.sum(~np.isnan(y))
    stat[f'SSSanomean{suffix}'] = np.round(np.nanmean(y), 2)
    stat[f'SSSanomedian{suffix}'] = np.round(np.nanmedian(y), 2)
    stat[f'SSSanostd{suffix}'] = np.nanstd(y)
    stat[f'SSSanorms{suffix}'] = np.sqrt(np.nanmean(y * np.conj(y)))
    if len(y[~np.isnan(y)]) > 0:
        stat[f'SSSanoiqr{suffix}'] = np.percentile(y[~np.isnan(y)], 75) - np.percentile(y[~np.isnan(y)], 25)
    else:
        stat[f'SSSanoiqr{suffix}'] = np.nan
    stat[f'SSSanostd1{suffix}'] = std_robust(y)
    stat[f'SSSanor2{suffix}'] = r_square(sat, insitu)
    
    return stat

def std_robust(x):
    """Robust standard deviation calculation"""
    x_clean = x[~np.isnan(x)]
    if len(x_clean) == 0:
        return np.nan
    return 1.4826 * np.nanmedian(np.abs(x_clean - np.nanmedian(x_clean)))

def r_square(x, y):
    """Calculate R-squared"""
    mask = ~np.isnan(x) & ~np.isnan(y)
    if np.sum(mask) < 2:
        return np.nan
    return np.corrcoef(x[mask], y[mask])[0, 1] ** 2

def ffgridrms(lon, lat, data, dx, dy, lon_min, lat_min, lon_max, lat_max):
    """Simplified gridding function - placeholder for actual implementation"""
    # This is a simplified version - you would need to implement actual gridding
    # based on your specific requirements
    nx = int((lon_max - lon_min) / dx)
    ny = int((lat_max - lat_min) / dy)
    return np.random.randn(ny, nx)  # Placeholder

def make_map(LON, LAT, data, clim, figname, title):
    """Simplified mapping function - placeholder for actual implementation"""
    plt.figure(figsize=(10, 8))
    plt.contourf(LON, LAT, data, levels=20, cmap='RdBu_r')
    plt.colorbar(label='SSS Difference')
    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(f'{figname}.png', dpi=300, bbox_inches='tight')
    plt.close()

def make_hist(data, xmin, xmax, bin_width, figname, title):
    """Create histogram"""
    bins = np.arange(xmin, xmax + bin_width, bin_width)
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
    plt.title(title)
    plt.xlabel('SSS Difference')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{figname}.png', dpi=300, bbox_inches='tight')
    plt.close()


    
    # Mock parameters
    class RegionMask:
        def __init__(self):
            self.id = 'GO'
    
    regindbasinmask = RegionMask()
    threshold = {'RR': 1.0, 'U10': 3.0}
    
    # Run analysis on all data
    conditions = plot_conditional_analyses(
        mdb=mdb,
        pathfig='output/figures/',
        regindbasinmask=regindbasinmask,
        sat_product='smos-l3-catds-cpdc-v335-1m-25km',
        insitu_product='metof',
        insitu_database='EN4',
        threshold=threshold,
        res=2.0,
        lat_int=[-90, 90],
        lon_int=[-180, 180],
        h1=None,
        opt_plot=True
    )
    
    print("Python translation of plot_conditional_analyses completed!")
    print("This is a framework - you'll need to:")
    print("1. Implement proper gridding functions")
    print("2. Implement proper mapping functions")
    print("3. Adapt data structures to your specific format")
    print("4. Add error handling and validation")