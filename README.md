# Jupyter Notebooks for Pi-MEP Satellite & In-situ Data Analysis

Please visit first the [Pi-MEP website](https://www.salinity-pimep.org/)

## Structure

- `notebooks/`: analysis and visualization
- `scripts/`: utility functions or data processing
- `data/`: raw or downloaded data (not tracked in Git)
- `output/`: figures and netcdf files produced by the notebooks (not tracked in Git)

## Environment

```bash
conda env create -f env.yml
conda activate pimep
```
## Download the regional masks

wget -r -np -nH --cut-dirs=1 -A "*.nc" -P data/ https://pimep.ifremer.fr/diffusion/mask/

## Download all the "reduced" Match-up NetCDF files (33GB)

wget -r -np -nH --cut-dirs=1 -A "*.nc" -P data/ https://pimep.ifremer.fr/diffusion/data_concat/

