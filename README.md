# Jupyter Project Template

A template for organizing Jupyter-based projects.

## Structure

- `notebooks/`: analysis and visualization
- `scripts/`: utility functions or data processing
- `data/`: raw or downloaded data (not tracked in Git)
- `output/`: figures and netcdf files produced by the notebooks 

## Environment

```bash
conda env create -f env.yml
conda activate your-env-name
```
## Download the regional masks

wget -r -np -nH --cut-dirs=1 -A "*.nc" -P data/ https://pimep.ifremer.fr/diffusion/mask/