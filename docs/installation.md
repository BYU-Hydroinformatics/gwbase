# Installation

## Requirements

- Python 3.8+
- Jupyter Notebook

## Setup

```bash
# Clone repository
git clone https://github.com/BYU-Hydroinformatics/gwbase.git
cd gwbase

# Create environment
conda create -n gwbase python=3.9
conda activate gwbase

# Install packages
pip install -r requirements.txt

# Test installation
jupyter notebook
```

## Data Structure

Place data files in:
```
data/raw/
├── groundwater/          # Well time series
├── hydrography/          # Shapefiles (basin, streams, etc.)
└── streamflow/           # USGS gage data
```

## Troubleshooting

- **GeoPandas errors**: `conda install -c conda-forge geopandas`
- **Missing shapefiles**: Check all .shp, .shx, .dbf, .prj files present
- **Memory issues**: Use data subsets for testing