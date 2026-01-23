# Installation

## Requirements

GWBASE requires Python 3.8 or higher. The following packages are required:

### Core Dependencies

| Package | Purpose |
|---------|---------|
| pandas | Data manipulation and analysis |
| numpy | Numerical computing |
| geopandas | Geospatial data handling |
| networkx | Stream network graph analysis |
| scipy | Scientific computing (interpolation, statistics) |
| scikit-learn | Machine learning (mutual information) |
| tqdm | Progress bars |

### Visualization Dependencies

| Package | Purpose |
|---------|---------|
| matplotlib | Plotting |
| seaborn | Statistical visualization |

## Installation Methods

### Option 1: Install from Source

Clone the repository and install in development mode:

```bash
git clone https://github.com/your-org/gwbase.git
cd gwbase
pip install -e .
```

### Option 2: Install Dependencies Only

If you prefer to use GWBASE without installing as a package, install the dependencies:

```bash
pip install pandas numpy geopandas networkx scipy scikit-learn tqdm matplotlib seaborn
```

Then add the project directory to your Python path:

```python
import sys
sys.path.insert(0, '/path/to/gwbase')
import gwbase
```

## Verifying Installation

After installation, verify that GWBASE loads correctly:

```python
import gwbase
print(f"GWBASE version: {gwbase.__version__}")
```

Expected output:
```
GWBASE version: 0.1.0
```

## Optional: Conda Environment

For reproducibility, create a dedicated conda environment:

```bash
conda create -n gwbase python=3.10
conda activate gwbase

# Install geospatial dependencies via conda (recommended)
conda install -c conda-forge geopandas networkx

# Install remaining dependencies via pip
pip install scipy scikit-learn tqdm matplotlib seaborn
```

## Troubleshooting

### GeoPandas Installation Issues

GeoPandas has dependencies (GDAL, GEOS, PROJ) that can be difficult to install via pip. If you encounter issues:

```bash
# macOS with Homebrew
brew install gdal geos proj
pip install geopandas

# Or use conda (recommended)
conda install -c conda-forge geopandas
```

### Import Errors

If you see import errors after installation, ensure:

1. Your Python environment is activated
2. The gwbase package directory is in your Python path
3. All dependencies are installed in the same environment

```python
# Check if gwbase is importable
try:
    import gwbase
    print("GWBASE imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
```
