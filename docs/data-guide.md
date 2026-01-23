# Data Guide

This guide describes how to organize input data for GWBASE and the required file formats.

## Directory Structure

GWBASE expects data organized in the following structure:

```
data/
├── raw/
│   ├── hydrography/
│   │   ├── streams.shp          # Stream network shapefile
│   │   ├── catchments.shp       # Catchment polygons shapefile
│   │   └── ...                  # Associated .dbf, .prj, .shx files
│   │
│   ├── groundwater/
│   │   ├── wells.csv            # Well locations (from USGS NWIS)
│   │   └── water_levels.csv     # Water table elevation time series
│   │
│   ├── streamflow/
│   │   ├── gage_info.csv        # Gage metadata
│   │   └── daily_discharge.csv  # Daily streamflow data
│   │
│   └── bfd/
│       └── bfd_classification.csv  # Baseflow-dominated period flags
│
├── processed/
│   └── (intermediate outputs)
│
└── features/
    └── (final analysis outputs)
```

---

## Required Input Files

### 1. Stream Network (streams.shp)

**Format:** ESRI Shapefile (or GeoPackage)

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `LINKNO` | Integer | Unique identifier for each stream segment |
| `DSLINKNO` | Integer | Downstream segment ID (0 or null for outlets) |
| `geometry` | LineString | Stream segment geometry |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Length` | Float | Segment length |
| `strmOrder` | Integer | Strahler stream order |

**Example:**
```
LINKNO    DSLINKNO    geometry
123456    123457      LINESTRING(...)
123457    123458      LINESTRING(...)
123458    0           LINESTRING(...)  # Outlet
```

**Notes:**
- LINKNO values must be unique
- DSLINKNO = 0 or null indicates network outlet
- Coordinate reference system (CRS) should be defined in .prj file

---

### 2. Catchment Polygons (catchments.shp)

**Format:** ESRI Shapefile (or GeoPackage)

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `linkno` or `LINKNO` | Integer | Catchment ID matching stream LINKNO |
| `geometry` | Polygon | Catchment boundary |

**Notes:**
- Each catchment should correspond to one stream segment
- CRS must match or be transformable to stream network CRS

---

### 3. Well Locations (wells.csv)

**Format:** CSV (downloaded from USGS NWIS or similar)

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Well_ID` | String/Integer | Unique well identifier (USGS site number) |
| `lat_dec` | Float | Latitude in decimal degrees |
| `long_dec` | Float | Longitude in decimal degrees |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Well_Name` | String | Well name or description |
| `GSE` | Float | Ground surface elevation (feet) |
| `AquiferID` | Integer | Aquifer identifier |
| `Aquifer_Name` | String | Aquifer name |
| `State` | String | State abbreviation |

**Example (from USGS NWIS):**
```csv
Well_ID,Well_Name,lat_dec,long_dec,GSE,AquiferID,Aquifer_Name,State
381033113480701,(C-30-18)25aad- 1,38.17579566,-113.8027496,7098.0,1,GSL Basin,UT
381037113474001,(C-30-17)30bab- 1,38.17630556,-113.7955,7193.0,1,GSL Basin,UT
381152113442801,(C-30-17)15cab- 1,38.1978333,-113.7411667,6550.0,1,GSL Basin,UT
382113113435401,(C-28-17)22dda- 1,38.35357117,-113.732473,5775.0,1,GSL Basin,UT
```

**Notes:**
- Well_ID should match the IDs used in the water level time series file
- Coordinates must be in decimal degrees (WGS84)
- GSE (ground surface elevation) is used for context but not required
- GWBASE automatically creates point geometries from lat/lon coordinates

---

### 4. Water Level Time Series (water_levels.csv)

**Format:** CSV

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `well_id` | String/Integer | Well identifier (must match wells.csv) |
| `date` | Date | Measurement date (YYYY-MM-DD) |
| `wte` | Float | Water table elevation (feet above datum) |

**Example:**
```csv
well_id,date,wte
381033113480701,2000-01-15,7045.5
381033113480701,2000-03-22,7043.2
381033113480701,2000-06-10,7038.8
381037113474001,2000-01-20,7152.1
381037113474001,2000-04-05,7149.3
```

**Notes:**
- Dates should be in ISO format (YYYY-MM-DD)
- WTE should be in consistent units (typically feet above mean sea level)
- Missing values can be empty or NaN
- Data does not need to be at regular intervals (PCHIP handles irregular sampling)
- Well_ID must match the Well_ID column in wells.csv

---

### 5. Gage Information (gage_info.csv)

**Format:** CSV

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `id` | String/Integer | Unique gage identifier (e.g., USGS site number) |
| `latitude` | Float | Gage latitude (decimal degrees) |
| `longitude` | Float | Gage longitude (decimal degrees) |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `name` | String | Gage name/description |
| `COMID_v2` | Integer | NHDPlus COMID for reach matching |
| `drainage_area` | Float | Contributing drainage area |

**Example:**
```csv
id,name,latitude,longitude,COMID_v2
10126000,BEAR RIVER NEAR CORINNE UT,41.5698,-112.0842,23456789
10128500,WEBER RIVER NEAR PLAIN CITY UT,41.3012,-112.0876,23456790
```

---

### 6. Daily Streamflow Data (daily_discharge.csv)

**Format:** CSV

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `gage_id` | String/Integer | Gage identifier (must match gage_info.csv) |
| `date` | Date | Date (YYYY-MM-DD) |
| `q` | Float | Daily mean discharge (cubic feet per second) |

**Example:**
```csv
gage_id,date,q
10126000,1990-01-01,245.0
10126000,1990-01-02,238.0
10126000,1990-01-03,242.0
10128500,1990-01-01,189.0
```

**Notes:**
- One row per gage per day
- Missing days can be omitted (will not pair with well data)
- Q values should be positive; negative or zero values may indicate data issues

---

### 7. BFD Classification (bfd_classification.csv)

**Format:** CSV

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `gage_id` | String/Integer | Gage identifier |
| `date` | Date | Date (YYYY-MM-DD) |
| `bfd` | Integer | Baseflow-dominated flag (1=BFD, 0=not BFD) |

**Example:**
```csv
gage_id,date,bfd
10126000,1990-01-01,1
10126000,1990-01-02,1
10126000,1990-01-03,0
10126000,1990-01-04,0
```

**Notes:**
- BFD classification typically from a separate ML model
- Only dates with BFD=1 are used for baseline calculation
- Records with BFD=0 can still be included in analysis

---

### 8. Reach Elevations (reach_elevations.csv)

**Format:** CSV

**Required Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Reach_ID` | Integer | Stream segment ID (matches LINKNO) |
| `Avg_GSE` | Float | Average ground surface elevation (meters) |

**Optional Columns:**

| Column | Type | Description |
|--------|------|-------------|
| `Min_GSE` | Float | Minimum elevation along reach |
| `Max_GSE` | Float | Maximum elevation along reach |
| `Latitude` | Float | Reach centroid latitude |
| `Longitude` | Float | Reach centroid longitude |

**Example:**
```csv
Reach_ID,Avg_GSE,Latitude,Longitude
123456,1523.4,41.234,-111.876
123457,1518.2,41.241,-111.869
```

**Notes:**
- Elevations typically extracted from DEM
- Units should be meters for consistency with elevation filtering

---

## Data Preparation Checklist

Before running GWBASE, verify:

### Spatial Data
- [ ] Stream shapefile has LINKNO and DSLINKNO columns
- [ ] Catchment shapefile has linkno column matching streams
- [ ] All shapefiles have defined CRS (.prj file)

### Well Data (CSV)
- [ ] wells.csv has Well_ID, lat_dec, long_dec columns
- [ ] Coordinates are in decimal degrees (WGS84)
- [ ] Well_ID values match between wells.csv and water_levels.csv
- [ ] water_levels.csv has well_id, date, wte columns

### Temporal Data
- [ ] Water level dates are in YYYY-MM-DD format
- [ ] Streamflow dates are in YYYY-MM-DD format
- [ ] Date ranges overlap between wells and streamflow
- [ ] BFD classification covers streamflow date range

### ID Matching
- [ ] Well_ID in water_levels.csv matches Well_ID in wells.csv
- [ ] gage_id in streamflow matches id in gage_info.csv
- [ ] gage_id in BFD classification matches gage_id in streamflow
- [ ] Reach_ID in elevations matches LINKNO in streams

### Units
- [ ] WTE is in feet (or document if meters)
- [ ] Streamflow Q is in cfs (cubic feet per second)
- [ ] Reach elevations are in meters
- [ ] Coordinates are in decimal degrees (geographic)

---

## Common Data Issues

### Issue: Wells not matching to gages

**Symptoms:** Few or no wells assigned to gages in Step 2

**Causes:**
- Wells outside catchment boundaries
- CRS mismatch between wells and catchments
- well_id column naming (case sensitivity)

**Solutions:**
- Verify coordinate ranges overlap
- Check column names: `print(wells_gdf.columns)`
- Visualize data overlap in GIS

### Issue: No paired records after Step 7

**Symptoms:** Empty or very small paired dataset

**Causes:**
- Date ranges don't overlap between wells and streamflow
- gage_id mismatch between datasets
- All dates have BFD=0

**Solutions:**
- Check date ranges: `well_ts['date'].min()`, `streamflow['date'].min()`
- Verify ID matching: `set(well_data['gage_id']) & set(streamflow['gage_id'])`
- Check BFD distribution: `bfd['bfd'].value_counts()`

### Issue: Elevation filter removes all data

**Symptoms:** No records retained after Step 6

**Causes:**
- WTE much lower than stream elevation
- Unit mismatch (WTE in feet, elevation in meters)
- Buffer too restrictive

**Solutions:**
- Check units and convert if needed
- Increase buffer distance: `filter_by_elevation(..., distance_buffer_meters=50)`
- Use sensitivity analysis: `analyze_elevation_sensitivity()`

---

## Example Data Loading

```python
import gwbase
import os

# Set paths
data_dir = 'data/raw'

# Load hydrography
stream_gdf = gwbase.load_hydrography_data(
    os.path.join(data_dir, 'hydrography/streams.shp')
)
catchment_gdf = gwbase.load_hydrography_data(
    os.path.join(data_dir, 'hydrography/catchments.shp')
)

# Load groundwater data from CSV files
wells_gdf, well_ts, well_info = gwbase.load_groundwater_data(
    well_locations_path=os.path.join(data_dir, 'groundwater/wells.csv'),
    timeseries_path=os.path.join(data_dir, 'groundwater/water_levels.csv')
)

# Or load well locations separately
wells_gdf = gwbase.load_well_locations(
    os.path.join(data_dir, 'groundwater/wells.csv')
)

# Load streamflow
gage_info = gwbase.load_gage_info(
    os.path.join(data_dir, 'streamflow/gage_info.csv')
)
streamflow = gwbase.load_streamflow_data(
    os.path.join(data_dir, 'streamflow/daily_discharge.csv')
)

# Load classifications
bfd_class = gwbase.load_baseflow_classification(
    os.path.join(data_dir, 'bfd/bfd_classification.csv')
)
reach_elev = gwbase.load_reach_elevations(
    os.path.join(data_dir, 'processed/reach_elevations.csv')
)
```

---

## Downloading Data from USGS NWIS

Well data can be downloaded from the USGS National Water Information System (NWIS):

1. Go to https://waterdata.usgs.gov/nwis/gw
2. Select your geographic area
3. Choose "Site Information" to get well locations
4. Choose "Groundwater levels" to get water level time series
5. Export as tab-delimited or CSV format
6. Rename columns to match expected format (Well_ID, lat_dec, long_dec, date, wte)

**Column Mapping from NWIS:**
| NWIS Column | GWBASE Column |
|-------------|---------------|
| site_no | Well_ID |
| dec_lat_va | lat_dec |
| dec_long_va | long_dec |
| alt_va | GSE |
| lev_dt | date |
| lev_va | wte |
