"""Central configuration for the IR2RGB project."""

# ---- Earth Engine / data acquisition ----
THERMAL_BAND = "ST_B10"          # Landsat 8/9 Collection 2 Level-2 surface temperature band
RGB_BANDS = ["SR_B4", "SR_B3", "SR_B2"]  # Red, Green, Blue surface reflectance bands
CLOUD_COVER_MAX = 10             # percent

# Scenes to pull (lon, lat) centers -- pick diverse terrain types
SCENE_CENTERS = [
    {"name": "urban_1",        "lon": -74.0060, "lat": 40.7128},   # NYC
    {"name": "urban_2",        "lon": -122.4194, "lat": 37.7749},  # SF
    {"name": "coastal_1",      "lon": -80.1918, "lat": 25.7617},   # Miami
    {"name": "coastal_2",      "lon": 151.2093, "lat": -33.8688},  # Sydney
    {"name": "agricultural_1", "lon": -97.7431, "lat": 41.5868},   # Nebraska farmland
    {"name": "agricultural_2", "lon": 2.3522,   "lat": 48.8566},   # Paris outskirts/farmland
    {"name": "desert_1",       "lon": -111.8910, "lat": 33.4484},  # Phoenix
    {"name": "desert_2",       "lon": 31.2357,  "lat": 30.0444},   # Cairo
]

DATE_RANGE = ("2023-01-01", "2024-01-01")

# ---- Tiling ----
PATCH_SIZE = 256
PATCH_STRIDE = 256  # non-overlapping tiles; set smaller for overlap/augmentation

# ---- Paths ----
RAW_DIR = "data_raw"          # downloaded full scenes (GeoTIFF)
TILES_DIR = "data_tiles"      # tiled patches (.npy)
SPLIT_FILE = "splits.json"    # which scenes go to train/val/test

TRAIN_SCENES_FRAC = 0.7
VAL_SCENES_FRAC = 0.15
# remaining fraction goes to test

# ---- Training ----
BATCH_SIZE = 16
NUM_EPOCHS = 60
LR = 2e-4
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_EVERY_STEPS = 200

# Loss weights (full model)
LAMBDA_L1 = 1.0
LAMBDA_PERCEPTUAL = 0.1
LAMBDA_STRUCTURAL = 0.5
