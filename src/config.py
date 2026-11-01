# config.py
# config.py
# One place for ALL settings. For Molai or Jim Corbett you only add a new
# entry to AREAS and change STUDY_AREA. No other file needs changing.

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

STUDY_AREA = "molai"          # <- change this to switch forest

# ---- 80% / 20% weighting (used by the models in Phase 2) ----
LC_WEIGHT = 0.8              # land-cover (forest image history)
AUX_WEIGHT = 0.2             # auxiliary inputs (slope, elevation)

AREAS = {
    "sgnp": {
        "name": "Sanjay Gandhi National Park",
        "data_dir": ROOT / "data" / "sgnp",
        "lulc_pattern": "LULC_{year}.tif",      # file name template
        "years": [1993, 2003, 2013, 2023],      # last year = target
        "nodata": 15,                           # value used outside the park
        "crs": "EPSG:32643",                    # UTM zone 43N (Mumbai)
        "classes": {
            0: "Vegetation",
            1: "Urban",
            2: "Coastal features",
            3: "Water",
            4: "Barren",
        },
        "colors": {
            0: "#006400", 1: "#e41a1c", 2: "#7CFC00",
            3: "#00008B", 4: "#FFD700",
        },
        "remap": {
            1993: [0, 2, 1, 4, 3],
            2003: [1, 0, 3, 2, 4],
            2013: [1, 0, 3, 2, 4],
            2023: [0, 1, 2, 3, 4],   # already standard
        },
        # auxiliary rasters: name -> file. Add "elevation": "elevation.tif" later.
        "aux": {"slope": "slope.tif"},
    },
    # "molai": {...},  "corbett": {...}   <- copy the block above and edit
        "molai": {
        "name": "Molai Forest",
        "data_dir": ROOT / "data" / "molai",
        "lulc_pattern": "LULC_{year}.tif",
        "years": [2000, 2025],                  # one 25-year step -> forecast 2050
        "nodata": 15,
        "crs": "EPSG:32646",                    # UTM zone 46N (Assam)
        "classes": {
            0: "Water bodies",
            1: "Bare soil, sand, urban",
            2: "Dry grasslands, shrubs",
            3: "Croplands, grasslands",
            4: "Forests, healthy crops",
        },
        "colors": {
            0: "#005CE6", 1: "#FF0000", 2: "#73B273",
            3: "#70A800", 4: "#267300",
        },
        "aux": {"slope": "Molai_Slope_Final.tif"},   # elevation: add later if you get a DEM
        "forest_classes": [4],
        "best_model": "unet_w80",
        "block": 32,                            # small area -> small blocks (pixels)
    },
}


def get_area(name=None):
    """Return the settings dict for the chosen forest."""
    cfg = AREAS[name or STUDY_AREA]
    cfg["output_dir"] = ROOT / "outputs" / (name or STUDY_AREA)
    cfg["output_dir"].mkdir(parents=True, exist_ok=True)
    return cfg