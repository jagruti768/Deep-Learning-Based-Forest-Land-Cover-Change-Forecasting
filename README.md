# Forest Land-Cover Change Forecasting with Deep Learning

Deep-learning system that learns how land cover changed over past decades from classified satellite maps, then forecasts the next map. It reports forest loss and urban expansion as maps, tables and statistics.

**Study areas**

| Area | Status | Years | Classes |
|---|---|---|---|
| Sanjay Gandhi National Park (SGNP), Mumbai | Done | 1993, 2003, 2013, 2023 | Vegetation, Urban, Coastal features, Water, Barren |
| Molai Forest, Assam | In progress | 2000, 2025 | Water, Bare soil/urban, Dry grass/shrubs, Cropland/grassland, Forest/healthy crops |
| Jim Corbett National Park | Planned | - | To be defined |

Each area has its own class list, so everything is driven by one settings file (`src/config.py`).

---

## 1. What this project does

1. Reads classified land-cover GeoTIFF maps (30 m pixels) for several years.
2. Fixes class-number mismatches between files (see "Important data lesson").
3. Adds an auxiliary layer (terrain slope).
4. Trains models to predict the next map from the earlier maps.
5. Scores the predictions against the real map on spatial hold-out blocks.
6. Produces forecast maps, area tables and charts.

### 80 / 20 weighting
Land-cover history gets **80 %** weight and auxiliary inputs (slope, later elevation) get **20 %**. This is a fixed design constraint in `models.py`: two branches (land cover, auxiliary), each normalised, then fused as `0.8 * land_cover + 0.2 * auxiliary`. It is not learned.

### What this project is NOT (yet)
- It does not segment raw Sentinel-2 or Landsat images. It works on already-classified maps supplied by the teacher.
- It does not detect forest fires, illegal mining or soil erosion directly. It tracks class changes (for example vegetation to urban, water reduction).
- Email alerts are skipped for now.

---

## 2. Project structure

```
lulc_forecast/
├── data/
│   ├── sgnp/            LULC_1993.tif, LULC_2003.tif, LULC_2013.tif, LULC_2023.tif, slope.tif
│   └── molai/           raw Molai files (see Molai section)
├── outputs/             generated results, one sub-folder per area
├── src/
│   ├── config.py                 
│   ├── data_prep.py              
│   ├── prepare_molai.py          
│   ├── inspect_tif.py            
│   ├── inspect_aux.py            
│   ├── splits.py                 
│   ├── features.py               
│   ├── metrics.py                
│   ├── plots.py                  
│   ├── run_baselines.py          
│   ├── models.py                 
│   ├── dl_utils.py               
│   ├── train.py                  train a model
│   ├── compare.py                
│   ├── report_model.py           
│   ├── make_report_tables.py     
│   ├── predict_future.py         
│   ├── export_tif.py             
├── requirements.txt
└── README.md
```

---

## 3. Setup (Windows, VS Code, CPU only)

Tested with Python 3.12.8 on a laptop with no NVIDIA GPU (about 8 GB RAM).

```bash
python -m venv venv
venv\Scripts\activate
pip install numpy matplotlib rasterio scikit-learn scipy tqdm torch
# for the dashboard
pip install streamlit folium streamlit-folium reportlab
```

If `import torch` fails with `WinError 126`, install the Microsoft Visual C++ Redistributable (x64) and restart.

---

## 4. How to run (SGNP)

```bash
python src/data_prep.py sgnp
python src/run_baselines.py
python src/train.py unet --epochs 15 --lc_weight 0.8
python src/compare.py
python src/report_model.py unet_w80
python src/make_report_tables.py sgnp unet_w80
python src/predict_future.py unet_w80
```

Outputs go to `outputs/sgnp/` (models, predicted maps, figures, CSV tables).

### Other useful options
```bash
python src/train.py unet --lc_weight 0.5          # change the 80/20 balance
python src/train.py unet --seed 1                 # another random seed
python src/train.py convlstm_unet                 # recurrent variant
python src/train.py pixel_lstm                    # per-pixel LSTM
```

---

## 5. Experimental design (SGNP)

- Inputs: 1993, 2003, 2013 (+ slope). Target: 2023.
- The hold-out test is spatial. The area is cut into blocks, with 60 % of blocks for training, 10 % for validation and 30 % for testing. This avoids the cheating that happens when neighbouring pixels land in both train and test.
- The best epoch is chosen on the validation blocks. All scores below are on **test blocks only** (849,129 pixels, about 764 km²).
- Training uses random 64×64 patches with flips and rotations. One "epoch" is 600 patches.

---

## 6. Results (SGNP, test blocks)

| Model | OA % | Kappa | Macro F1 | Change-only acc % | FoM % |
|---|---|---|---|---|---|
| U-Net (80/20, seed 0) | 75.96 | 0.669 | 0.765 | 38.62 | 32.60 |
| U-Net (100 % land cover, no slope) | 75.60 | 0.663 | 0.762 | 37.49 | 31.64 |
| ConvLSTM + U-Net (seed 0) | 75.64 | 0.664 | 0.763 | 36.01 | 30.81 |
| Random Forest | 74.91 | 0.653 | 0.750 | 33.59 | 28.74 |
| Per-pixel LSTM | 73.15 | 0.631 | 0.739 | 24.72 | 21.71 |
| Persistence (2023 = 2013) | 69.88 | 0.587 | 0.714 | 0 | 0 |

OA = overall accuracy, FoM = Figure of Merit (how well actual changes are predicted).

**Findings**
1. All deep models beat the "nothing changes" baseline (about +6 points OA for U-Net).
2. U-Net and ConvLSTM+U-Net are statistically tied across 3 seeds (U-Net mean OA 75.64, ConvLSTM+U-Net 75.45). U-Net is kept because it is faster, simpler and smaller.
3. Spatial context matters more than time order: the per-pixel LSTM is the weakest.
4. The 80/20 design works and does not hurt, but slope (5 coarse classes) adds no measurable accuracy. A real elevation model is still untested.
5. Predictions are smoother than the real speckled maps. Only about 38 % of truly changed pixels get the right class. Area totals are more reliable than exact pixel locations.

### SGNP forecast: 2033
The model learned a 10-year step, so inputs 2003, 2013, 2023 give 2033.

| Class | 2023 km² | 2033 km² (forecast) | Change |
|---|---|---|---|
| Vegetation | 675.6 | 575.4 | -14.8 % |
| Urban | 773.5 | 960.2 | +24.1 % |
| Coastal | 211.6 | 183.8 | -13.1 % |
| Water | 125.4 | 126.2 | +0.6 % |
| Barren | 999.9 | 940.3 | -6.0 % |

This is a projection with about 76 % held-out accuracy, not a fact. It assumes the past change pattern continues.

---

## 7. Important data lesson: class codes differ between files

The class numbers meant different things in different SGNP files (checked through the embedded colour tables). Everything is remapped to the standard codes of the 2023 file:

`0 Vegetation, 1 Urban, 2 Coastal features, 3 Water, 4 Barren`

Always inspect the colour table of any new file (`python src/inspect_tif.py`) before training.

---

## 8. Molai Forest (in progress)

- Years 2000 and 2025, forecast 2050 (one 25-year step).
- Only one input map exists, so LSTM and ConvLSTM do not apply. Compared: U-Net, persistence, Random Forest (CA-Markov planned).
- Data caveat: only about 24 % of pixels keep their class between 2000 and 2025. The 2025 map is a 17 January (dry season) picture and the season of the 2000 map is unknown. Part of the change is probably a season or classification effect. Area-level trends are more defensible than pixel-level 2050 maps.
- Slope is nearly flat (mean about 2°), so it is expected to add little.

```bash
python src/prepare_molai.py
# set STUDY_AREA = "molai" in src/config.py
python src/data_prep.py molai
python src/run_baselines.py
python src/train.py unet --patch 32 --epochs 80
python src/predict_future.py unet_w80
```

---


## 10. Known limitations

- Only 3 time steps and one training transition in SGNP.
- Forecasts assume the historical change rule continues. A single step is predicted, not recursive.
- Predictions are smoother than reality.
- Classification accuracy of the source maps (OA 95-98 %, kappa 0.93-0.975, from the source paper) is a different quantity from our forecast accuracy. Do not mix them.
---

## 11. Roadmap

- [x] SGNP data preparation, baselines, deep models, report tables
- [ ] Molai training and forecast
- [ ] Dashboard (Stages 1-3)
- [ ] Real DEM (SRTM) for elevation
- [ ] Recursive forecasting for arbitrary years
- [ ] Forest-loss risk map
- [ ] CA-Markov baseline
- [ ] Jim Corbett
- [ ] Optional: email alerts, raw satellite image segmentation

---
