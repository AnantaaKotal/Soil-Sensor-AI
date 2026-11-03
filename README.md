# SoilSense — AI-Ready Sensor Data Workbench

An open research prototype for sensor quality-control screening and missing-data reconstruction using a pretrained masked time-series Transformer.

## What is actually available

- **Fault screening:** residual-based scores comparing observed sensor readings with leave-one-sensor-out Transformer predictions.
- **Imputation:** cross-sensor Transformer reconstruction of eligible missing cells, preserving originals and adding `__was_imputed` flags.
- **CSV upload, interactive plots, downloadable CSV results.**
- **GitHub Pages landing page** in `docs/`.

**Important scope limits:** This is the *preliminary Transformer model*, **not** a graph Transformer or a diffusion imputer. The available materials did not include runnable trained diffusion code/weights. Model weights were trained on a **single orchard dataset**, so performance for other sites, units or sensor configurations is unknown. Output alerts are **not verified faults**. The published synthetic-fault benchmark does not establish field diagnostic accuracy.

## Deployment (GitHub website + Python app)

GitHub Pages hosts the front-end landing page; GitHub Pages **cannot execute Python**. Deploy the actual app using [Streamlit Community Cloud](https://share.streamlit.io/), which can deploy Python apps directly from a GitHub repo.

1. Create a **public GitHub repository**, e.g. `soil-sense`, and upload **all files** from this folder to its root. Keep `models/soil_transformer.pt` (pretrained weights) and `models/soil_fault_metrics.json`.
2. In the repository, go to **Settings → Pages → Build and deployment**, set **Deploy from a branch**, choose `main` and `/docs`. The public project page will usually be `https://YOUR-USERNAME.github.io/soil-sense/`.
3. Go to [share.streamlit.io](https://share.streamlit.io/), connect your GitHub account and deploy from your repository, branch `main`, main file **`app.py`**.
4. Copy the Streamlit app URL, then edit `docs/index.html` and replace `const APP_URL = "";` with `const APP_URL = "https://YOUR-APP.streamlit.app";`. Commit the edit. The **Launch data workbench** button will then open the working app.
5. Test the included `examples/sample_sensor_data.csv` with each tool. Share the GitHub Pages URL as the project homepage.

Alternatively, share the Streamlit app link directly and skip the Pages landing page.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## CSV schema

A timestamp column (e.g. `timestamp`) plus these **exactly named channels**:

`CO2_30`, `CO2_60`, `O2_30`, `O2_60`, `VWC_30`, `VWC_60`, `SoilT_30`, `SoilT_60`, `Air_P`, `ESFD_Flux`.

Example data: `examples/sample_sensor_data.csv`.

- CSV timestamps must be parseable and unique (prefer ISO 8601). Input timestamps are sorted; timezone-naive timestamps are treated as UTC.
- Readings must be in the same physical units and instrumentation conventions as the original orchard data. Unknown units may make predictions meaningless.
- Predictions are only attempted for *complete 9-row windows of 5-minute consecutive timestamps* (40 minutes centered on the predicted reading).
- At least **three other sensors** must have an observed reading at the target time; otherwise the model returns no estimate. If an entire row is absent, or a large timestamp gap exists, this version will not recreate the missing timestamps.
- All other original CSV columns are preserved in downloads.
- Uploaded file limit: 15 MB in this prototype.

### Fault-screening output

For each channel: `__expected`, `__score` (absolute deviation in training-standard-deviation units), and `__candidate_fault` flag. Also includes a `candidate_fault_count` column.

Two threshold modes:

- **Dataset-relative (default):** marks high residuals relative to the uploaded sample, per sensor. This is a ranking/triage rule, not a validated probability or calibrated false-positive rate.
- **Original orchard calibration:** compares residuals with thresholds from the pilot orchard validation period. **High false-alert rates were observed under distribution shift.** Do not assume 5% alerts in new files.

### Imputation output

Preserves observed readings, replaces some missing cells with masked Transformer cross-sensor predictions, and adds `__was_imputed` boolean columns. Prediction reliability on completely missing channels or new environmental events has not been validated. This is not adversarial diffusion imputation.

## Extending the project

For the future trained diffusion model, add a separate `diffusion_engine.py` exposing a `reconstruct(df, metadata)` function and integrate it as an additional Streamlit tool. Do not label Transformer predictions as diffusion outputs.

For broader deployment, add schema mapping, documented physical units, validation against expert-reviewed faults, event labels, calibrated uncertainty, upload size/rate limits and a data handling/privacy policy.

## License and dataset provenance

Before publishing, confirm ownership and redistribution permission for model weights, example dataset, and any contributed model or dataset. No license is assigned automatically. The sample CSV contains a short excerpt of the provided orchard data and should be removed or replaced if redistribution is not permitted.
