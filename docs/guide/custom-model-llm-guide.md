# Add a custom model (LLM quick guide)

This is a high-level workflow for adding a new ML model to the miner (for example, RandomForest, XGBoost, or your own regressor).

Goal: give an LLM the right project context, let it implement/tune the model, then plug it back into miner preflight so users can select it at runtime.

---

## What to give the LLM

From repo root, provide these files first:

- `miner_model_energy/pipeline.py` (main train/predict/save/load routing)
- `miner_model_energy/ml_config.py` (YAML config normalization/validation)
- `neurons/miner.py` (interactive model selection before miner startup)
- `model_params.yaml` (model hyperparameters in YAML)
- `miner_model_energy/models_cart.py` (simple reference model structure)
- `miner_model_energy/models_linear.py` (another minimal reference pattern)

If your new model is sequence-based (like RNN/LSTM style), also provide:

- `miner_model_energy/models_lstm.py`
- `miner_model_energy/models_rnn.py`

---

## LLM prompt template (copy/paste)

Use this prompt with the files above attached:

```text
Add a new model type named "<your_model>" to this miner project.

Requirements:
1) Create miner_model_energy/models_<your_model>.py using the same bundle/train/predict/save/load pattern as existing models.
2) Wire it into miner_model_energy/pipeline.py everywhere model_type is routed:
   - train_model
   - predict_single_test_row
   - predict_for_timestamp (if applicable)
   - persist_training_result
   - load_training_bundle_from_manifest
3) Add a models.<your_model> block with defaults in model_params.yaml.
4) Update miner_model_energy/ml_config.py to normalize/validate the new model config.
5) Update neurons/miner.py interactive prompt so users can choose "<your_model>" in preflight.

Constraints:
- Keep behavior backward-compatible for existing model types.
- Follow existing naming and style conventions in this repo.
```

---

## Practical notes

- You can use `models_cart.py` as the easiest starting template for non-sequence regressors.
- Renaming classes alone is not enough; routing in `pipeline.py` and miner preflight selection must be updated.
- Keep model artifact naming explicit (for example `model_<your_model>.joblib`) to avoid collisions.
- Shift-based train/prod alignment knobs live in `model_params.yaml` under `data`:
- `train_feature_time_shift_min`: shifts raw weather columns forward in training and, when enabled, derives time/cyclical features from the same shifted timestamp (currently applied for `supabase_storage` source).
  - `train_disable_horizon_label_shift_when_feature_shifted`: when enabled with shift mode, training uses same-row `Total Load` as label instead of creating `Total Load (horizon)`.
- This shift mode is a pragmatic approximation for matching production-style forecast inputs.

---

## User plugin workflow (external train in Colab / VM)

Miners can export a **plugin folder** under `persistence.artifact_dir` (see `model_params.yaml`) with everything needed to train a custom **regression** model offline, then deploy the saved weights on the next miner start.

### Preflight menu

At startup, choose **[3] Custom model plugin**. You can either:

1. **Create a new folder** — the miner writes:
   - `training_dataset_full.csv` — engineered training rows (same features as `prepare_training_data` in `pipeline.py`)
   - `feature_contract.json` — canonical ordered feature list, target column name, horizon/shift flags, validation split hint
   - `plugin_metadata.json` — schema version, paths, optional `keras_sequence_n_steps` after deploy
   - `custom_train_colab.ipynb` — starter notebook (sklearn + Keras examples; **do not** change feature columns vs the contract)

2. **Use an existing folder** — after you upload `*.joblib` or `*.keras` (or a TensorFlow SavedModel directory), the miner scans candidates, runs a **compatibility probe** (same live/CSV feature path as `live_probe_feature_matrix_for_custom` in `pipeline.py`), and asks to deploy.

### Contract (A + B)

- **Auto-detect** the model file among allowed artifacts (`.joblib`, `.keras`, `.h5`, SavedModel dirs); if several match, the miner prompts you to pick one.
- **`plugin_metadata.json` is always required** for a valid plugin folder (created on export).

### Supported custom model types

- **scikit-learn**: any regressor with `.predict(X)` where `X` has shape `(n_samples, n_features)` matching `feature_contract.json` (use a `Pipeline` if you need scaling).
- **TensorFlow / Keras**: dense input `(batch, n_features)` **or** fixed sequence input `(batch, n_steps, n_features)`; sequence length must match what the live feature builder produces (see built-in LSTM/RNN behavior in `pipeline.py`).

### Deploy-time checks

Before deploy, the miner loads the model and runs one **probe prediction** on engineered features for a current timestamp (Supabase/Supabase Storage) or the nearest test-row timestamp (CSV). Failures (missing columns, wrong shape, NaNs, load errors) are logged and you get:

- **[1] Exit** — stop the miner
- **[2] Baseline** — run moving-average predictions
- **[3] Train built-in** — enter the usual linear / cart / rnn / lstm flow

### Troubleshooting

- **Supabase “no forecast row” on deploy**: the compatibility probe first tries timestamps near “now” (5‑minute grid), then wider windows, then the **latest** test-table row for your `forecast_horizon_min`. If deploy still fails, your test table may have no rows for that horizon.
- **Sklearn `InconsistentVersionWarning`**: train and deploy with the **same scikit-learn major/minor** (e.g. match Colab’s version in `pip freeze` on the VM) to avoid subtle prediction drift.
- **Keras load errors (`quantization_config`, deserialize)**: the model was saved with **newer Keras** than your VM’s TensorFlow supports. Align exact versions between Colab and VM before training (current recommended pin for this project: `tensorflow==2.21.0`, `keras==3.12.1`), then re-save the `.keras` artifact.

### Security

Loading `joblib` / pickle executes arbitrary code if the file is hostile. Only deploy artifacts you trained yourself or fully trust.

### Code map

- `miner_model_energy/custom_plugin_runtime.py` — export, scan, load, metadata updates
- `miner_model_energy/pipeline.py` — `live_probe_feature_matrix_for_custom` for probe + runtime inputs
- `miner_model_energy/inference_runtime.py` — `CustomModelPredictor`
- `neurons/miner.py` — preflight menus and wiring

