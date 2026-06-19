# Day 01, 2026 06 20

## What got built today

The project scaffold and the entire data layer. The dataset is NASA C MAPSS FD001 (turbofan run to failure runs). By end of day, raw text files can be parsed into clean DataFrames, sensors that carry no information are dropped, values are normalized, RUL labels are attached using a piecewise linear scheme, and a sliding window generator turns the time series into fixed shape tensors ready for a PyTorch model. A torch Dataset wraps it all.

## Commits in order

1. initial project scaffold: created folders, requirements.txt, .gitignore, README skeleton
2. add data download and inspect scripts: data/download.py and data/inspect.py
3. data loader and preprocessing with sensor selection: src/data/loader.py and src/data/preprocess.py
4. windowing util and readme cleanup: src/data/windows.py plus a README rewrite
5. torch dataset wrapper: src/data/dataset.py

## File by file walkthrough

### requirements.txt
Lists every Python package the project needs: torch (the model), pandas and numpy (data handling), scikit learn (baseline regression and metrics), fastapi and uvicorn (the API later), matplotlib and seaborn (EDA plots), jupyter and notebook (running notebooks), pyyaml (config files), pytest (tests), httpx (for API tests), gradio (the dashboard later).

### .gitignore
Tells git to ignore generated and private stuff: data/raw/ (raw dataset, too big to commit), checkpoints/ (model weights), __pycache__/, *.pt and *.pth (PyTorch weights), .env (secrets), .DS_Store, build artifacts, virtual envs, IDE config folders, and notebook checkpoints.

### README.md
The public face of the repo. Sections: Problem (why predictive maintenance matters), Approach (shared GRU + three heads), Dataset (FD001 spec), Architecture (ASCII diagram), Results (placeholder table), Demo (placeholder), Setup (clone + install). Used as a placeholder, will be filled in with real numbers at the end.

### data/download.py
Purpose: fetch the C MAPSS dataset, or print clear manual instructions if automatic download fails.
Key parts:
* tries to download a public mirror of CMAPSSData.zip into data/raw/
* if the URL fails (NASA mirrors change), prints fallback instructions pointing at the official NASA Prognostics Center page
* unzips and keeps only the FD001 files: train_FD001.txt, test_FD001.txt, RUL_FD001.txt

### data/inspect.py
Purpose: sanity check the data is loaded correctly. Run once after download.
Key parts:
* loads each file with the loader
* prints shape, dtypes, head, basic stats per sensor
* histograms the per engine cycle length distribution (so we know what window size makes sense)

### src/data/loader.py
Purpose: turn the raw text files into pandas DataFrames.
Key parts:
* DATA_DIR points to data/raw at repo root
* COLS is the fixed column schema for C MAPSS: engine_id, cycle, three operating settings (os1 to os3), 21 sensors (s1 to s21)
* load_train() reads train_FD001.txt
* load_test() reads test_FD001.txt
* load_rul() reads RUL_FD001.txt (one number per test engine: ground truth RUL at the last cycle in the test file)
* _load(filename) is the private helper: pd.read_csv with whitespace separator since the files are space delimited, no header, columns set to COLS

### src/data/preprocess.py
Purpose: clean and label the loaded data so it's ready to window.
Key parts:
* DROP_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]: these sensors are nearly constant in FD001 and add no information. This is a well known finding for this dataset and dropping them reduces noise and parameter count.
* SENSOR_COLS: the remaining 14 informative sensors
* add_rul_labels(df, rul_clip=125): for every row, RUL equals (last cycle of that engine) minus (current cycle). Clipped at 125 because the engine is essentially healthy until degradation kicks in. Without clipping the model would waste capacity trying to predict large RUL values that have no discriminative signal. 125 is the standard choice in the C MAPSS literature.
* fit_scaler(train_df): computes per sensor min and max on the training set only. We never peek at test stats.
* normalize(df, mins, maxs): min max scales each sensor to roughly [0, 1]. denom is guarded against zero division by replacing 0 with 1.
* prepare_train(train_df, rul_clip): calls add_rul_labels then fit_scaler then normalize. Returns the prepared df plus the saved mins and maxs so we can apply the same transform to the test set.
* prepare_test(test_df, rul_df, mins, maxs): applies the saved scaling to the test set. The true RUL for the last cycle of each test engine comes from rul_df.

### src/data/windows.py
Purpose: turn long time series per engine into fixed shape supervised examples.
Constants:
* WINDOW_SIZE = 30: the model looks at 30 cycles at a time. Long enough to capture trend, short enough that most engines produce many windows.
* FORECAST_HORIZON = 5: the forecast head predicts 5 cycles ahead.

Functions:
* make_windows(df, window_size, forecast_horizon): groups by engine_id, slides a window across each engine's history, and for each window position returns:
    X: the input window (30 cycles, 14 sensors)
    y_rul: the RUL value at the last cycle of the window
    y_fore: the next 5 cycles of sensor values (the target for the forecast head)
  Loop range goes up to n minus window_size minus forecast_horizon plus 1 so that we always have a valid 5 step future to predict.
* make_test_windows(df, window_size): returns the last window per engine. Used at inference time when we only care about predicting from current state, not training. If the engine's history is shorter than 30 cycles we left pad with zeros.

### src/data/dataset.py
Purpose: wrap the windowed arrays in a PyTorch Dataset so a DataLoader can batch and shuffle them.
* class CMAPSSWindows(Dataset): stores X, y_rul, y_fore as torch tensors.
* __len__ returns number of windows.
* __getitem__(idx) returns the triple at index idx so the multitask trainer can compute both losses on the same batch.

## Design decisions made today

* Drop sensors 1, 5, 6, 10, 16, 18, 19: they are constant or near constant in FD001 and just add noise.
* Window size 30: balances trend capture vs sample count.
* Forecast horizon 5: short enough that predictions stay accurate, long enough to be useful for early warning.
* RUL clip 125: standard C MAPSS choice. Treats engines as healthy beyond 125 cycles remaining.
* Min max normalization fit on train only: prevents data leakage from test set.
* One Dataset class serving all three tasks: X is shared, y_rul and y_fore both come back per item. This is the foundation of the shared backbone design.

## What to study before interview about today's work

* What is NASA C MAPSS FD001 and how it was generated (simulated turbofan run to failure under one operating condition and one fault mode).
* Why we drop near constant sensors: zero variance features add parameters without information.
* Why piecewise linear RUL with clipping: assumes engines are in a healthy regime early on so the model focuses learning capacity on the degradation phase.
* Why min max scaling fit only on train: leakage and reproducibility.
* Why sliding windows for time series: convert a variable length sequence problem into a fixed shape supervised learning problem.
* Why one Dataset returning both targets: the shared backbone needs the same input to produce both loss signals in a single forward pass.
