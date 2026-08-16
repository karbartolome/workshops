# Environments

This folder contains reusable environment definitions for workshops.
Each environment version lives in its own folder and includes:

- `env_creation.sh`: script used to create the virtual environment.
- `requirements.txt`: pinned Python packages installed by the script.

## Available Environments

| env_name | version | path | description |
| --- | --- | --- | --- |
| `ds_model` | `1.0` | `envs/ds_model/1.0` | Standard libraries for data science modelling. |

## Requirements

The current environment scripts use `uv` to create virtual environments and install packages.

Install `uv` before running any environment script:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installing, restart the terminal or make sure `uv` is available in your `PATH`:

```
uv --version
```

## Create An Environment

Move into the folder for the environment version you want to create:

```
cd envs/ds_model/1.0
```

Make the script executable if needed:

```
chmod +x env_creation.sh
```

Review `requirements.txt` and `env_creation.sh` before running the script.

Run the script from inside the environment folder:

```
./env_creation.sh
```

By default, the script:

- Installs Python `3.12` using `uv`.
- Creates a local virtual environment in `.venv`.
- Installs pinned packages from `requirements.txt`.
- Registers a Jupyter kernel named `ds_model_1_0`.

Activate the environment:

```
source .venv/bin/activate
```

## Optional Overrides

The script supports environment variables so you can customize the setup without editing the file.

Use a different Python version:

```
PYTHON_VERSION=3.13 ./env_creation.sh
```

Use a different virtual environment directory:

```
VENV_DIR=.venv_ds_model ./env_creation.sh
```

Use a different Jupyter kernel name:

```
ENV_NAME=my_ds_env ./env_creation.sh
```

Use a different requirements file:

```
REQUIREMENTS_FILE=requirements-dev.txt ./env_creation.sh
```

## Data Science Modelling Environment

The `ds_model/1.0` environment includes pinned packages for common modelling workflows:

- Core data work: `numpy`, `pandas`, `scipy`, `pyarrow`.
- Modelling: `scikit-learn`, `statsmodels`, `imbalanced-learn`.
- Boosting models: `xgboost`, `lightgbm`, `catboost`.
- Experimentation and explainability: `optuna`, `shap`.
- Visualization and reporting: `matplotlib`, `seaborn`, `plotly`, `great-tables`.
- Notebook support: `jupyterlab`, `notebook`, `ipykernel`.
- Utility and IO: `openpyxl`, `xlrd`, `SQLAlchemy`, `requests`, `python-dotenv`, `tqdm`, `joblib`.

## Folder Structure

Use this structure when adding a new environment:

```
envs/
  {env_name}/
    {version}/
      env_creation.sh
      requirements.txt
```

Example:

```
envs/
  ds_model/
    1.0/
      env_creation.sh
      requirements.txt
```

Keep package versions pinned in `requirements.txt` so environment changes are explicit and reviewable.
