# Entornos

Esta carpeta contiene definiciones de entornos reutilizables para los talleres.
Cada versión de un entorno vive en su propia carpeta e incluye:

- `env_creation.sh`: script usado para crear el entorno virtual.
- `requirements.txt`: paquetes de Python fijados (pinned) que instala el script.

## Entornos disponibles

| env_name | version | path | descripción |
| --- | --- | --- | --- |
| `ds_model` | `1.0` | `envs/ds_model/1.0` | Librerías estándar para modelado de ciencia de datos. |

## Requisitos

Los scripts de entornos actuales usan `uv` para crear entornos virtuales e instalar paquetes.

Instalá `uv` antes de ejecutar cualquier script de entorno:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Después de instalarlo, reiniciá la terminal o asegurate de que `uv` esté disponible en tu `PATH`:

```
uv --version
```

## Crear un entorno

Ubicate en la carpeta de la versión del entorno que querés crear:

```
cd envs/ds_model/1.0
```

Si hace falta, dale permisos de ejecución al script:

```
chmod +x env_creation.sh
```

Revisá `requirements.txt` y `env_creation.sh` antes de ejecutar el script.

Ejecutá el script desde dentro de la carpeta del entorno:

```
./env_creation.sh
```

Por defecto, el script:

- Instala Python `3.12` usando `uv`.
- Crea un entorno virtual local en `.venv`.
- Instala los paquetes fijados en `requirements.txt`.
- Registra un kernel de Jupyter llamado `ds_model_1_0`.

Activá el entorno:

```
source .venv/bin/activate
```

## Overrides opcionales

El script admite variables de entorno para personalizar la instalación sin editar el archivo.

Usar una versión distinta de Python:

```
PYTHON_VERSION=3.13 ./env_creation.sh
```

Usar un directorio distinto para el entorno virtual:

```
VENV_DIR=.venv_ds_model ./env_creation.sh
```

Usar un nombre distinto para el kernel de Jupyter:

```
ENV_NAME=my_ds_env ./env_creation.sh
```

Usar un archivo de requisitos distinto:

```
REQUIREMENTS_FILE=requirements-dev.txt ./env_creation.sh
```

## Entorno de Modelado de Ciencia de Datos

El entorno `ds_model/1.0` incluye paquetes fijados para flujos de trabajo de modelado habituales:

- Trabajo con datos: `numpy`, `pandas`, `scipy`, `pyarrow`.
- Modelado: `scikit-learn`, `statsmodels`, `imbalanced-learn`.
- Modelos de boosting: `xgboost`, `lightgbm`, `catboost`.
- Experimentación e interpretabilidad: `optuna`, `shap`.
- Visualización y reportes: `matplotlib`, `seaborn`, `plotly`, `great-tables`.
- Soporte de notebooks: `jupyterlab`, `notebook`, `ipykernel`.
- Utilidades e I/O: `openpyxl`, `xlrd`, `SQLAlchemy`, `requests`, `python-dotenv`, `tqdm`, `joblib`.

## Estructura de carpetas

Usá esta estructura al agregar un nuevo entorno:

```
envs/
  {env_name}/
    {version}/
      env_creation.sh
      requirements.txt
```

Ejemplo:

```
envs/
  ds_model/
    1.0/
      env_creation.sh
      requirements.txt
```

Mantené las versiones de los paquetes fijadas en `requirements.txt` para que los cambios de entorno sean explícitos y fáciles de revisar.
