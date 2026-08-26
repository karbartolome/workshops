#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-ds_model_1_0}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VENV_DIR="${VENV_DIR:-.venv}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-requirements.txt}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to create this environment."
  echo "Install it first: https://docs.astral.sh/uv/getting-started/installation/"
  exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "Requirements file not found: ${REQUIREMENTS_FILE}"
  exit 1
fi

uv python install "${PYTHON_VERSION}"
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"
uv pip install --python "${VENV_DIR}/bin/python" -r "${REQUIREMENTS_FILE}"

echo "Environment created in ${VENV_DIR}"
echo "Activate with: source ${VENV_DIR}/bin/activate"

"${VENV_DIR}/bin/python" -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})"

echo "Jupyter kernel created: Python (${ENV_NAME})"
