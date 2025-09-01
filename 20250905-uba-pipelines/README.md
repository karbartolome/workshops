# Del dato al modelo

Virtual environment:

```
brew install python@3.12
brew link python@3.12 --force
python3.12 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install uv
uv pip install -r requirements.txt

python -m ipykernel install --user --name=.venv --display-name "uba-pipelines"
```