# Workshops

> Karina Bartolomé

## 2024

[2024-05-13: Calibración de probabilidades](https://karbartolome.github.io/workshops/20240513-uba-calibracion/slides)
CIMBAGE (IADCOM) - Facultad Ciencias Económicas (UBA)

## 2025

[2025-09-05: Del dato al modelo](https://karbartolome.github.io/workshops/20250905-uba-pipelines/slides)
CIMBAGE (IADCOM) - Facultad Ciencias Económicas (UBA)


## Configs

Para configurar kernelspec: 

conda create -n quarto-env python=3.10
conda activate quarto-env
conda install notebook ipykernel
ipython kernel install --user --name quarto-env --display-name quarto-env
conda install numpy
etc.

```
jupyter: 
  kernelspec:
    name: "quarto-env"
    language: "python"
    display_name: "quarto-env"
```
