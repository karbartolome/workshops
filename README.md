# workshops


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