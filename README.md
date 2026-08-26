# Workshops

Repositorio de talleres sobre tópicos de Ciencia de Datos.

## Estructura del repositorio

Cada taller vive en su propia carpeta dentro de [slides/](slides/), por ejemplo [slides/20240513-uba-calibracion/](slides/20240513-uba-calibracion/). Una carpeta se considera un taller siempre que contenga una presentación `slides*.html` ya renderizada (el `.qmd` fuente, notebooks, imágenes, etc. pueden convivir junto a ella).

El home del sitio se crea a partir de 3 archivos:

-   [**main.py**](main.py) — el script generador. Se debe ejecutar: `python main.py` cada vez que se agregue un nuevo taller o cambie una descripción/tag. El script:
    1.  Recorre `slides/` en busca de carpetas que contengan una presentación renderizada.
    2.  Agrega una entrada esqueleto a [metadata/slides_descriptions.json](metadata/slides_descriptions.json) por cada nueva presentación descubierta (título/fecha/descripción/tags quedan vacíos para que los completes).
    3.  Regenera [index.qmd](index.qmd) con una grilla de tarjetas filtrable, una tarjeta por presentación, ordenadas por fecha.
    4.  Renderiza `index.qmd` a `index.html` con el CLI de Quarto (se puede omitir este paso con `python main.py --no-render`).
    5.  Reescribe [404.html](404.html) para que los links a las URLs antiguas de los talleres (previas a la migración, por ejemplo `.../20240513-uba-calibracion/slides`, de cuando las presentaciones no estaban dentro de `slides/`) sigan redirigiendo a su nueva ubicación bajo `slides/`. GitHub Pages sirve este archivo cuando la ruta solicitada no existe, por lo que el redirect ocurre del lado del cliente.
-   [**index.qmd**](index.qmd) — el código fuente Quarto generado para la homepage. `main.py` lo sobrescribe en cada ejecución, solo se edita desde el generador (`main.py`), no este archivo directamente.
-   [**index.html**](index.html) — la homepage renderizada, producida por Quarto a partir de `index.qmd`. Este es el archivo que GitHub Pages sirve efectivamente en `/`. También es regenerado por `main.py`, no debe ser editado a mano.

## Python envs

Los entornos de Python reutilizables para los talleres viven en [envs/](envs/). Consultá [envs/README.md](envs/README.md) para instrucciones detalladas sobre cómo crear y registrar un entorno (kernel de Jupyter incluido) usando `uv`.

Una vez creado el entorno, referenciá su kernel en el frontmatter del `.qmd`:

```         
jupyter: 
  kernelspec:
    name: "ds_model_1_0"
    language: "python"
    display_name: "Python (ds_model_1_0)"
```