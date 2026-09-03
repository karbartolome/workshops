import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def plot_monotonic_comparison(
    df,
    feature,
    target,
    modelo_sin_restriccion,
    modelo_con_restriccion,
    bins=8,
    color1="#003f7e",
    color2="#FF9933",
    alpha=0.05,
    fig_width=8,
    fig_height=4,
):
    """Compara datos observados y probabilidades predichas con y sin restricción monotónica.

    Genera dos gráficos: el primero muestra los datos observados junto con el
    promedio del target por bin de la variable `feature`; el segundo compara
    las probabilidades predichas por ambos modelos a lo largo del rango de
    `feature`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con los datos de entrada.
    feature : str
        Nombre de la variable a graficar en el eje x.
    target : str
        Nombre de la variable objetivo (binaria).
    modelo_sin_restriccion : sklearn estimator
        Modelo ajustado sin restricción monotónica.
    modelo_con_restriccion : sklearn estimator
        Modelo ajustado con restricción monotónica.
    bins : int, optional
        Cantidad de bins utilizados para resumir los datos observados, por defecto 8.

    Returns
    -------
    None
        La función muestra los gráficos directamente (`plt.show()`) y no retorna ningún valor.
    """
    data = df.copy()

    # Datos observados y promedio por bin
    grid = np.arange(data[feature].min(), data[feature].max() + 1)
    X_grid = pd.DataFrame({feature: grid})

    obs = (
        data.assign(_bin=pd.cut(data[feature], bins=bins))
        .groupby("_bin", observed=False)
        .agg(feature_medio=(feature, "mean"), target_medio=(target, "mean"))
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.scatter(data[feature], data[target], alpha=alpha, s=35, color=color1)
    ax.plot(
        obs["feature_medio"],
        obs["target_medio"],
        marker="o",
        linewidth=1,
        color="black",
    )
    ax.set(xlabel=feature, ylabel=f"Target ({target})")
    plt.show()

    # Dependencia parcial
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    display_sin_restriccion = PartialDependenceDisplay.from_estimator(
        modelo_sin_restriccion,
        data,
        features=[feature],
        ax=ax,
        line_kw={
            "color": color1,
            "label": "Sin restricción",
            "linewidth": 2.5,
            "alpha": 0.9,
        },
    )

    PartialDependenceDisplay.from_estimator(
        modelo_con_restriccion,
        data,
        features=[feature],
        ax=display_sin_restriccion.axes_,
        line_kw={
            "color": color2,
            "label": "Con restricción",
            "linewidth": 2.5,
            "alpha": 0.9,
        },
    )

    plt.xlabel(feature)
    plt.tight_layout()
    plt.show()
