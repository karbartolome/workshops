"""
Funciones para visualizar conceptos sobre interpretabilidad en machine learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_interpretability_frontier(models, radius=9, add_xai=True):
    """
    Grafica la frontera de posibilidades entre interpretabilidad y poder predictivo.

    Visualiza un conjunto de modelos de machine learning dentro de un espacio
    bidimensional que representa el trade-off entre interpretabilidad y capacidad
    predictiva. Destaca la región donde convergen ensambles, XAI y restricciones
    monotónicas como una mejora en ambas dimensiones.

    Parámetros
    ----------
    models : list of str
        Lista con los nombres de los modelos a visualizar.
    radius : int, default=9
        Radio de la circunferencia que define la frontera de posibilidades.
    add_xai : bool, default=True
        Indica si se debe agregar la anotación de "Ensambles + XAI + Restricciones monotónicas" en la gráfica.

    Retorna
    -------
    None
        Muestra la gráfica directamente con `plt.show()`.
    """
    thetas = np.radians(np.linspace(15, 80, len(models)))

    model_positions = {
        model: (radius * np.sin(theta), radius * np.cos(theta))
        for model, theta in zip(models, thetas)
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    x_curve = np.linspace(0, radius, 500)
    y_curve = np.sqrt(radius**2 - x_curve**2)

    ax.plot(
        x_curve, y_curve,
        color="black",
        linestyle="--",
        lw=2,
        zorder=1
    )

    x_models = [pos[0] for pos in model_positions.values()]
    y_models = [pos[1] for pos in model_positions.values()]

    sns.scatterplot(
        x=x_models, y=y_models,
        s=300,
        ax=ax,
        zorder=2
    )

    label_bbox = dict(
        boxstyle="round,pad=0.25",
        fc="white",
        ec="none",
        alpha=0.9
    )

    for model_name, (x_pos, y_pos) in model_positions.items():
        ax.text(
            x_pos + 0.25, y_pos, model_name,
            va="center",
            fontsize=12,
            bbox=label_bbox,
            zorder=4
        )

    if add_xai:
        x_ensemble, y_ensemble = model_positions["Ensambles"]
        x_enhanced = x_ensemble + 3.5
        y_enhanced = y_ensemble - 0.75

        ax.annotate(
            "",
            xy=(x_enhanced, y_enhanced),
            xytext=(x_ensemble, y_ensemble),
            arrowprops=dict(
                arrowstyle="->",
                color="darkorange",
                lw=1,
                linestyle="--",
                connectionstyle="angle,angleA=0,angleB=-90",
                shrinkA=10,
                shrinkB=10
            ),
            zorder=2
        )

        ax.scatter(
            x_enhanced, y_enhanced,
            color="darkorange",
            s=300,
            zorder=3
        )

        ax.text(
            x_enhanced + 0.25,
            y_enhanced,
            "Ensambles\n+XAI\n+Restricciones monotónicas",
            color="darkorange",
            va="center",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.25",
                fc="white",
                ec="darkorange",
                lw=0.8,
                alpha=0.9
            ),
            zorder=4
        )

    ax.set_xlabel("Interpretabilidad", fontsize=12)
    ax.set_ylabel("Poder predictivo", fontsize=12)

    ax.set_xlim(0, radius + 2)
    ax.set_ylim(0, radius + 1)

    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()
