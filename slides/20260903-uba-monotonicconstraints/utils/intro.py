import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_interpretability_frontier(models, R=9):
    thetas = np.radians(np.linspace(15, 80, len(models)))

    data = {
        model: (R * np.sin(theta), R * np.cos(theta))
        for model, theta in zip(models, thetas)
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    # --- Possibility frontier ---
    x_curve = np.linspace(0, R, 500)
    y_curve = np.sqrt(R**2 - x_curve**2)

    ax.plot(
        x_curve, y_curve,
        color="black",
        linestyle="--",
        lw=2,
        zorder=1
    )

    # --- Models ---
    x = [v[0] for v in data.values()]
    y = [v[1] for v in data.values()]

    sns.scatterplot(
        x=x, y=y,
        s=300,
        ax=ax,
        zorder=2
    )

    text_bbox = dict(
        boxstyle="round,pad=0.25",
        fc="white",
        ec="none",
        alpha=0.9
    )

    # --- Labels ---
    for name, (xi, yi) in data.items():
        ax.text(
            xi + 0.25, yi, name,
            va="center",
            fontsize=12,
            bbox=text_bbox,
            zorder=4
        )

    # --- Ensemble + XAI + monotonic constraints ---
    x_ensemble, y_ensemble = data["Ensambles"]

    x_orange = x_ensemble + 3.5
    y_orange = y_ensemble - 0.5

    ax.annotate(
        "",
        xy=(x_orange, y_orange),
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
        x_orange, y_orange,
        color="darkorange",
        s=300,
        zorder=3
    )

    ax.text(
        x_orange + 0.25,
        y_orange,
        "Ensambles + XAI\n+ Restricciones monotónicas",
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

    # --- Axes ---
    ax.set_xlabel("Interpretabilidad", fontsize=12)
    ax.set_ylabel("Poder porangeictivo", fontsize=12)

    ax.set_xlim(0, R + 2)
    ax.set_ylim(0, R + 1)

    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.show()