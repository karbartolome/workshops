import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from IPython.display import display


def plot_distribution(
    data, obs_column="y_obs", pred_column="pred_prob", positive_class=1,
    class_0_color=None, class_1_color=None
):
    if class_0_color is None:
        class_0_color = 'blue'
    if class_1_color is None:
        class_1_color = 'red'
    #sns.set(font_scale=0.8)
    plt.figure(figsize=(4, 2))
    sns.kdeplot(
        data=data[data[obs_column] == positive_class],
        x=pred_column,
        color=class_1_color,
        label="Class 1",
    )
    sns.kdeplot(
        data=data[data[obs_column] != positive_class],
        x=pred_column,
        color=class_0_color,
        label="Class 0",
    )
    plt.xlim([0, 1])
    plt.legend()
    plt.show()


def plot_calibration_models(
    y_obs, y_pred_prob, y_pred_prob_cal, bins=10, strategy="uniform", figsize=(6,6),display_bins=False
) -> None:
    """
    Gráfico comparando probabilidad vs probabilidad calibrada
    Inputs:
        y_obs: Valor observado (0,1)
        y_pred_prob: Valor predicho por el modelo no calibrado
        y_pred_prob_cal: Valor predicho por el modelo calibrado
        bins: Cantidad de bins
        strategy: 'uniform', 'quantile'
    Returns:
        plot
    """
    prob_true, prob_pred = calibration_curve(
        y_obs, y_pred_prob, n_bins=bins, strategy=strategy
    )

    plt.figure(figsize=figsize)
    plt.plot(
        prob_pred,
        prob_true,
        marker="o",
        color="red",
        label="Probabilidades no calibradas",
    )
    if y_pred_prob_cal is not None:
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_obs, y_pred_prob_cal, n_bins=bins, strategy=strategy
        )

        plt.plot(
            prob_pred_cal,
            prob_true_cal,
            marker="o",
            color="blue",
            label="Probabilidades calibradas",
        )

    plt.axline([0, 0], [1, 1], label="Calibración perfecta", c="green")
    plt.text(0.8, 0.85, "Línea 45°", fontsize=10, rotation=45, c='green')

    plt.legend()
    #plt.title("Probabilidades calibradas vs no calibradas")
    plt.xlabel("Probabilidad predicha promedio")
    plt.ylabel("Fracción de positivos (clase positiva=1)")
    plt.text(0.10, 0.70, "Subestimación", fontsize=10, rotation=45)
    plt.text(0.70, 0.10, "Sobreestimación", fontsize=10, rotation=45)

    if display_bins:
        return (pd.DataFrame({
            'prob_true': prob_true,
            'prob_pred': prob_pred
        }))


def gain_table(preds, pred_column="pred_prob", bins=10, title=""):
    preds["Bin"] = pd.qcut(preds[pred_column], q=bins, duplicates="drop")

    gain_table = (
        preds.groupby("Bin", as_index=False)
        .agg(
            N=("y_obs", "count"),
            N_clase1=("y_obs", "sum"),
            avg_prob=(pred_column, "mean"),
        )
        .assign(
            perc_clase1=lambda x: x["N_clase1"] / x["N"],
            diff=lambda x: np.abs(x["avg_prob"] - x["perc_clase1"]),
        )
    )
    gain_table["N_acum"] = gain_table["N"].cumsum()
    gain_table["N_clase1_acum"] = gain_table["N_clase1"].cumsum()
    gain_table["TPR"] = gain_table["N_clase1_acum"] / gain_table["N_acum"]
    gain_table["Bin"] = gain_table["Bin"].apply(
        lambda x: f"({x.left*100:.2f}%, {x.right*100:.2f}%]"
    )
    gain_table = (
        gain_table.style.background_gradient(axis=0)
        .format(
            {
                "avg_prob": "{:.2%}",
                "perc_clase1": "{:.2%}",
                "diff": "{:.3f}",
                "TPR": "{:.2%}",
            }
        )
        .background_gradient(subset=["diff"], cmap="Reds", vmin=0, vmax=1)
        .hide(axis=0)
        #.set_caption(f"Tabla de ganancias {title}")
    )
    return display(gain_table)
