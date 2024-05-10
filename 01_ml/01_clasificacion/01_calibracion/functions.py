import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from IPython.display import display
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    recall_score,
    brier_score_loss,
)
from great_tables import GT, from_column, style, loc, html


color_verde = "#255255"
color_verde_claro = "#BDCBCC"


def plot_distribution(
    data,
    obs_column="y_obs",
    pred_column="pred_prob",
    positive_class=1,
    class_0_color=None,
    class_1_color=None,
):
    if class_0_color is None:
        class_0_color = "blue"
    if class_1_color is None:
        class_1_color = "red"
    # sns.set(font_scale=0.8)
    plt.figure(figsize=(3.5, 1.7))
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
    y_obs,
    y_pred_prob,
    y_pred_prob_cal_sigmoid,
    y_pred_prob_cal_isotonic,
    bins=10,
    strategy="uniform",
    figsize=(6, 6),
    display_bins=False,
    model_name="",
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
    if y_pred_prob_cal_sigmoid.notna().any():
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_obs, y_pred_prob_cal_sigmoid, n_bins=bins, strategy=strategy
        )

        plt.plot(
            prob_pred_cal,
            prob_true_cal,
            marker="o",
            color="blue",
            label="Probabilidades calibradas (sigmoide)",
        )
    if y_pred_prob_cal_isotonic.notna().any():
        prob_true_cal, prob_pred_cal = calibration_curve(
            y_obs, y_pred_prob_cal_isotonic, n_bins=bins, strategy=strategy
        )

        plt.plot(
            prob_pred_cal,
            prob_true_cal,
            marker="o",
            color='green',
            label="Probabilidades calibradas (isotonic)",
        )

    plt.axline([0, 0], [1, 1], label="Calibración perfecta", c=color_verde)
    plt.text(0.8, 0.85, "Línea 45°", fontsize=10, rotation=45, c=color_verde)

    plt.legend()
    plt.title('Reliability diagram')
    plt.xlabel("Probabilidad predicha promedio")
    plt.ylabel("Fracción de positivos (clase positiva=1)")
    plt.text(0.10, 0.70, "Subestimación", fontsize=10, rotation=45)
    plt.text(0.70, 0.10, "Sobreestimación", fontsize=10, rotation=45)


def calibrate_model(
    pipe,
    X_valid,
    y_valid,
    X_test,
    y_test,
    cal_sigmoid=True,
    cal_isotonic=True,
    model_name="",
    plot_cal=True,
):
    preds = pd.DataFrame(
        {
            "y_obs": y_test,
            "pred_class": pipe.predict(X_test),
            "pred_prob": pipe.predict_proba(X_test)[:, 1],
        }
    )

    preds["pred_sigmoid"] = np.nan
    preds["pred_isotonic"] = np.nan
    pipe_sigmoid = None
    pipe_isotonic = None

    if cal_sigmoid:
        # Calibración sigmoide (Platt's scaling)
        pipe_sigmoid = CalibratedClassifierCV(pipe, cv="prefit", method="sigmoid")
        pipe_sigmoid.fit(X_valid, y_valid)
        prob_pos_sigmoid = pipe_sigmoid.predict_proba(X_test)[:, 1]
        preds["pred_sigmoid"] = prob_pos_sigmoid
        preds["pred_class_sigmoid"] = [1 if i > 0.5 else 0 for i in prob_pos_sigmoid]

    if cal_isotonic:
        # Calibración isotónica
        pipe_isotonic = CalibratedClassifierCV(pipe, cv="prefit", method="isotonic")
        pipe_isotonic.fit(X_valid, y_valid)
        prob_pos_isotonic = pipe_isotonic.predict_proba(X_test)[:, 1]
        preds["pred_isotonic"] = prob_pos_isotonic
        preds["pred_class_isotonic"] = [1 if i > 0.5 else 0 for i in prob_pos_isotonic]

    if plot_cal:
        plot_calibration_models(
            y_obs=preds["y_obs"],
            y_pred_prob=preds["pred_prob"],
            y_pred_prob_cal_sigmoid=preds["pred_sigmoid"],
            y_pred_prob_cal_isotonic=preds["pred_isotonic"],
            bins=7,
            strategy="uniform",
            figsize=(4, 4),
            display_bins=True,
            model_name=model_name,
        )
        plt.legend(loc="center left", bbox_to_anchor=(0, -0.3))
        plt.show()

    return pipe_sigmoid, pipe_isotonic, preds


def plot_calibration(
    preds, y_pred_prob_calibrated, y_pred_prob="pred_prob", model_name=""
):
    plt.figure(figsize=(4, 4))
    sns.scatterplot(
        x=y_pred_prob,
        y=y_pred_prob_calibrated,
        data=preds,
        color="blue",
        alpha=0.7,
        label="Calibración sklearn",
    )
    sns.scatterplot(
        x="pred_prob",
        y="y_obs",
        data=preds,
        alpha=0.3,
        color="black",
        label="Valores observados",
    )
    plt.axline([0, 0], [1, 1], label="Calibración perfecta", c=color_verde)
    plt.text(0.8, 0.85, "Línea 45°", fontsize=10, rotation=45, c=color_verde)
    plt.ylabel("Probabilidad calibrada")
    plt.xlabel("Probabilidad no calibrada")
    plt.legend(loc="center left", bbox_to_anchor=(0, -0.3))
    plt.title(f"Calibración {model_name}")
    plt.ylim([-0.01, 1.01])


def calibration_table(preds, pred_column="pred_prob", bins=10, title=""):
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

    gain_table["Bin"] = gain_table["Bin"].apply(
        lambda x: f"({x.left*100:.2f}%, {x.right*100:.2f}%]"
    )

    gain_table = (
        GT(gain_table.round(2))
        .cols_label(
            avg_prob=html("Probabilidad<br>promedio"),
            perc_clase1=html("% Clase 1<br>(observado)"),
            diff=html("Diferencia <br>|avg prob - % clase 1|"),
            N_clase1=html("N Clase 1"),
        )
        .fmt_percent(["avg_prob", "perc_clase1", 'diff'])
        .data_color(
            columns=["avg_prob", "perc_clase1"],
            palette=[color_verde_claro, color_verde],
        )
        .data_color(columns=["diff"], palette=["white", "red"], domain=[0,1])
        #.tab_header(title)
    )
    return display(gain_table)


def calculate_clf_metrics(y_true, y_pred, y_pred_prob, name):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred)
    metrics["Recall"] = recall_score(y_true, y_pred)
    metrics["F1"] = f1_score(y_true, y_pred)
    metrics["ROC AUC"] = roc_auc_score(y_true, y_pred_prob)
    metrics["Log Loss"] = log_loss(y_true, y_pred_prob)
    metrics["Brier Loss"] = brier_score_loss(y_true, y_pred_prob)
    metrics_df = pd.DataFrame(metrics, index=[name])
    return metrics_df
