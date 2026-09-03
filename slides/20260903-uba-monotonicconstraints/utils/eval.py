import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    log_loss,
)

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
    data_features = data[modelo_sin_restriccion.feature_names_in_]

    display_sin_restriccion = PartialDependenceDisplay.from_estimator(
        modelo_sin_restriccion,
        data_features,
        features=[feature],
        response_method='predict_proba',
        method="brute",
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
        data_features,
        features=[feature],
        response_method='predict_proba',
        method="brute",
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


def compute_classification_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calcula métricas de clasificación binaria para un conjunto de predicciones.

    Parameters
    ----------
    y_true : array-like
        Valores observados de la variable objetivo (binaria).
    y_pred_proba : array-like
        Probabilidades predichas para la clase positiva.
    threshold : float, optional
        Umbral utilizado para convertir probabilidades en clases, por defecto 0.5.

    Returns
    -------
    dict
        Diccionario con las métricas `roc_auc`, `ks`, `accuracy`, `precision`,
        `recall` y `logloss`.
    """
    y_pred = (np.asarray(y_pred_proba) >= threshold).astype(int)
    ks = ks_2samp(
        np.asarray(y_pred_proba)[np.asarray(y_true) == 1],
        np.asarray(y_pred_proba)[np.asarray(y_true) == 0],
    ).statistic

    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "ks": ks,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "logloss": log_loss(y_true, y_pred_proba),
    }


def compute_train_test_metrics(model, X_train, y_train, X_test, y_test, threshold=0.5):
    """Calcula métricas de clasificación de un modelo sobre train y test.

    Parameters
    ----------
    model : sklearn estimator
        Modelo ajustado con método `predict_proba`.
    X_train, X_test : pandas.DataFrame
        Variables explicativas de entrenamiento y evaluación.
    y_train, y_test : array-like
        Variable objetivo de entrenamiento y evaluación.
    threshold : float, optional
        Umbral utilizado para convertir probabilidades en clases, por defecto 0.5.

    Returns
    -------
    pandas.DataFrame
        DataFrame con una fila por conjunto (`Train`, `Test`) y una columna
        por métrica.
    """
    rows = []
    for dataset, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        metrics = compute_classification_metrics(
            y, model.predict_proba(X)[:, 1], threshold=threshold
        )
        rows.append({"dataset": dataset, **metrics})

    return pd.DataFrame(rows)
