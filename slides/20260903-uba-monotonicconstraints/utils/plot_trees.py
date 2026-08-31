"""
Funciones para visualizar árboles de decisión de modelos HistGradientBoosting.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_hist_gradient_boosting_tree(model, iteration, feature_names=None, X_reference=None, ax=None):
    """
    Visualiza un árbol individual de un HistGradientBoostingClassifier entrenado.

    Dibuja la estructura del árbol con sus nodos de decisión y hojas, incluyendo
    información sobre las probabilidades predichas en cada hoja si se proporciona
    un conjunto de referencia de datos.

    Parámetros
    ----------
    model : HistGradientBoostingClassifier
        Modelo entrenado del que se extraerá el árbol.
    iteration : int
        Índice del árbol a visualizar (0-based).
    feature_names : list of str, optional
        Nombres de las variables características. Si es None, se utilizan nombres
        genéricos como 'x0', 'x1', etc.
    X_reference : array-like de forma (n_samples, n_features), optional
        Datos de referencia para calcular probabilidades predichas en cada hoja.
    ax : matplotlib.axes.Axes, optional
        Objeto de ejes donde dibujar. Si es None, se crea uno nuevo.

    Retorna
    -------
    ax : matplotlib.axes.Axes
        El objeto de ejes con el árbol dibujado.
    """
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(model.n_features_in_)]

    tree = model._predictors[iteration][0]
    nodes = tree.nodes
    node_positions = {}
    node_labels = {}
    leaf_probabilities = {}

    if X_reference is not None:
        if hasattr(X_reference, "loc"):
            X_values = X_reference.loc[:, feature_names].to_numpy()
        else:
            X_values = np.asarray(X_reference)

        leaf_ids = _leaf_ids_for_tree(tree, X_values)
        predicted_probabilities = model.predict_proba(X_reference)[:, 1]

        for leaf_id in np.unique(leaf_ids):
            in_leaf = leaf_ids == leaf_id
            leaf_probabilities[int(leaf_id)] = {
                "mean_probability": predicted_probabilities[in_leaf].mean(),
                "n": in_leaf.sum(),
            }

    def walk(node_id, depth=0, x_pos=0):
        node = nodes[node_id]
        if node["is_leaf"]:
            node_positions[node_id] = (x_pos, -depth)
            label = ""
            if node_id in leaf_probabilities:
                mean_probability = leaf_probabilities[node_id]["mean_probability"]
                n = leaf_probabilities[node_id]["n"]
                label = f"P media = {mean_probability:.2f}\nn = {n}"
            node_labels[node_id] = label
            return x_pos + 1

        left_child = int(node["left"])
        right_child = int(node["right"])
        next_x = walk(left_child, depth + 1, x_pos)
        next_x = walk(right_child, depth + 1, next_x)
        left_x = node_positions[left_child][0]
        right_x = node_positions[right_child][0]
        node_positions[node_id] = ((left_x + right_x) / 2, -depth)
        feature = feature_names[int(node["feature_idx"])]
        node_labels[node_id] = f"{feature} <= {node['num_threshold']:.1f}"
        return next_x

    walk(0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for node_id, node in enumerate(nodes):
        if node["is_leaf"]:
            continue
        x0, y0 = node_positions[node_id]
        for child_id in (int(node["left"]), int(node["right"])):
            x1, y1 = node_positions[child_id]
            ax.plot([x0, x1], [y0, y1], color="0.45", linewidth=1)

    for node_id, (x_pos, y_pos) in node_positions.items():
        ax.text(
            x_pos,
            y_pos,
            node_labels[node_id],
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.35"},
        )

    ax.set_title(f"Iteración {iteration + 1}")
    ax.axis("off")
    return ax

def _leaf_ids_for_tree(tree, X_values):
    """
    Obtiene el identificador de hoja alcanzado por cada fila en los datos.

    Para cada muestra en los datos de entrada, recorre el árbol siguiendo
    las condiciones de decisión hasta alcanzar una hoja y retorna su identificador.

    Parámetros
    ----------
    tree : TreePredictor
        Estructura del árbol del modelo.
    X_values : array-like de forma (n_samples, n_features)
        Datos de entrada.

    Retorna
    -------
    leaf_ids : ndarray de forma (n_samples,)
        Identificador de la hoja alcanzada para cada muestra.
    """
    nodes = tree.nodes
    leaf_ids = []

    for row in X_values:
        node_id = 0
        while not nodes[node_id]["is_leaf"]:
            node = nodes[node_id]
            feature_idx = int(node["feature_idx"])
            threshold = node["num_threshold"]
            if row[feature_idx] <= threshold:
                node_id = int(node["left"])
            else:
                node_id = int(node["right"])
        leaf_ids.append(node_id)

    return np.array(leaf_ids)