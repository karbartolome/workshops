import numpy as np
import matplotlib.pyplot as plt

def plot_hist_gradient_boosting_tree(model, iteration, feature_names=None, X_reference=None, ax=None):
    """Plot one internal tree from a fitted HistGradientBoostingClassifier."""
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(model.n_features_in_)]

    tree = model._predictors[iteration][0]
    nodes = tree.nodes
    positions = {}
    labels = {}
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
            positions[node_id] = (x_pos, -depth)
            # label = f"leaf\nlog-odds += {node['value']:.3f}"
            label = ""
            if node_id in leaf_probabilities:
                mean_probability = leaf_probabilities[node_id]["mean_probability"]
                n = leaf_probabilities[node_id]["n"]
                label += f"P media = {mean_probability:.2f}\nn = {n}"
            labels[node_id] = label
            return x_pos + 1

        left = int(node["left"])
        right = int(node["right"])
        next_x = walk(left, depth + 1, x_pos)
        next_x = walk(right, depth + 1, next_x)
        left_x = positions[left][0]
        right_x = positions[right][0]
        positions[node_id] = ((left_x + right_x) / 2, -depth)
        feature = feature_names[int(node["feature_idx"])]
        labels[node_id] = f"{feature} <= {node['num_threshold']:.1f}"
        return next_x

    walk(0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    for node_id, node in enumerate(nodes):
        if node["is_leaf"]:
            continue
        x0, y0 = positions[node_id]
        for child in (int(node["left"]), int(node["right"])):
            x1, y1 = positions[child]
            ax.plot([x0, x1], [y0, y1], color="0.45", linewidth=1)

    for node_id, (x_pos, y_pos) in positions.items():
        ax.text(
            x_pos,
            y_pos,
            labels[node_id],
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.35"},
        )

    ax.set_title(f"Iteración {iteration + 1}")
    ax.axis("off")
    return ax

def _leaf_ids_for_tree(tree, X_values):
    """Return the leaf id reached by each row in X_values for a TreePredictor."""
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