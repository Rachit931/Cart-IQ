from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture


def train_clustering(
    X,
    model_name,
    params,
):
    """
    Train final clustering model.

    Parameters:
    -----------
    X:
        Scaled feature matrix

    model_name:
        Name of selected algorithm

    params:
        Best parameters from tuning

    Returns:
    --------
    model
    labels
    """

    print("\n===== TRAINING FINAL CLUSTERING MODEL =====")

    if model_name == "KMeans":

        model = KMeans(**params)

        labels = model.fit_predict(X)

    elif model_name == "Agglomerative":

        model = AgglomerativeClustering(**params)

        labels = model.fit_predict(X)

    elif model_name == "GaussianMixture":

        model = GaussianMixture(**params)

        model.fit(X)

        labels = model.predict(X)

    else:

        raise ValueError(f"Unsupported model: {model_name}")

    print(f"Model: {model_name}")

    print(f"Number of Clusters: " f"{len(set(labels))}")

    return (
        model,
        labels,
    )
