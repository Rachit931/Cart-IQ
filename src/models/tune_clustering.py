from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


def tune_clustering(X):
    """
    Hyperparameter tuning
    for clustering models.

    Models:
    --------
    - KMeans
    - Agglomerative Clustering
    - Gaussian Mixture

    Metrics:
    --------
    - Silhouette Score
    - Calinski-Harabasz Score
    - Davies-Bouldin Score

    Parameters:
    -----------
    X:
        Scaled feature matrix

    Returns:
    --------
    best_model_name
    best_params
    """

    best_silhouette = -1

    best_model_name = None

    best_params = None

    print("\n===== CLUSTERING TUNING =====")

    # KMEANS

    for k in range(2, 11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)

        calinski = calinski_harabasz_score(
            X,
            labels,
        )

        davies = davies_bouldin_score(
            X,
            labels,
        )

        print(f"\nKMeans | k={k}")

        print(f"Silhouette: {silhouette:.4f}")

        print(f"Calinski-Harabasz: {calinski:.2f}")

        print(f"Davies-Bouldin: {davies:.4f}")

        if silhouette > best_silhouette:

            best_silhouette = silhouette

            best_model_name = "KMeans"

            best_params = {
                "n_clusters": k,
                "random_state": 42,
                "n_init": 20,
            }

    # AGGLOMERATIVE

    for k in range(2, 11):

        model = AgglomerativeClustering(
            n_clusters=k,
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)

        calinski = calinski_harabasz_score(
            X,
            labels,
        )

        davies = davies_bouldin_score(
            X,
            labels,
        )

        print(f"\nAgglomerative | k={k}")

        print(f"Silhouette: {silhouette:.4f}")

        print(f"Calinski-Harabasz: {calinski:.2f}")

        print(f"Davies-Bouldin: {davies:.4f}")

        if silhouette > best_silhouette:

            best_silhouette = silhouette

            best_model_name = "Agglomerative"

            best_params = {
                "n_clusters": k,
            }

    # GAUSSIAN MIXTURE

    for k in range(2, 11):

        model = GaussianMixture(
            n_components=k,
            random_state=42,
        )

        labels = model.fit_predict(X)

        silhouette = silhouette_score(X, labels)

        calinski = calinski_harabasz_score(
            X,
            labels,
        )

        davies = davies_bouldin_score(
            X,
            labels,
        )

        print(f"\nGaussianMixture | k={k}")

        print(f"Silhouette: {silhouette:.4f}")

        print(f"Calinski-Harabasz: {calinski:.2f}")

        print(f"Davies-Bouldin: {davies:.4f}")

        if silhouette > best_silhouette:

            best_silhouette = silhouette

            best_model_name = "GaussianMixture"

            best_params = {
                "n_components": k,
                "random_state": 42,
            }

    print("\n===== BEST MODEL =====")

    print(f"Model: {best_model_name}")

    print(f"Parameters: {best_params}")

    print(f"Best Silhouette Score: " f"{best_silhouette:.4f}")

    return (
        best_model_name,
        best_params,
    )
