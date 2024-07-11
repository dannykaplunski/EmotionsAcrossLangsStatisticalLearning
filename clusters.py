from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd

drop_columns = ['file_name', 'sample_rate', 'bit_depth', 'gender', 'emotion']
# Calculate centroids of the clusters
def calculate_centroids(data, labels):
    centroids = []
    for label in np.unique(labels):
        cluster_points = data[labels == label]
        centroids.append(cluster_points.mean(axis=0))
    return np.array(centroids)

# create the hierarchical clustering model features and save the centroids
def create_hierarchical_clustering_model(df):
    #drop the columns that are not features for that model
    new_df = df.copy().drop(columns=drop_columns + ['language'])

    linkage_matrix = linkage(new_df, method='ward', metric='euclidean')
    # create clusters
    five_clusters = fcluster(linkage_matrix, 5, criterion='maxclust')
    #create centroids
    centroids_5_clusters = calculate_centroids(new_df, five_clusters)
    #create features
    df['cluster_by_hierarcal_5_features'] = [cluster - 1 for cluster in five_clusters]
    return df, centroids_5_clusters

# Function to assign new observation
def assign_cluster(new_observation, centroids):
    distances = cdist([new_observation], centroids)
    return np.argmin(distances) + 1

def create_kmeans_model(df):

    df_with_language = df.copy().drop(columns=drop_columns )
    df_without_language = df_with_language.copy().drop(columns='language')
    df_with_language = pd.get_dummies(df_with_language, columns=['language'])
    kmeans_kwargs = {
        "init": "random",
        "n_init": 40,
        "max_iter": 300
    }
    sse,sse2 = [],[]
# A list holds the silhouette coefficients for each k
    tryks = 15 #max number of clusters to try

    for k in range(2, tryks):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_with_language)
        sse.append(kmeans.inertia_)

    for k in range(2, tryks):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(df_without_language)
        sse2.append(kmeans.inertia_)
    
    print("Length of x:", len(range(2, tryks)))
    print("Length of y:", len(sse))
    #find elbow mathematically:
    kl = KneeLocator(
        range(2, tryks),  # This range should generate indices for each value in `sse`
        sse,              # SSE values for each k
        curve="convex",
        direction="decreasing"
    )

    kelbow = kl.elbow
    kl2 = KneeLocator(
        range(2, tryks), sse2, curve="convex", direction="decreasing"
    )
    kelbow2 = kl2.elbow
    #create kmeans cluster algorithms
    kmeanselbow = KMeans(n_clusters=kelbow, **kmeans_kwargs)
    elbowclusters = kmeanselbow.fit_predict(df_with_language)

    kmeanselbow2 = KMeans(n_clusters=kelbow2, **kmeans_kwargs)
    elbowclusters2 = kmeanselbow2.fit_predict(df_without_language)

    #assigm clusters to the dataframe
    df['kmeans_wlang_elbow'] = elbowclusters
    df['kmeans_wolang_elbow'] = elbowclusters2
    return df, elbowclusters, elbowclusters2
