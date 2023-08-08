import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import os
from ValuesDict import *

def perform_cluster():
    data_folder = "Texts"
    file_paths = []
    for filename in os.listdir(data_folder):
        file_paths.append("Texts/" + filename)

    # List of dictionaries
    dictionaries = get_dicts()
    dict_names = get_dicts_names()
    # Initialize the vector array
    num_files = len(file_paths)
    num_dicts = len(dictionaries)
    dataset = np.zeros((num_files, num_dicts))

    # Iterate over the files
    for file_index, file_path in enumerate(file_paths):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            words = text.split()

            # Iterate over the words
            for word in words:
                # Check if the word belongs to any of the dictionaries
                for dict_index, dictionary in enumerate(dictionaries):
                    if word in dictionary:
                        # Increment the corresponding element in the vector array
                        dataset[file_index, dict_index] += 1

    # Perform clustering using K-means
    num_clusters = 7  # Set the number of clusters
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(dataset)

    # Store file paths and vectors for each cluster
    clusters = [[] for _ in range(num_clusters)]
    cluster_vectors = [[] for _ in range(num_clusters)]
    for file_index, cluster_label in enumerate(cluster_labels):
        clusters[cluster_label].append(file_paths[file_index])
        cluster_vectors[cluster_label].append(dataset[file_index])

    # Print the file paths and average values for each cluster
    for cluster_index, (cluster_files, cluster_vector_list) in enumerate(zip(clusters, cluster_vectors)):
        print("Cluster", cluster_index)
        for file_path in cluster_files:
            print(file_path)

        # Calculate the average vector for the cluster
        cluster_vectors_avg = np.mean(cluster_vector_list, axis=0)
        print("Averages:")
        for dict_index, avg_value in enumerate(cluster_vectors_avg):
            parameter_name = dict_names[dict_index]
            print(parameter_name, "=", avg_value)
        print()

    # Plot the clusters
    plt.scatter(dataset[:, 0], dataset[:, 1], c=cluster_labels, cmap='viridis')
    plt.title("Vector Clustering")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

