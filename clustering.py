import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data(filepath):
    result = []
    # open file in read mode and append each row to the list
    with open(filepath, mode = 'r') as file:
        csvFile = csv.DictReader(file)
        for row in csvFile:
            result.append(row)
    return result

def calc_features(row):
    # manually convert each column to float and save
    x1 = float(row['Population'])
    x2 = float(row['Net migration'])
    x3 = float(row['GDP ($ per capita)'])
    x4 = float(row['Literacy (%)'])
    x5 = float(row['Phones (per 1000)'])
    x6 = float(row['Infant mortality (per 1000 births)'])

    return np.array([x1, x2, x3, x4, x5, x6], dtype=np.float64)

def hac(features):
    n = len(features)
    cluster_index = dict()
    index = 0

    for i in features:
        cluster_index[index] = [i]
        index += 1

    #resulting array
    Z = np.zeros((n-1, 4), dtype=np.float64)

    # maintain a distance maitrix to avoid recalculating distances!
    distance_matrix = np.zeros((n, n))
    #calculate the matrix
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(cluster_index[i][0] - cluster_index[j][0])
            distance_matrix[j, i] = distance_matrix[i, j]
    
    for iteration in range(1, n):
        # find distances between each point and find the closest one
        closest_value = np.inf
        closest_cluster = [0, 0]
        for i in range(n-1 + iteration):
            for j in range(i+1, n-1 + iteration):
                # update closest
                if distance_matrix[i, j] > 0 and distance_matrix[i, j] < closest_value:
                    closest_value = distance_matrix[i, j]
                    closest_cluster = [i, j]

        Z[iteration-1, 0] = min(closest_cluster)
        Z[iteration-1, 1] = max(closest_cluster)
        Z[iteration-1, 2] = closest_value

        # create and add the new merged cluster
        new_cluster = list()
        for i in cluster_index[closest_cluster[0]]:
            new_cluster.append(i)
        for i in cluster_index[closest_cluster[1]]:
            new_cluster.append(i)
        cluster_index[n-1 + iteration] = new_cluster  # Store the new cluster

        # add the # of elements in the new cluster
        Z[iteration-1, 3] = len(cluster_index[n - 1 + iteration])

        # update the distance_matrix dimensions
        new_matrix = np.zeros((n+iteration, n+iteration))

        # copy the values in distance matrix into the new one
        for i in range(n - 1 + iteration):
            for j in range(n - 1 + iteration):
                new_matrix[i, j] = distance_matrix[i, j]

        # delte the rows and columns of the previouse clusters
        for i in range(n+iteration):
            # delete the column value
            new_matrix[i, min(closest_cluster)] = -1
            new_matrix[i, max(closest_cluster)] = -1
            # delete the row value
            new_matrix[min(closest_cluster), i] = -1
            new_matrix[max(closest_cluster), i] = -1

        # add the last column
        for i in range(n - 1 + iteration):
            # ignore the removed rows
            if new_matrix[i, i] != -1:
                # find the single linkage distance
                shortest_distance = np.inf
                for d1 in cluster_index[n - 1 + iteration]:
                    for d2 in cluster_index[i]:
                        distance = np.linalg.norm(d2 - d1)
                        if distance < shortest_distance:
                            shortest_distance = distance
                new_matrix[i, n-1+iteration] = shortest_distance
                new_matrix[n-1+iteration, i] = shortest_distance
            else:
                new_matrix[i, n-1+iteration] = -1
                new_matrix[n-1+iteration, i] = -1

        # add the last row value
        new_matrix[n-1+iteration, n-1+iteration] = 0

        distance_matrix = new_matrix

    return Z

def fig_hac(Z, names):
    # set up the plot
    fig = plt.figure()
    dendrogram(Z, labels=names, leaf_rotation=90)
    plt.tight_layout()
    return fig

def normalize_features(features):
    # convert to 2D NumPy array
    feature_matrix = np.array(features)
    # get rows and columns
    num_samples, num_features = feature_matrix.shape

    # initialize arrays to hold the min and max values for each feature
    col_min = np.zeros(num_features)
    col_max = np.zeros(num_features)

    # calculate the minimum and maximum for each feature (column)
    for feature_index in range(num_features):
        col_min[feature_index] = np.min(feature_matrix[:, feature_index])
        col_max[feature_index] = np.max(feature_matrix[:, feature_index])
    normalized_features = []

    # normalize each feature vector
    for sample_index in range(num_samples):
        current_vector = feature_matrix[sample_index]
        normalized_vector = np.zeros(num_features)
        # normalize each element in the current feature vector
        for feature_index in range(num_features):
            normalized_value = (current_vector[feature_index] - col_min[feature_index]) / (col_max[feature_index] - col_min[feature_index])
            normalized_vector[feature_index] = normalized_value
        normalized_features.append(normalized_vector)

    # convert back to NumPy array
    return [np.array(vector) for vector in normalized_features]