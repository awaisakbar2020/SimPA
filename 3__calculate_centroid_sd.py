import pandas as pd
import numpy as np  # Import NumPy for square root calculation


# Read the CSV file containing clustering results
filename = "data/LTDS_clustered_data.csv"
data = pd.read_csv(filename)

# Drop the person_id column (assuming it's not needed for calculations)
data = data.drop('person_id', axis=1)

# Group data by cluster label
clustered_data = data.groupby('cluster')


def get_cluster_centroid(cluster):
    # Get cluster data for this group (including the 'cluster' column)
    cluster_data = cluster

    # Calculate centroid (mean of each feature)
    centroid = cluster_data.mean()
    return centroid


def get_cluster_standard_deviation(cluster):
    # Get cluster data for this group
    cluster_data = cluster.copy()  # Make a copy to avoid modifying original
    # Drop the 'cluster' label before applying variance
    cluster_data = cluster_data.drop('cluster', axis=1)

    # Calculate standard deviation for each feature (square root of variance)
    standard_deviation = cluster_data.var().apply(np.sqrt)  # Apply np.sqrt to each value

    # Add 'cluster' label back using the original cluster index
    standard_deviation['cluster'] = cluster.name  # 'name' attribute holds the cluster label

    return standard_deviation


def get_cluster_size(cluster):
    # Get the size of the cluster (number of data points)
    cluster_size = len(cluster)
    return cluster_size


# Apply centroid function to each cluster group and store results in a DataFrame
centroid_df = clustered_data.apply(get_cluster_centroid)

# Apply function to get cluster size and store results in a Series
cluster_size_series = clustered_data.apply(get_cluster_size)

# Combine centroid and cluster size DataFrames into a single DataFrame
combined_df = pd.concat([centroid_df, cluster_size_series.rename('cluster_size')], axis=1)

# Save centroid and size results to a new CSV file
centroid_filename = "data/LTDS_cluster_centroids.csv"
combined_df.to_csv(centroid_filename, index=False)
print(f"Cluster centroids and sizes saved to: {centroid_filename}")

# Apply standard deviation function to each cluster group
standard_deviation_df = clustered_data.apply(get_cluster_standard_deviation)

# Save standard deviation results to a separate CSV file
standard_deviation_filename = "data/LTDS_cluster_standard_deviation.csv"
standard_deviation_df.to_csv(standard_deviation_filename, index=False)
print(f"Cluster standard deviation saved to: {standard_deviation_filename}")
