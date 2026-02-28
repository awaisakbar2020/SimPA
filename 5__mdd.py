import pandas as pd
import numpy as np
from scipy.stats import truncnorm

# Load data from CSV files
centroids_df = pd.read_csv("data/LTDS_centroids_normalised_0_1.csv")
standard_deviations_df = pd.read_csv("data/LTDS_standard_deviation_normalised_0_1.csv")

# Extract only feature columns (excluding 'cluster' and 'cluster_size')
new_features = [col for col in centroids_df.columns if col not in ['cluster', 'cluster_size']]  # All features except cluster and size
features = new_features

def generate_cluster_data_dirichlet(centroid, std_dev, num_datapoints, features):
    new_data = pd.DataFrame(index=range(num_datapoints), columns=features)
    
    for feature in features:
        if feature == 'total_distance':
            # Ensure that total_distance is always non-negative
            low = max(0, centroid[feature] - std_dev[feature])
        else:
            low = centroid[feature] - np.where(std_dev[feature] < 0, 1e-6, std_dev[feature])  # Enforce positive std_dev
        high = centroid[feature] + std_dev[feature]
        try:
            new_data[feature] = truncnorm.rvs((low - centroid[feature]) / std_dev[feature],
                                                 (high - centroid[feature]) / std_dev[feature],
                                                 loc=centroid[feature], scale=std_dev[feature], size=num_datapoints)
        except ValueError:
            print(f"Warning: ValueError encountered for feature {feature}. Using default value of 0.")
            new_data[feature] = 0  # Handle error by setting to 0 (adjust as needed)
    
    # Normalize values within each distance category (d1, d2, ..., d5) to sum to 1
    for i in range(1, 6):  # Loop through distance categories (d1 to d5)
        category_cols = [col for col in features if col.startswith(f'd{i}_')]
        new_data[category_cols] = new_data[category_cols].apply(lambda x: np.exp(x) / np.exp(x).sum(), axis=1)

    return new_data

# Get the total number of data points to generate
total_datapoints = 1000

# Get cluster sizes
cluster_sizes = centroids_df['cluster_size']

# Calculate weights based on cluster size proportions
weights = cluster_sizes / np.sum(cluster_sizes)

# Generate data for each cluster based on weights and store them in a list
cluster_data_list = []
for i in range(len(centroids_df)):
    centroid = centroids_df[features].iloc[i]
    std_dev = standard_deviations_df[features].iloc[i]
    num_datapoints_per_cluster = int(np.round(weights[i] * total_datapoints))

    # Generate data for this cluster using the updated function
    cluster_data = generate_cluster_data_dirichlet(centroid, std_dev, num_datapoints_per_cluster, features)
    cluster_data["cluster"] = i  # Add cluster label
    cluster_data_list.append(cluster_data)

# Combine data from all clusters into a single DataFrame
all_data = pd.concat(cluster_data_list, ignore_index=True)

# No distance calculation needed as percentages are replaced with probabilities

# Drop count columns (assuming they don't exist anymore)
# all_data.drop(columns=['walk_count', 'cycle_count', 'drive_count', 'pt_count'], inplace=True)

# Save the generated data to a CSV file
all_data.to_csv('data/generated_data_dirichlet.csv', index=False)
print("Generated data saved to: generated_data.csv")
