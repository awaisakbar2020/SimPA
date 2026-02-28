import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Read the CSV file
filename = "data/LTDS_dataset_refined.csv"
#filename = "data/data_imputed.csv"
data = pd.read_csv(filename)

# Extract relevant columns for clustering
clustering_data = data.drop(columns=['person_id', 'age', 'female','total_trips'])

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 40):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 40), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Based on the elbow curve, choose the optimal number of clusters
optimal_num_clusters = 4  

# Perform KMeans clustering
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add cluster labels to the original DataFrame
data['cluster'] = clusters

# Print the cluster centers
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
cluster_centers_df = pd.DataFrame(cluster_centers, columns=clustering_data.columns)
print("Cluster Centers:")
print(cluster_centers_df)

# Save the clustered data to a new CSV file
output_filename = "data/LTDS_clustered_data.csv"
data.to_csv(output_filename, index=False)

print(f"Clustered data saved to: {output_filename}")
