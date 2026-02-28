import pandas as pd
from sklearn.cluster import KMeans

# Load the data
data = pd.read_csv('data/synthetic_users.csv')

# Filter users from cluster 2
cluster_2_users = data[data['cluster'] == 2]

# Define the travel preference columns
travel_columns = ['d1_drive', 'd1_pt', 'd1_cycle', 'd1_walk', 
                  'd2_drive', 'd2_pt', 'd2_cycle', 'd2_walk', 
                  'd3_drive', 'd3_pt', 'd3_cycle', 'd3_walk', 
                  'd4_drive', 'd4_pt', 'd4_cycle', 'd4_walk', 
                  'd5_drive', 'd5_pt', 'd5_cycle', 'd5_walk']

# Get travel preference data for cluster 2 users
preferences = cluster_2_users[travel_columns]

# Define the number of diverse users you want to select
x=30
n_clusters = x  # Set x to the number of users you want

# Apply KMeans to group users based on travel preferences
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(preferences)

# Select the users closest to each cluster center
closest_users = []
for i in range(n_clusters):
    users_in_cluster = preferences[cluster_labels == i]
    cluster_center = kmeans.cluster_centers_[i]
    distances = ((users_in_cluster - cluster_center) ** 2).sum(axis=1)
    closest_user_idx = distances.idxmin()
    closest_users.append(closest_user_idx)

# Get the data for the selected users
diverse_users = cluster_2_users.loc[closest_users]

# Save the diverse users to a CSV file
diverse_users.to_csv('data/diverse_users.csv', index=False)

print("Diverse users saved to 'data/diverse_users.csv'")
