import pandas as pd

def analyze_travel_behavior(user_data):
    # If 'person_id' is the index after grouping, access it differently:
    if 'person_id' in user_data.columns:
        person_id = user_data['person_id'].iloc[0]
    else:
        person_id = user_data.index[0]  # Assuming 'person_id' is now the index after groupby

    user_stats = {
        "person_id": person_id,
        "age": user_data["age"].iloc[0],
        "female": user_data["female"].iloc[0],
        "total_trips": user_data.shape[0]
    }

    # Initializing counters
    for d in range(1, 6):
        for mode in ["drive", "pt", "cycle", "walk"]:
            user_stats[f'd{d}_{mode}'] = 0

    for i in range(len(user_data)):
        distance = user_data["distance"].iloc[i]
        mode = user_data["travel_mode"].iloc[i]
        if 0.5 <= distance < 2:
            user_stats[f'd1_{mode}'] += 1
        elif 2 <= distance < 4:
            user_stats[f'd2_{mode}'] += 1
        elif 4 <= distance < 6:
            user_stats[f'd3_{mode}'] += 1
        elif 6 <= distance < 8:
            user_stats[f'd4_{mode}'] += 1
        elif distance >= 8:
            user_stats[f'd5_{mode}'] += 1

    # Calculate probability distributions for each distance range
    for distance_range in ["d1", "d2", "d3", "d4", "d5"]:
        total_trips_in_range = sum(user_stats[f"{distance_range}_{mode}"] for mode in ["drive", "pt", "cycle", "walk"])
        if total_trips_in_range > 0:  # Avoid division by zero
            for mode in ["drive", "pt", "cycle", "walk"]:
                user_stats[f"{distance_range}_{mode}"] /= total_trips_in_range

    return user_stats

# Load data and apply function
data = pd.read_csv("data/LTDS_dataset_km.csv")
user_travel_data = data.groupby("person_id")
user_travel_stats = user_travel_data.apply(analyze_travel_behavior)

# Save results to CSV
filename = "data/LTDS_dataset_processed.csv"
user_travel_stats.to_csv(filename, index=False)
print(f"Travel behavior analysis results saved to: {filename}")
