import torch
import numpy as np
import pandas as pd
import os
from policy_network import PolicyNetwork

# Define environment for reward generation (user preferences)
class TravelEnv:
    def __init__(self, data):
        self.data = data
        self.num_states = len(data)
    
    def get_state(self, idx):
        return torch.tensor(self.data[idx, :3], dtype=torch.float32)  # First 3 columns as input features
    
    def get_user_preference(self, idx):
        return np.argmax(self.data[idx, 3:])  # Last 4 columns as user preferences (highest probability)

    def reward(self, action, user_pref):
        return 1 if action == user_pref else -1  # Reward +1 for match, -1 for no match

def test_policy(env, policy_network):
    total_reward = 0
    correct_predictions = 0

    for idx in range(env.num_states):
        state = env.get_state(idx)
        user_pref = env.get_user_preference(idx)

        # Get action probabilities from the policy network
        action_probs = policy_network(state)
        action = torch.argmax(action_probs).item()

        # Compute reward and check for correct prediction
        reward = env.reward(action, user_pref)
        total_reward += reward
        if reward == 1:
            correct_predictions += 1

    accuracy = correct_predictions / env.num_states
    avg_reward = total_reward / env.num_states
    return accuracy, avg_reward

def test_files():
    folder_path = 'data/alpha_u/test_a1_a2/'  # Path to test datasets
    model_path = "models/alpha1_c2_best_model.pth"  # Path to the single model
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize policy network once
    policy_network = PolicyNetwork(input_size=3, output_size=4)

    # Load the pre-trained model once
    try:
        policy_network.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Model not found at {model_path}. Exiting...")
        return []

    test_results = []

    for file in files:
        print(f"Testing file: {file}")
        df = pd.read_csv(os.path.join(folder_path, file))
        data = df.values

        # Initialize environment
        env = TravelEnv(data)

        # Test the policy network
        accuracy, avg_reward = test_policy(env, policy_network)

        # Append results
        test_results.append({
            'file': file,
            'accuracy': accuracy,
            'avg_reward': avg_reward
        })

        print(f"File: {file}, Accuracy: {accuracy:.2f}, Avg Reward: {avg_reward:.2f}")
    
    # Save aggregated test results
    results_df = pd.DataFrame(test_results)
    results_df.to_csv("test_results_summary_a1.csv", index=False)
    print("Test results saved to test_results_summary_a1.csv")
    
    return test_results

if __name__ == "__main__":
    test_results = test_files()
    print("All test files processed.")
