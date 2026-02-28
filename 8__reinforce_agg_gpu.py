import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from policy_network import PolicyNetwork
import matplotlib.pyplot as plt

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define environment for reward generation (user preferences)
class TravelEnv:
    def __init__(self, data):
        self.data = data
        self.num_states = len(data)
    
    def get_state(self, idx):
        return torch.tensor(self.data[idx, :3], dtype=torch.float32, device=device)  # First 3 columns as input features
    
    def get_user_preference(self, idx):
        return np.argmax(self.data[idx, 3:])  # Last 4 columns as user preferences (highest probability)

    def reward(self, action, user_pref):
        return 1 if action == user_pref else -1  # Reward +1 for match, -1 for no match

# REINFORCE algorithm with early stopping and tracking loss
def reinforce(env, policy_network, optimizer, num_episodes=3000, gamma=0.99, patience=10, threshold=1e-2):
    episode_rewards = []  # Store total rewards per episode for convergence check
    episode_losses = []  # Store loss per episode
    best_avg_reward = -float('inf')  # Best average reward seen so far
    patience_counter = 0  # Tracks how long we've gone without improvement

    results = []  # To store results per episode (for saving to file)

    for episode in range(num_episodes):
        log_probs = []
        rewards = []
        total_reward = 0  # Track total reward in each episode
        total_loss = 0  # Track total loss in each episode

        for idx in range(env.num_states):  # Each state in the dataset
            state = env.get_state(idx)
            user_pref = env.get_user_preference(idx)

            # Get action probabilities from the policy network
            action_probs = policy_network(state)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()  # Sample an action from the distribution

            # Log the action probabilities and reward
            log_probs.append(action_dist.log_prob(action))
            reward = env.reward(action.item(), user_pref)
            rewards.append(reward)
            total_reward += reward  # Sum total rewards

        # Compute returns (discounted rewards)
        returns = []
        G = 0
        for reward in reversed(rewards):
            G = reward + gamma * G
            returns.insert(0, G)  # Insert discounted reward at the beginning

        returns = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            loss = -log_prob * G
            policy_loss.append(loss)
            total_loss += loss.item()

        optimizer.zero_grad()

        # Use torch.stack() instead of torch.cat()
        policy_loss = torch.stack(policy_loss).sum()
        
        policy_loss.backward()
        optimizer.step()

        # Append total reward and loss for this episode
        episode_rewards.append(total_reward)
        episode_losses.append(policy_loss.item())  # Append the total loss for the episode

        if (episode + 1) % 100 == 0:
            avg_reward = sum(episode_rewards[-100:]) / 100
            avg_loss = sum(episode_losses[-100:]) / 100
            print(f"Episode {episode+1}/{num_episodes}, Total Reward: {total_reward}, "
                  f"Avg Reward (last 100): {avg_reward}, Avg Loss (last 100): {avg_loss}")
            
            # Append to results list
            results.append([episode+1, total_reward, avg_reward, avg_loss])

            # Check for convergence based on average reward improvement
            if avg_reward > best_avg_reward + threshold:
                best_avg_reward = avg_reward  # Update best average reward
                patience_counter = 0  # Reset patience counter since we've improved
            else:
                patience_counter += 1  # Increment patience counter

            # Early stopping check: stop if no improvement for `patience` checks
            if patience_counter >= patience:
                print(f"Converged after {episode+1} episodes with Avg Reward: {best_avg_reward}")
                break

    return results, episode_rewards, episode_losses

def process_files():
    folder_path = 'data/alpha_u/train_a2/'  # Path to training files
    user_ids_df = pd.read_csv('data/diverse_users.csv')  # Load user IDs
    user_ids = set(user_ids_df['user_id'].astype(str))  # Convert to a set of strings for faster lookup
    
    # Filter files based on user IDs in the 'user_id' column
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and f.split('.')[0][1:] in user_ids]
    
    aggregated_results = []
    
    for file in files:
        print(f"Processing file: {file}")
        df = pd.read_csv(os.path.join(folder_path, file))  # Read each file
        data = df.values  # Convert to numpy array

        # Initialize environment and policy network
        env = TravelEnv(data)
        policy_network = PolicyNetwork(input_size=3, output_size=4).to(device)  # Move to GPU if available

        # Load pretrained model (if available)
        pretrained_model_path = "models/alpha1_c2_best_model.pth"
        try:
            policy_network.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print(f"Loaded pretrained model from {pretrained_model_path}")
        except FileNotFoundError:
            print(f"No pretrained model found at {pretrained_model_path}, starting from scratch.")

        # Initialize optimizer
        optimizer = optim.Adam(policy_network.parameters(), lr=0.001)

        # Run the REINFORCE algorithm
        results, episode_rewards, episode_losses = reinforce(env, policy_network, optimizer, num_episodes=3000)

        # Append results for this file
        aggregated_results.append({
            'file': file,
            'results': results,
            'rewards': episode_rewards,
            'losses': episode_losses
        })

        # Save individual results for this file
        results_dir = "data/training/a2_500s/"
        os.makedirs(results_dir, exist_ok=True)  # Ensure the directory exists
        results_df = pd.DataFrame(results, columns=['Episode', 'Total Reward', 'Avg Reward', 'Avg Loss'])
        results_df.to_csv(f"{results_dir}training_results_{file}.csv", index=False)
        print(f"Results saved for {file}")

        # Save the policy network after training
        models_dir = "models/a2_500s/"
        os.makedirs(models_dir, exist_ok=True)  # Ensure the directory exists
        torch.save(policy_network.state_dict(), f"{models_dir}trained_policy_{file}.pth")
        print(f"Policy network saved for {file}")
    
    return aggregated_results

if __name__ == "__main__":
    aggregated_results = process_files()
    print("All files processed.")
