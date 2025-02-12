import torch
import torch.nn.functional as F
import numpy as np
from simple_llm import SimpleDecoderLLM, SimpleTokenizer
import gymnasium as gym
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

class PPO:
    """Proximal Policy Optimization (PPO) implementation for discrete action spaces.
    
    This implementation includes:
    - Generalized Advantage Estimation (GAE)
    - Value function clipping
    - Entropy bonus for exploration
    - Support for batched episodes
    - Early stopping on success rate
    
    Attributes:
        device: torch.device - CPU or CUDA device for computations
        model: SimpleDecoderLLM - The policy/value network
        optimizer: torch.optim.Adam - Optimizer for model parameters
        gamma: float - Discount factor for future rewards [0, 1]
        epsilon: float - PPO clipping parameter
        c1: float - Value loss coefficient
        c2: float - Entropy coefficient
        batch_size: int - Number of episodes to collect before updating
    """
    
    def __init__(
        self, 
        model: SimpleDecoderLLM,
        batch_size: int = 64,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1,
        c2: float = 0.01,    ):
        """Initialize PPO agent.
        
        Args:
            model: Policy/value network (outputs action logits and state values)
            batch_size: Number of episodes to collect before updating
            lr: Learning rate for Adam optimizer
            gamma: Discount factor for future rewards
            epsilon: PPO clipping parameter
            c1: Value loss coefficient
            c2: Entropy coefficient for exploration bonus
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        self.batch_size = batch_size

    def compute_gae(
        self, 
        rewards: List[float], 
        values: List[float], 
        gamma: float = 0.99, 
        lambda_: float = 0.95
    ) -> List[float]:
        """Compute Generalized Advantage Estimation (GAE).
        
        GAE blends between TD(0) advantage and full Monte Carlo advantage:
        - lambda=0: TD(0) advantage (low variance, some bias)
        - lambda=1: Monte Carlo advantage (high variance, no bias)
        
        Args:
            rewards: List of episode rewards [r_0, r_1, ..., r_T]
            values: List of state values [V(s_0), V(s_1), ..., V(s_T)]
            gamma: Discount factor for future rewards
            lambda_: GAE parameter for advantage estimation
        
        Returns:
            advantages: List of GAE advantage estimates [A_0, A_1, ..., A_T]
            
        Note:
            For the last step T, next_value is 0 since episode has ended
            (either reached goal, hole, or timeout)
        """
        advantages = []
        gae = 0
        
        # Go backwards through the non-padded rewards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # For last real step, next value is 0 if episode ended
                next_value = 0
            else:
                next_value = values[t + 1]
                
            # TD error: δ_t = r_t + γV(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
            
            # GAE: A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
            gae = delta + gamma * lambda_ * gae
            advantages.insert(0, gae)
            
        return advantages
    
    def construct_states_from_episodes(self, episodes: List[List[int]]) -> List[List[int]]:
        """Construct states from episodes.
        Note that for the transformer model, the states are the BOS token and the current action.
        So if episode is [up, down, right], the states are [[BOS], [BOS, up], [BOS, up, down]].
        Notice that the last action is predicted but is not part of the state input.
        
        Args:
            episodes: List of episodes, each containing a sequence of actions
            
        Returns:
            states: List of states, each containing a sequence of states
        """
        states = []
        for episode in episodes:
            # add BOS token
            states.append([self.model.tokenizer.bos_token_id] + episode[:-1])
        return states

    def collect_batch(self, env) -> List[Dict]:
        """Collect a batch of episodes using current policy.
        
        Samples batch_size sequences from the model and evaluates them
        in the environment. Stores results for policy updates.
        
        Args:
            env: Gymnasium environment (FrozenLake-v1)
            
        Returns:
            results: List of episode results, each containing:
                - episode: List[int] - Token sequence including BOS
                - length: int - Episode length
                - reward: float - Total episode reward
                - rewards: List[float] - Step-wise rewards
                - terminated: bool - Whether episode ended naturally
                
        Note:
            Also stores internally:
            - self.episodes: List[List[int]] - Token sequences
            - self.episode_log_probs: torch.Tensor - Action log probabilities
            - self.episode_values: torch.Tensor - State values
            - self.episode_rewards: List[List[float]] - Episode rewards
        """
        # Sample batch_size sequences from the model
        samples = self.model.sample_actions(num_sequences=self.batch_size)
        
        # Evaluate all sequences in the environment
        results = self.evaluate_rollouts(env, samples['action_sequences'])
        
        # Align samples with actual episodes
        self.align_samples_with_episodes(samples, results)

        return results

    def pad_sequences(
        self, 
        sequences: List[List[int]], 
        padding_value: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad sequences to same length and create attention mask.
        
        Args:
            sequences: List of variable length sequences
            padding_value: Value to use for padding (default: 0)
            
        Returns:
            padded_sequences: torch.Tensor [batch_size, max_len] - Padded sequences
            attention_mask: torch.Tensor [batch_size, max_len] - Mask (0.0 for real tokens, -inf for padding)
            
        Note:
            Attention mask is used to prevent attention to padded tokens
        """
        # Find max length in batch
        max_len = max(len(seq) for seq in sequences)
        
        # Create attention mask (0.0 for real tokens, -inf for padding)
        attention_mask = torch.tensor([
            [0.0] * len(seq) + [float('-inf')] * (max_len - len(seq))
            for seq in sequences
        ], device=self.device)
        
        # Pad sequences with padding token
        padded_sequences = torch.tensor([
            seq + [padding_value] * (max_len - len(seq))
            for seq in sequences
        ], device=self.device)
        
        return padded_sequences, attention_mask
    
    def normalize_advantages(
        self, 
        advantages: torch.Tensor, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """Normalize advantages across the batch, ignoring padded values.
        
        Args:
            advantages: torch.Tensor [batch_size, max_len] - Advantage estimates
            mask: torch.Tensor [batch_size, max_len] - Boolean mask (True for real values)
            
        Returns:
            normalized_advantages: torch.Tensor [batch_size, max_len] - Normalized advantages
            
        Note:
            Normalization helps stabilize training by making advantages have
            zero mean and unit variance across the batch.
        """
        valid_advantages = advantages[mask]  # Select only non-padded values
        mean = valid_advantages.mean()
        std = valid_advantages.std(unbiased=False) + 1e-8  # Prevent division by zero

        normalized_advantages = (advantages - mean) / std
        normalized_advantages[~mask] = 0  # Keep padded values as zero
        return normalized_advantages

    def compute_advantages_and_returns(
        self
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute advantages and returns for all episodes in batch.
        
        Returns:
            padded_advantages: torch.Tensor [batch_size, max_len] - GAE advantages
            padded_returns: torch.Tensor [batch_size, max_len] - Discounted returns
            
        Note:
            - Advantages are normalized across the batch
            - Returns are used for value function training
            - Both are padded to max episode length
        """
        advantages_list = []
        returns_list = []
        
        # First compute all advantages
        for rewards, values in zip(self.episode_rewards, self.episode_values):
            
            # Compute advantages for this episode
            advantages = self.compute_gae(rewards, values)
            returns = [a + v for a, v in zip(advantages, values)]
            
            advantages_list.append(advantages)
            returns_list.append(returns)
        
        # Pad advantages and get mask
        padded_advantages, attention_mask = self.pad_sequences(
            advantages_list,
            padding_value=0.0
        )
        
        # Convert attention mask to boolean mask (0.0 -> True, -inf -> False)
        mask = (attention_mask == 0.0)
        
        # Normalize advantages using the mask
        padded_advantages = self.normalize_advantages(padded_advantages, mask)
        
        # Pad returns (no normalization needed)
        padded_returns, _ = self.pad_sequences(
            returns_list,
            padding_value=0.0
        )
        
        return padded_advantages, padded_returns

    def compute_losses(
        self, 
        padded_advantages: torch.Tensor,  # [batch_size, max_len]
        padded_returns: torch.Tensor      # [batch_size, max_len]
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO losses for policy and value functions.
        
        Args:
            padded_advantages: Normalized GAE advantages [batch_size, max_len]
            padded_returns: Discounted returns [batch_size, max_len]
            
        Returns:
            losses: Dictionary containing:
                - actor_loss: Policy loss with PPO clipping
                - critic_loss: Value function MSE loss
                - entropy: Policy entropy (for exploration)
                - total_loss: Combined loss for optimization
                
        Note:
            Uses stored episode data (self.episodes, self.episode_log_probs)
            to compute policy ratio for PPO clipping.
        """
        # Pad sequences with PAD token
        padded_sequences, attention_mask = self.pad_sequences(
            self.states, 
            padding_value=self.model.tokenizer.pad_token_id
        )
        
        # Forward pass with attention mask
        action_logits, values = self.model(
            padded_sequences,
            attention_mask
        )
        
        # Create distribution from logits
        attention_mask = attention_mask == 0.0  # convert to boolean mask
        dist = Categorical(logits=action_logits[attention_mask])
        # compute the entropy, i.e. spread of the distribution
        entropy = dist.entropy().mean()

        # Get log probs for new actions
        actions = torch.cat([
            torch.tensor(episode, device=self.device) for episode in self.episodes
        ])
        
        # get log probs for new actions
        new_log_probs = dist.log_prob(actions)
        
        
        # PPO objective with clipping
        ratio = torch.exp(new_log_probs - self.episode_log_probs)
        
        # compute scaled advantages
        masked_advantages = padded_advantages[attention_mask]
        surr1 = ratio * masked_advantages

        # compute clipped surrogate
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * masked_advantages

        # Compute individual losses
        actor_loss = -torch.min(surr1, surr2).mean()

        # compute critic loss
        critic_loss = F.mse_loss(values[attention_mask], padded_returns[attention_mask])
        
        # Total loss with coefficients
        total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy': entropy,
            'total_loss': total_loss
        }

    def update(self):
        """Perform multiple epochs of PPO updates.
        
        Computes advantages and returns, then updates policy and value
        functions using PPO clipping and value function clipping.
        
        Note:
            Performs 4 epochs of updates by default.
            Uses stored episode data from collect_batch().
        """
        # Compute advantages and returns
        padded_advantages, padded_returns = self.compute_advantages_and_returns()
        
        # Multiple epochs of updates
        for _ in range(4):
            # Compute losses
            losses = self.compute_losses(padded_advantages, padded_returns)
            
            # Update model
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            self.optimizer.step()

    def evaluate_rollouts(self, env, sequences):
        """Evaluate a batch of action sequences in the environment.
        
        Args:
            env: Gymnasium environment (FrozenLake-v1)
            sequences: List[List[int]] - Batch of action sequences to evaluate
            
        Returns:
            results: List[Dict] containing for each episode:
                - episode: List[int] - Actual token sequence (actions only)
                - length: int - Episode length
                - reward: float - Total episode reward
                - rewards: List[float] - Step-wise rewards
                - terminated: bool - Whether episode ended naturally
        """
        results = []
        
        for sequence in sequences: 
            # Run this sequence in environment
            state, _ = env.reset()
            episode_length = 0
            total_reward = 0
            terminated = False
            step_rewards = []
            actual_episode = []
            
            for action in sequence:
                    
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                if reward == 1:  # Reached goal
                    # Base reward for finding goal is 1.0
                    # Additional bonus up to 0.5 for efficiency
                    # Shorter episodes get bigger bonus
                    efficiency = 1.0 / max(episode_length, 1)  # Prevent division by zero
                    reward = 1.0 + (0.5 * efficiency)  # Range: [1.0, 1.5]
                
                total_reward += reward
                episode_length += 1
                step_rewards.append(reward)
                actual_episode.append(action)
                state = next_state
                
                if terminated or truncated:
                    break
            
            results.append({
                'episode': actual_episode,
                'length': episode_length,
                'reward': total_reward,
                'rewards': step_rewards,
                'terminated': terminated
            })
        
        return results

    def align_samples_with_episodes(self, samples: Dict, results: List[Dict]) -> None:
        """Align sampled log_probs and values with actual episodes.
        
        Args:
            samples: Dict containing log_probs and values from model sampling
            results: List of episode results from environment evaluation
            
        Note:
            Updates self.episode_log_probs, self.episode_values, and self.episode_rewards
            to match actual episode lengths (which may be shorter due to early termination)
        """
        # Align log_probs and values with actual episodes
        self.episode_log_probs = []
        self.episode_values = []
        
        for i, result in enumerate(results):
            episode_length = len(result['episode'])
            # Take only the log_probs and values up to actual episode length
            self.episode_log_probs.append(samples['log_probs'][i][:episode_length])
            self.episode_values.append(samples['values'][i][:episode_length])
        
        # Convert to tensors
        self.episode_log_probs = torch.cat([
            torch.tensor(l, device=self.device) for l in self.episode_log_probs
        ])
        
        # rewards
        self.episode_rewards = [result['rewards'] for result in results]
        # episodes
        self.episodes = [result['episode'] for result in results]
        # states
        self.states = self.construct_states_from_episodes(self.episodes)

def train_ppo():
    """Train a PPO agent on FrozenLake environment.
    
    Returns:
        model: Trained SimpleDecoderLLM model
        
    Note:
        - Uses deterministic FrozenLake (is_slippery=False)
        - Implements early stopping when success rate > 90%
        - Prints training statistics every iteration
    """
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Initialize environment, tokenizer and model
    env = gym.make("FrozenLake-v1", is_slippery=False)
    tokenizer = SimpleTokenizer()
    model = SimpleDecoderLLM(tokenizer)
    ppo = PPO(model, batch_size=256)
    
    num_iterations = 1000
    for iteration in range(num_iterations):
        # Collect rollouts
        print(f"Iteration {iteration}")
        results = ppo.collect_batch(env)
        
        # Calculate episode statistics
        num_episodes = len(results)
        avg_episode_length = sum(r['length'] for r in results) / num_episodes
        
        # Count successes (episodes ending with reward 1, meaning goal reached)
        num_successes = sum(1 for r in results if r['rewards'][-1] >= 1)
        success_rate = num_successes / num_episodes
        
        print(f"Iteration {iteration}:")
        print(f"  Total reward: {sum(r['reward'] for r in results)}")
        print(f"  Average episode length: {avg_episode_length:.1f}")
        print(f"  Success rate: {success_rate:.1%} ({num_successes}/{num_episodes})")
        
        # Early stopping
        if success_rate > 0.9:
            print("Success rate > 90%, stopping training!")
            break
            
        # Update policy
        ppo.update()
    
    return model

def visualize_path(model, env):
    """Visualize a single path through the environment using matplotlib.
    
    Args:
        model: Trained SimpleDecoderLLM model
        env: FrozenLake environment
        
    Returns:
        path: List[int] - Sequence of states visited
        reward: float - Total reward achieved
        
    Note:
        - Draws grid with start (S), goal (G), and holes (H)
        - Shows arrows for each action taken
        - Numbers indicate step sequence
    """
    import matplotlib.pyplot as plt
    
    state, _ = env.reset()
    done = False
    total_reward = 0
    path = []
    actions = []
    
    # Get initial state
    current_pos = state
    path.append(current_pos)
    
    # Start with BOS token
    sequence = [model.tokenizer.bos_token_id]
    device = next(model.parameters()).device
    
    while not done:
        # Get model prediction
        input_ids = torch.tensor([sequence], device=device)
        attention_mask = torch.zeros(1, len(sequence), device=device)
        
        with torch.no_grad():
            action_logits, _ = model(input_ids, attention_mask)
            action = action_logits[0, -1].argmax().item()
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update sequence and path
        sequence.append(action)
        current_pos = next_state
        path.append(current_pos)
        actions.append(action)
        total_reward += reward
        
        if len(path) > 100:  # Safety break
            break
    
    # Create a figure
    plt.figure(figsize=(8, 8))
    
    # Draw the grid
    for i in range(5):
        plt.axhline(y=i, color='gray', linestyle='-', alpha=0.5)
        plt.axvline(x=i, color='gray', linestyle='-', alpha=0.5)
    
    # Mark special cells
    plt.text(0.5, 3.5, 'S', ha='center', va='center', fontsize=20)  # Start
    plt.text(3.5, 0.5, 'G', ha='center', va='center', fontsize=20)  # Goal
    plt.text(1.5, 2.5, 'H', ha='center', va='center', fontsize=20)  # Holes
    plt.text(3.5, 2.5, 'H', ha='center', va='center', fontsize=20)
    plt.text(0.5, 0.5, 'H', ha='center', va='center', fontsize=20)
    
    # Add arrows for each action
    action_to_arrow = {
        0: ('←', (-0.2, 0)),   # Left
        1: ('↓', (0, -0.2)),   # Down
        2: ('→', (0.2, 0)),    # Right
        3: ('↑', (0, 0.2)),    # Up
    }
    
    # Convert positions to grid coordinates
    for i in range(len(path)-1):
        pos = path[i]
        action = actions[i]
        
        # Convert position to grid coordinates (flip y because matplotlib's origin is bottom-left)
        x = pos % 4
        y = 3 - (pos // 4)  # Flip y coordinate
        
        # Center of the grid cell
        cell_center = (x + 0.5, y + 0.5)
        
        # Add arrow
        arrow_symbol, (dx, dy) = action_to_arrow[action]
        plt.arrow(cell_center[0], cell_center[1], dx, dy,
                 head_width=0.1, head_length=0.1, fc='red', ec='red')
        
    
    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.title(f'Path through FrozenLake (Reward: {total_reward})')
    plt.axis('off')
    plt.show()
    
    print(f"Path positions: {path}")
    print(f"Path length: {len(path)}")
    print(f"Total reward: {total_reward}")
    
    return path, total_reward

if __name__ == "__main__":
    print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Train model
    model = train_ppo()
    
    # Visualize a path
    env = gym.make("FrozenLake-v1", is_slippery=False)  # No render mode needed
    path, reward = visualize_path(model, env)
    env.close()
