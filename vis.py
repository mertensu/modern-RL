import matplotlib.pyplot as plt
import torch


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
        plt.axhline(y=i, color="gray", linestyle="-", alpha=0.5)
        plt.axvline(x=i, color="gray", linestyle="-", alpha=0.5)

    # Mark special cells
    plt.text(0.5, 3.5, "S", ha="center", va="center", fontsize=20)  # Start
    plt.text(3.5, 0.5, "G", ha="center", va="center", fontsize=20)  # Goal
    plt.text(1.5, 2.5, "H", ha="center", va="center", fontsize=20)  # Holes
    plt.text(3.5, 2.5, "H", ha="center", va="center", fontsize=20)
    plt.text(0.5, 0.5, "H", ha="center", va="center", fontsize=20)

    # Add arrows for each action
    action_to_arrow = {
        0: ("←", (-0.2, 0)),  # Left
        1: ("↓", (0, -0.2)),  # Down
        2: ("→", (0.2, 0)),  # Right
        3: ("↑", (0, 0.2)),  # Up
    }

    # Convert positions to grid coordinates
    for i in range(len(path) - 1):
        pos = path[i]
        action = actions[i]

        # Convert position to grid coordinates (flip y because matplotlib's origin is bottom-left)
        x = pos % 4
        y = 3 - (pos // 4)  # Flip y coordinate

        # Center of the grid cell
        cell_center = (x + 0.5, y + 0.5)

        # Add arrow
        arrow_symbol, (dx, dy) = action_to_arrow[action]
        plt.arrow(
            cell_center[0],
            cell_center[1],
            dx,
            dy,
            head_width=0.1,
            head_length=0.1,
            fc="red",
            ec="red",
        )

    plt.xlim(0, 4)
    plt.ylim(0, 4)
    plt.title(f"Path through FrozenLake (Reward: {total_reward})")
    plt.axis("off")
    plt.show()

    print(f"Path positions: {path}")
    print(f"Path length: {len(path)}")
    print(f"Total reward: {total_reward}")

    return path, total_reward
