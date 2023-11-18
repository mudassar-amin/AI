import numpy as np
import random
import matplotlib.pyplot as plt


# Initialize the grid
grid_size = (10, 10)
rewards = np.zeros(grid_size)
rewards[0, 8] = -500
rewards[8, 6] = -100
rewards[5, 9] = 1000

# Define illegal positions and terminal positions
illegal_positions = {(1, 2), (2, 2), (3, 2), (4, 2), (3, 6), (4, 6), (5, 6), (6, 6), (5, 7), (6, 7)}
terminal_pos = {(5, 9)}

# Initialize a shared Q-matrix for the bounty hunter and the assistant
num_states = np.prod(grid_size)
num_actions = 5  # Add a "Wait" action
Q_shared = np.random.rand(num_states, num_actions)

# Learning parameters
learning_rate = 0.9
discount = 0.75
epochs = 3000
convergence_threshold = 1e-6

start_positions = [(0, i) for i in range(2)] + [(9, i) for i in range(2)]

error_rates = []
convergence_time = None
Q_values_per_episode = []

# Probability of bank robber moving to the new safe house
f2_house = 0.35

# Epsilon-greedy parameters
epsilon = 0.01  # Adjust this value as needed

# Lists to store Q-values for each episode
Q_values_per_episode = []

# Run episodes
for epoch in range(epochs):
    start_pos = random.choice(start_positions)
    start_pos2 = random.choice(start_positions)
    hunter_pos = start_pos
    asist_pos = start_pos2
    robber_position = (5, 9)  # Initial position of the robber
    Q_values_per_episode.append(np.copy(Q_shared))
    
    total_reward = 0

    while hunter_pos not in terminal_pos or asist_pos not in terminal_pos:

        robber_position = (8, 6)  # Reset robber's position at the start of each episode

        if hunter_pos == robber_position or asist_pos == robber_position:
            total_reward += 1000  # Reward for catching the robber
            break

        # Calculate the current_state_q_index before the action selection
        current_state_q_index1 = hunter_pos[0] * grid_size[1] + hunter_pos[1]
        current_state_q_index2 = asist_pos[0] * grid_size[1] + asist_pos[1]

        # Epsilon-greedy action selection for both agents
        if random.random() < epsilon:
            move_hunter = random.choice(range(5))  # Choose a random action
            move_assist = random.choice(range(5))
        else:
            move_hunter = np.argmax(Q_shared[current_state_q_index1])  # Choose the action with the highest Q-value
            move_assist = np.argmax(Q_shared[current_state_q_index2])

        # Update the positions based on the selected actions
        actions = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]  # Up, Left, Down, Right, Wait
        new_poshunter = (hunter_pos[0] + actions[move_hunter][0], hunter_pos[1] + actions[move_hunter][1])

        # Check if new position for the hunter is valid
        if (
            new_poshunter[0] < 0 or new_poshunter[0] >= grid_size[0] or
            new_poshunter[1] < 0 or new_poshunter[1] >= grid_size[1] or
            new_poshunter in illegal_positions
        ):
            new_poshunter = hunter_pos

        new_pos_assistant = (asist_pos[0] + actions[move_assist][0], asist_pos[1] + actions[move_assist][1])

        # Check if new position for the assistant is valid
        if (
            new_pos_assistant[0] < 0 or new_pos_assistant[0] >= grid_size[0] or
            new_pos_assistant[1] < 0 or new_pos_assistant[1] >= grid_size[1] or
            new_pos_assistant in illegal_positions
        ):
            new_pos_assistant = asist_pos

        # Store the Q-values for the current state-action pairs
        Q_values1 = Q_shared[current_state_q_index1][move_hunter]
        Q_values2 = Q_shared[current_state_q_index2][move_assist]

        # Bank robber's move
        if random.random() < f2_house:
            # Move to the new safe house in F2
            robber_position = (5, 2)
        else:
            # Move randomly within the grid (excluding illegal positions)
            possible_moves = [
                (robber_position[0] - 1, robber_position[1]),
                (robber_position[0], robber_position[1] - 1),
                (robber_position[0] + 1, robber_position[1]),
                (robber_position[0], robber_position[1] + 1),
            ]
            possible_moves = [
                move
                for move in possible_moves
                if (
                    move[0] >= 0
                    and move[0] < grid_size[0]
                    and move[1] >= 0
                    and move[1] < grid_size[1]
                    and move not in illegal_positions
                )
            ]
            
            robber_position = random.choice(possible_moves)

        # Check if either the bounty hunter or the assistant captures the robber
        if hunter_pos == robber_position:
            break

        if asist_pos == robber_position:
            break

        # Calculate the new state's Q-value
        new_state_reward1 = rewards[new_poshunter[0], new_poshunter[1]]
        new_state_reward2 = rewards[new_pos_assistant[0], new_pos_assistant[1]]

        # Update the shared Q-values for both the bounty hunter and the assistant
        Q_shared[current_state_q_index1][move_hunter] += learning_rate * (
            new_state_reward1
            + discount * np.max(Q_shared[new_poshunter[0] * grid_size[1] + new_poshunter[1]])
            - Q_values1
        )

        Q_shared[current_state_q_index2][move_assist] += learning_rate * (
            new_state_reward2
            + discount * np.max(Q_shared[new_pos_assistant[0] * grid_size[1] + new_pos_assistant[1]])
            - Q_values2
        )

        # Update the positions of the bounty hunter and the assistant
        hunter_pos = new_poshunter
        asist_pos = new_pos_assistant

    # Record Q-values for this episode
    Q_values_per_episode.append(Q_shared[current_state_q_index1])

    if epoch > 0:
        # Calculate error rate
        error_rate = np.mean(np.abs(Q_shared - Q_shared_old))
        error_rates.append(error_rate)

        if error_rate < convergence_threshold and convergence_time is None:
            convergence_time = epoch

    Q_shared_old = np.copy(Q_shared)

# Check if either the bounty hunter or the assistant caught the robber after learning
if hunter_pos == robber_position or asist_pos == robber_position:
    print("The robber has been caught!")
else:
    print("The robber got away.")

# Plot the evolution of error rate
plt.plot(error_rates, color='green', linestyle='-', label='Cooperative Hunt')
plt.title("Evolvement of Error Rate")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")

if convergence_time is not None:
    plt.axvline(x=convergence_time, color='red', linestyle='--', label=f'Convergence at Epoch {convergence_time}')
plt.legend()
plt.show()




