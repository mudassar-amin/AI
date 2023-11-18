import numpy as np
import random
import matplotlib.pyplot as plt

GRID_SIZE = (10, 10)
rewards = np.zeros(GRID_SIZE)
rewards[0, 8] = -500
rewards[8, 6] = -100

illegal_positions = {(1, 2), (2, 2), (3, 2), (4, 2), (3, 6), (4, 6), (5, 6), (6, 6), (5, 7), (6, 7)}

num_states = np.prod(GRID_SIZE)
num_actions = 5 
Q_shared = np.random.rand(num_states, num_actions)

learning_rate = 0.1
discount = 0.9
epochs = 10
convergence_threshold = 1e-6
epsilon = 0.1  

f2_house = 0.35
error_rates = []
convergence_time = None

Q_values_per_episode = []


for epoch in range(epochs):
    hunter_pos = (random.randint(0, GRID_SIZE[0] - 1), random.randint(0, GRID_SIZE[1] - 1))
    asist_pos = (random.randint(0, GRID_SIZE[0] - 1), random.randint(0, GRID_SIZE[1] - 1))

    while True:
        robber_position = (5, 9) 
        if hunter_pos == robber_position or asist_pos == robber_position:
            break       
        current_state_q_index1 = hunter_pos[0] * GRID_SIZE[1] + hunter_pos[1]
        current_state_q_index2 = asist_pos[0] * GRID_SIZE[1] + asist_pos[1]

        
        if random.random() < epsilon:
            move_hunter = random.choice(range(5))  
            move_assist = random.choice(range(5))
        else:
            move_hunter = np.argmax(Q_shared[current_state_q_index1])  
            move_assist = np.argmax(Q_shared[current_state_q_index2])

        
        if move_hunter == 0:  
            new_poshunter = (hunter_pos[0] - 1, hunter_pos[1])
        elif move_hunter == 1:  
            new_poshunter = (hunter_pos[0], hunter_pos[1] - 1)
        elif move_hunter == 2: 
            new_poshunter = (hunter_pos[0] + 1, hunter_pos[1])
        elif move_hunter == 3:  
            new_poshunter = (hunter_pos[0], hunter_pos[1] + 1)
        elif move_hunter == 4: 
            new_poshunter = hunter_pos

        if move_assist == 0:  
            new_pos_assistant = (asist_pos[0] - 1, asist_pos[1])
        elif move_assist == 1:  
            new_pos_assistant = (asist_pos[0], asist_pos[1] - 1)
        elif move_assist == 2:  # Down
            new_pos_assistant = (asist_pos[0] + 1, asist_pos[1])
        elif move_assist == 3:  # Right
            new_pos_assistant = (asist_pos[0], asist_pos[1] + 1)
        elif move_assist == 4:  # Wait
            new_pos_assistant = asist_pos

        # Check if new positions are valid
        if (
            new_poshunter[0] < 0 or new_poshunter[0] >= GRID_SIZE[0] or
            new_poshunter[1] < 0 or new_poshunter[1] >= GRID_SIZE[1] or
            new_poshunter in illegal_positions
        ):
            new_poshunter = hunter_pos

        if (
            new_pos_assistant[0] < 0 or new_pos_assistant[0] >= GRID_SIZE[0] or
            new_pos_assistant[1] < 0 or new_pos_assistant[1] >= GRID_SIZE[1] or
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
                (robber_position[0],     robber_position[1] - 1),
                (robber_position[0] + 1, robber_position[1]),
                (robber_position[0],     robber_position[1] + 1),
            ]
            possible_moves = [
                move
                for move in possible_moves
                if (
                    move[0] >= 0
                    and move[0] < GRID_SIZE[0]
                    and move[1] >= 0
                    and move[1] < GRID_SIZE[1]
                    and move not in illegal_positions
                )
            ]

            robber_position = random.choice(possible_moves)

        # Calculate the new state's Q-value
        new_state_reward1 = rewards[new_poshunter[0], new_poshunter[1]]
        new_state_reward2 = rewards[new_pos_assistant[0], new_pos_assistant[1]]

        # Update the shared Q-values for both the bounty hunter and the assistant
        Q_shared[current_state_q_index1][move_hunter] += learning_rate * (
            new_state_reward1
            + discount * np.max(Q_shared[new_poshunter[0] * GRID_SIZE[1] + new_poshunter[1]])
            - Q_values1
        )

        Q_shared[current_state_q_index2][move_assist] += learning_rate *        (
            new_state_reward2
            + discount * np.max(Q_shared[new_pos_assistant[0] * GRID_SIZE[1] + new_pos_assistant[1]])
            - Q_values2
        )

        # Store Q-values for this state-action pair
        episode_Q_values1 = Q_shared[current_state_q_index1][move_hunter]
        episode_Q_values2 = Q_shared[current_state_q_index2][move_assist]

    # Record Q-values for this episode
    Q_values_per_episode.append(episode_Q_values1)

    if epoch > 0:
        # Calculate error rate
        error_rate = np.mean(np.abs(Q_shared - Q_shared_old))
        error_rates.append(error_rate)

        if error_rate < convergence_threshold and convergence_time is None:
            convergence_time = epoch

    Q_shared_old = np.copy(Q_shared)

# Check if either the bounty hunter or the assistant caught the robber after learning
if hunter_pos == robber_position:
    print("The bounty hunter caught the robber!")
elif asist_pos == robber_position:
    print("The assistant caught the robber!")
else:
    print("The robber got away.")

# Plot the evolution of error rate
plt.plot(error_rates, color='green', linestyle='-', label='Cooperative Agents')
plt.title("Evolvement of Error Rate")
plt.xlabel("Epochs")
plt.ylabel("Error Rate")
if convergence_time is not None:
    plt.axvline(x=convergence_time, color='red', linestyle='--', label=f'Convergence at Epoch {convergence_time}')
plt.legend()
plt.show()

# Plot the max Q-value for each episode
max_Q_values = [np.max(episode) for episode in Q_values_per_episode]

plt.figure()
plt.plot(max_Q_values, label='Max Q-Value')
plt.title('Max Q-Value for Each Episode')
plt.xlabel('Episode')
plt.ylabel('Max Q-Value')
plt.legend()
plt.show()

