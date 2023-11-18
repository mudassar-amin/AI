import numpy as np
import random
import matplotlib.pyplot as plt

grid_size = (10, 10)
rewards = np.zeros(grid_size)
rewards[0, 8] = -500
rewards[8, 6] = -100
rewards[5, 9] = 1000

illegal_positions = {(1, 2), (2, 2), (3, 2), (4, 2), (3, 6), (4, 6), (5, 6), (6, 6), (5, 7), (6, 7)}
terminal_pos = {(5, 9)}
num_states = np.prod(grid_size)
num_actions = 5  
Q_shared = np.random.rand(num_states, num_actions)

Q_robber = np.random.rand(num_states, num_actions)
learning_rate = 0.5
discount = 0.5
robber_learning_rate = 0.1
robber_discount = 0.8
epochs = 3000
convergence_threshold = 1e-6

epsilon = 0.1
f2_house = 0.35

start_positions = [(0, i) for i in range(2)] + [(9, i) for i in range(2)]

error_rates = []
steps_taken_per_episode = []

convergence_time = None
Q_values_per_episode = []
robber_position = (8, 6)  
has_loot = False

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))


for epoch in range(epochs):
    start_pos = random.choice(start_positions)
    start_pos2 = random.choice(start_positions)
    hunter_pos = start_pos
    asist_pos = start_pos2
    Q_values_per_episode.append(np.copy(Q_shared))

    total_reward = 0
    steps_taken = 0

    while hunter_pos not in terminal_pos or asist_pos not in terminal_pos:
        robber_position = (8, 6)
        steps_taken += 1

        if hunter_pos == robber_position or asist_pos == robber_position:
            total_reward += 1000
            break

        current_state_q_index1 = hunter_pos[0] * grid_size[1] + hunter_pos[1]
        current_state_q_index2 = asist_pos[0] * grid_size[1] + asist_pos[1]

        if random.random() < epsilon:
            move_hunter = random.choice(range(5))
            move_assist = random.choice(range(5))
        else:
            move_hunter = np.argmax(Q_shared[current_state_q_index1])
            move_assist = np.argmax(Q_shared[current_state_q_index2])

        actions = [(0, -1), (-1, 0), (1, 0), (0, 1), (0, 0)]  # Up, Left, Down, Right, Wait
        new_poshunter = (hunter_pos[0] + actions[move_hunter][0], hunter_pos[1] + actions[move_hunter][1])

        if (
            new_poshunter[0] < 0 or new_poshunter[0] >= grid_size[0] or
            new_poshunter[1] < 0 or new_poshunter[1] >= grid_size[1] or
            new_poshunter in illegal_positions
        ):
            new_poshunter = hunter_pos

        new_pos_assistant = (asist_pos[0] + actions[move_assist][0], asist_pos[1] + actions[move_assist][1])

        if (
            new_pos_assistant[0] < 0 or new_pos_assistant[0] >= grid_size[0] or
            new_pos_assistant[1] < 0 or new_pos_assistant[1] >= grid_size[1] or
            new_pos_assistant in illegal_positions
        ):
            new_pos_assistant = asist_pos

        Q_values1 = Q_shared[current_state_q_index1][move_hunter]
        Q_values2 = Q_shared[current_state_q_index2][move_assist]

        if random.random() < f2_house:
            robber_position = (5, 2)
        else:
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

        
        new_state_q_index = robber_position[0] * grid_size[1] + robber_position[1]
        max_next_q_value = np.max(Q_robber[new_state_q_index])
        new_state_reward = rewards[robber_position[0], robber_position[1]]

        Q_robber[current_state_q_index1][move_hunter] += robber_learning_rate * (
            new_state_reward + robber_discount * max_next_q_value - Q_values1
        )

        if hunter_pos == robber_position:
            break

        if asist_pos == robber_position:
            break

        new_state_reward1 = rewards[new_poshunter[0], new_poshunter[1]]
        new_state_reward2 = rewards[new_pos_assistant[0], new_pos_assistant[1]]

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

        hunter_pos = new_poshunter
        asist_pos = new_pos_assistant

    steps_taken_per_episode.append(steps_taken)
    Q_values_per_episode.append(Q_shared[current_state_q_index1])

    if epoch > 0:
        error_rate = np.mean(np.abs(Q_shared - Q_shared_old))
        error_rates.append(error_rate)

        if error_rate < convergence_threshold and convergence_time is None:
            convergence_time = epoch

    Q_shared_old = np.copy(Q_shared)  

if has_loot and robber_position in terminal_pos:
    print("The robber succeeded in reaching a hideout with the loot!")
else:
    print("The robber was caught before reaching a hideout with the loot.")


ax1.plot(error_rates, color='green', linestyle='-', label='Cooperative Hunt')
ax1.set_title("Evolution of Error Rate")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Error Rate")


ax2.plot(steps_taken_per_episode, color='blue', linestyle='-', label='Steps Taken')
ax2.set_title("Number of Steps Taken by Hunters")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps Taken")

ax1.legend()
ax2.legend()

plt.tight_layout()
plt.show()
