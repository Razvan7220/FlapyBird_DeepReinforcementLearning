import flappy_bird_gymnasium
import gymnasium
import time

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)
obs, _ = env.reset()

while True:
    # Next action:
    # (feed the observation to your agent here)
    action = env.action_space.sample()

    # Processing:
    obs, reward, terminated, _, info = env.step(action)

    # Slow down slightly so you can see what is happening
    time.sleep(0.05) 

    # Checking if the player is still alive
    if terminated:
        break

env.close()