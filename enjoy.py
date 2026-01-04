import gymnasium
import flappy_bird_gymnasium
from stable_baselines3 import PPO
import time

# 1. Încărcăm mediul, dar acum cu grafică ('human')
env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

# 2. Încărcăm modelul antrenat anterior
model_path = "flappy_bird_ppo_model"
try:
    model = PPO.load(model_path, env=env)
except FileNotFoundError:
    print("Nu am găsit fișierul modelului! Rulează întâi train.py.")
    exit()

# 3. Punem AI-ul la treabă
obs, _ = env.reset()
score = 0

while True:
    # Modelul prezice cea mai bună acțiune bazată pe ce "vede" (obs)
    action, _states = model.predict(obs, deterministic=True)
    
    # Executăm acțiunea în joc
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Adăugăm un pic de delay să putem urmări cu ochiul liber
    # time.sleep(0.01) 
    
    if terminated or truncated:
        obs, _ = env.reset()
        print(f"Joc terminat! Se restartează...")

env.close()