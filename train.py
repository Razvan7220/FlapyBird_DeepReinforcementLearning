import gymnasium
import flappy_bird_gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# 1. Configurare Mediu
env = gymnasium.make("FlappyBird-v0", render_mode=None, use_lidar=True)

# 2. Configurare Callback (Salvări automate)
# Salvăm modelul la fiecare 10.000 de pași într-un folder numit "logs"
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./logs/",
    name_prefix="flappy_model"
)

# 3. Configurare Model (Aici poți să te joci cu parametrii)
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0001, # Încearcă să schimbi asta data viitoare
    batch_size=128,        # Câte experiențe analizează o dată
    ent_coef=0.005,       # Curiozitate
    tensorboard_log="./flappy_bird_tensorboard/"
)

# 4. Start Antrenament
print("--- Start Antrenament (apasă Ctrl+C pentru a opri forțat) ---")

# Antrenăm pentru 200.000 de pași și folosim callback-ul pentru salvări
model.learn(total_timesteps=2000000, callback=checkpoint_callback)

# 5. Salvare Finală
model.save("flappy_bird_final")
print("Gata! Modelul final a fost salvat.")