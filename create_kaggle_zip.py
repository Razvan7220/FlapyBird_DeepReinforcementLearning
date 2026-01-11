"""
Create Kaggle-compatible ZIP archive with Unix-style paths.
"""
import zipfile
import os
from pathlib import Path

# Base directory
base_dir = Path(r"c:\Users\razva\Desktop\FlapyBird_DeepReinforcementLearning")
output_file = base_dir / "FlapyBird_Kaggle_OPTIMIZED.zip"

# Files to include
files_to_archive = [
    "dqn/__init__.py",
    "dqn/config.py",
    "dqn/dqn_agent.py",
    "dqn/network.py",
    "dqn/replay_buffer.py",
    "dqn/utils.py",
    "train_dqn.py",
    "evaluate.py",
    "requirements.txt",
    "KAGGLE_SETUP.md"
]

# Create ZIP with forward slashes (Unix-compatible)
with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for file_path in files_to_archive:
        full_path = base_dir / file_path
        if full_path.exists():
            # Use forward slashes in archive (Unix-compatible)
            arcname = file_path.replace('\\', '/')
            zipf.write(full_path, arcname=arcname)
            print(f"Added: {arcname}")
        else:
            print(f"Skipped (not found): {file_path}")

print(f"\nArchive created: {output_file.name}")
print(f"Size: {output_file.stat().st_size / 1024:.2f} KB")
print("Ready for Kaggle upload!")
