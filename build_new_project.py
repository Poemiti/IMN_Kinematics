from pathlib import Path
import subprocess
import shutil

base_dir = [
    Path("./config"),
    Path("./data"),
    Path("./src")
]

####################### ask for project name


print("\n" + "="*30 + "\nBuilding new project \n" + "="*30 + "\n")

project_name = input("Project name : ").strip()


if not project_name:
    raise ValueError("Project name cannot be empty.")


####################### Create project directories

existing_dirs = [
    root / project_name
    for root in base_dir
    if (root / project_name).exists()
]

if existing_dirs:
    print(f"\nError: project '{project_name}' already exists")
    raise SystemExit(1)


for root in base_dir:
    
    project_dir = root / project_name
    project_dir.mkdir(parents=True, exist_ok=True)

config_dir = Path("./config") / project_name
data_dir = Path("./data") / project_name
src_dir = Path("./src") / project_name


####################### YAML configuration files

condition_yaml = f"""\
# filter criteria before launching preprocessing / analysis etc...

# movement type
mvt_type: 
  CONTRA: t
  IPSI: 
  UNKNOWN: 

# condition
condition: 
  CHR: t
  CTRL: 

# subject
subject: 
  "1": t
  "2": 
  "3": t
"""

project_yaml = f"""\
# Hyperparameters (that don't change)

bodyparts:
  - Hand_base
  - Base_finger_2
  - Base_finger_3
  - Phalanx_1_finger_2
  - Phalanx_1_finger_3
  - finger_1
  - finger_2
  - finger_3
  - soft_pad
  - Front_PAD_corner
  - Back_PAD_corner
  - Lever_ball
  - Lever_base
  - Lever_corner
  - hand
  - wrist


# Absolute paths
model_path: "/media/filer2/T4b/UserFolders/Poemiti/Reaching-DLC-model-main/data/model/DLC-Poe-2026-03-27/"
raw_videos_path: "/media/filer2/T4b/UserFolders/Raphael/Raphael_M2_2024/Opto_SkilledReaching_2024_experiments/MOVIES/"


# Parameters for preprocessing
laser_on_duration: 0.325  # sec (0.3 of laser + 0.025 from pad_off to laser stim)
gap_scale: 0.2 # for outlier detection
distance_thresh: 0.6 # for outlier detection (cm)
max_outlier: 10 
shift_ref: "./data/BetaMouv/camera_shift/#517_CHR_CONTRA_Beta_RightHemi_leftView_high_LaserOff_2024-07-05.png"

"""


subject_info_yaml = f"""\
# Metadata relative to subjects used in '{project_name}' project

subject_01:
    contra_hemi: false
    info: "Short description..."

subject_02:
    contra_hemi: false
    info: "Short description..."
"""



####################### Write YAML files


yaml_files = {
    config_dir / "project_info.yaml": project_yaml,
    config_dir / "subject_info.yaml": subject_info_yaml,
    config_dir / "condition.yaml": condition_yaml,
}

for file_path, content in yaml_files.items():
    with file_path.open("w", encoding="utf-8") as f:
        f.write(content)

src_path = []
for path in list((src_dir / "Base").glob("*.*py")): 
    src_path.append(f"├── {path.name}")
src_path_name = '\n        '.join(src_path)

shutil.copytree(Path("./src") / "Base", src_dir, dirs_exist_ok=True)


####################### Print project tree

print("\n" + "=" * 30)
print(f"Project '{project_name}' created successfully!")
print("=" * 30)

print("\nThe following files have been added :")
print(f"""
./
├── config/
│   ├── {project_name}/
│   │   ├── paths.yaml
│   │   ├── project.yaml
│   │   └── subject_info.yaml
├── data/
│   └── {project_name}/
└── src/
    └── {project_name}/
        {src_path_name}
""")