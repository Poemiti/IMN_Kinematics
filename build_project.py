from pathlib import Path
import subprocess


base_dir = [
    Path("./config"),
    Path("./data"),
]

####################### ask for project name


print("\n" + "="*30 + "\nBuilding new project \n" + "="*30 + "\n")

project_name = input("Project name : ").strip()


if not project_name:
    raise ValueError("Project name cannot be empty.")


####################### Create project directories

base_dir = [
    Path("./config"),
    Path("./data"),
]

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



####################### YAML configuration files

project_yaml = f"""\
# Short description of the '{project_name}' project:
# ...
#

# Hyperparameters (should never move)
fps: 30
camera_view: ["left", "right"]
condition: 
    Opto_on
    Opto_off
"""


subject_info_yaml = f"""\
# Short description of the subjects used in '{project_name}' project
# ...
#

subject_01:
    contra_hemi: false
    info: "Short description..."

subject_02:
    contra_hemi: false
    info: "Short description..."
"""


paths_yaml = """\
# Short description of the paths
# ...
#

# Here is an example of organisation for the results:

# Input data
model: "path/to/model"
raw_videos: "path/to/video/"

# Output folder
data_root: ./data
database: "{data_root}/database"

# Inter-rat
inter_rat: "{data_root}/inter_rat"

# Per-rat
rat_root: "{data_root}/rat_{rat_name}"
raw_clips: "{rat_root}/raw_clips"

processed: "{rat_root}/processed"
dlc: "{processed}/dlc"
metrics: "{processed}/metrics"
luminosity: "{processed}/luminosity"
preprocessing: "{processed}/preprocessing"
analysis: "{rat_root}/analysis_{bodypart}"
"""



####################### Write YAML files


yaml_files = {
    config_dir / "project.yaml": project_yaml,
    config_dir / "subject_info.yaml": subject_info_yaml,
    config_dir / "paths.yaml": paths_yaml,
}

for file_path, content in yaml_files.items():
    with file_path.open("w", encoding="utf-8") as f:
        f.write(content)



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
└── data
    └── {project_name}/
""")