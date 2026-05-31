import os
import json

# Define your local directory paths
dataset_folder = os.path.abspath("./libero_xvla/libero_spatial")
output_json_path = "./libero_xvla/libero_spatial_meta.json"

hdf5_paths = []

# 1. Recursively find all HDF5 data files
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith((".hdf5", ".h5")):
            absolute_path = os.path.abspath(os.path.join(root, file))
            hdf5_paths.append(absolute_path)

# Keep the trajectory files deterministically ordered
hdf5_paths.sort()

# 2. Assemble the final configuration mapping matching the code snippet
meta_config = {
    "dataset_name": "libero_spatial",
    "robot_type": "libero",  # Routes to LiberoHandler
    "language_instruction_key": "language_instruction",
    "observation_key": [
        "observation/third_image",
        "observation/wrist_image"
    ],
    "datalist": hdf5_paths  # A clean, flat list of string file paths
}

# 3. Write out to json file
with open(output_json_path, "w") as json_file:
    json.dump(meta_config, json_file, indent=4)

print(f"🎉 Meta file successfully built for the absolute path reader!")
print(f"📦 Total trajectories mapped: {len(hdf5_paths)}")
print(f"💾 File ready at: {output_json_path}")