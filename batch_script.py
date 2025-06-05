import numpy as np
import pandas as pd
import argparse
import subprocess
import os


# ---- Fixed inputs ----
csv_file = "inputs.csv"

disk_surf_cgns = "./reference_geometry/disk_surface.cgns"
prop_region_mesh = "./reference_geometry/disk.cgns"
wing_vol_mesh = "./reference_geometry/wing_vol_front.cgns"

output_folders = "./output"

python_script_disk = "./tools/generate_new_disk_surface.py"
python_script_overset_mesh = "./tools/generate_new_overset_mesh.py"
python_script_optimization = "./tools/run_optimization.py"

baseline_prop_x = -0.2
baseline_prop_y = 0.0
baseline_prop_z = 0.3


# ---- Read input CSV ----
df = pd.read_csv(csv_file)

cases = df["case"].tolist()
prop_x_list = df["prop_x"].tolist()
prop_y_list = df["prop_y"].tolist()
prop_z_list = df["prop_z"].tolist()
rotation_list = df["rotation"].tolist()

dx = np.array(prop_x_list) - baseline_prop_x
dy = np.array(prop_y_list) - baseline_prop_y
dz = np.array(prop_z_list) - baseline_prop_z

# ---- Processing setup ----
parser = argparse.ArgumentParser()
parser.add_argument('--procs', type=str, default='28')
args = parser.parse_args()

# ---- Loop over the number of cases ----
for i in range(len(cases)):
    # Create folder and files for case output
    output_path = f"{output_folders}/{cases[i]}"
    subprocess.run(["mkdir", "-p", output_path])
    subprocess.run(["touch", f"{output_path}/generated_disk_surface.xyz"]) 
    subprocess.run(["touch", f"{output_path}/generated_overset_mesh.cgns"]) 

    # Generate disk surface
    subprocess.run(["python3.9", python_script_disk, disk_surf_cgns, str(dx[i]), str(dy[i]), str(dz[i]), f"{output_path}/generated_disk_surface.xyz"]) 
    print(" ---- ")
    print(f" ---- Done disk surface step for case {cases[i]}")
    print(" ---- ")

    # Generate volume mesh
    subprocess.run(["python3.9", python_script_overset_mesh, prop_region_mesh, str(dx[i]), str(dy[i]), str(dz[i]), wing_vol_mesh, f"{output_path}/generated_overset_mesh.cgns"])
    print(" ---- ")
    print(f" ---- Done disk surface step for case {cases[i]}")
    print(" ---- ")

    # Run with ADflow
    subprocess.run([
        "mpirun",
        "-np", args.procs,
        "python3.9",
        python_script_optimization,
        "--task", "clsolve_opt",
        "--mesh", f"{output_path}/generated_overset_mesh.cgns",
        "--disksurf", f"{output_path}/generated_disk_surface.xyz",
        "--ffd", "./reference_geometry/ffd_13x8.xyz",
        "--rotation", str(rotation_list[i]),
        "--output", f"{output_path}",
        "--procs", args.procs,
        "--prop_x", str(prop_x_list[i]),
        "--prop_y", str(prop_y_list[i]),
        "--prop_z", str(prop_z_list[i])
    ])
    print(" ---- ")
    print(f" ---- Done disk surface step for case {cases[i]}")
    print(" ---- ")

