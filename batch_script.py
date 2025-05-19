import numpy as np
import pandas as pd
import argparse
import subprocess
import os


# ---- Fixed inputs ----

csv_file = "inputs.csv"

disk_surf_cgns = "./reference_geometry/disk_surface.cgns"
#prop_region_mesh = "../reference_geometry/disk.cgns"
#wing_vol_mesh = "../tools/For_PROWIM_validation/Wing_volume_mesh/L2/wing_vol_front.cgns"

output_folders = "../output"
#new_disk_surf = "new_disk_surf.xyz"
#new_overset_mesh = "new_overset_mesh.cgns"

python_script_disk = "./tools/generate_new_disk_surface.py"
#python_script_overset_mesh = "../tools/generate_new_overset_mesh.py"
#python_script_ADflow = "../tools/aero_prop_wing.py"

baseline_prop_x = -0.2
baseline_prop_y = 0.0
baseline_prop_z = 0.3


# ---- Read input CSV ----

df = pd.read_csv(csv_file)

cases = df["case"].tolist()
prop_x_list = df["prop_x"].tolist()
prop_y_list = df["prop_y"].tolist()
prop_z_list = df["prop_z"].tolist()
#thrust_list = df["thrust"].tolist()
#AoA_list = df["AoA"].tolist()
#Mach_list = df["Mach"].tolist()
#swirl_factor_list = df["swirl_factor"].tolist()

dx = np.array(prop_x_list) - baseline_prop_x
dy = np.array(prop_y_list) - baseline_prop_y
dz = np.array(prop_z_list) - baseline_prop_z 


# ---- Loop over the number of cases ----

for i in range(len(cases)):
    # Create folder and files for case output
    output_path = f"./{output_folders}/{cases[i]}"
    subprocess.run(["mkdir", "-p", output_path])
    subprocess.run(["touch", f"{output_path}/generated_disk_surface.xyz"]) 

    # Generate disk surface
    subprocess.run(["python3.9", python_script_disk, disk_surf_cgns, str(dx[i]), str(dy[i]), str(dz[i]), f"{output_path}/generated_disk_surface.xyz"]) 
    print(" ---- ")
    print(f" ---- Done disk surface step for case {cases[i]}")
    print(" ---- ")

    # Generate volume mesh
    #subprocess.run(["python3.9", python_script_overset_mesh, prop_region_mesh, str(dx[i]), str(dy[i]), str(dz[i]), wing_vol_mesh, new_overset_mesh])
    #print(" ---- ")
    #print(" ---- Done disk overset mesh step for case " + str(i))
    #print(" ---- ")

    # Run with ADflow
    #subprocess.run(["mpirun", "-np", "28",  "python3.9", python_script_ADflow, new_disk_surf, new_overset_mesh, "output_" + str(i), str(prop_x_list[i]), str(prop_y_list[i]), str(prop_z_list[i]), str(thrust_list[i]), str(AoA_list[i]), str(Mach_list[i]), str(swirl_factor_list[i]) ])
    #print(" ---- ")
    #print(" ---- Done disk ADflow step for case " + str(i))
    #print(" ---- ")

