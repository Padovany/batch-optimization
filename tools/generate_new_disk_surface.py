import os
import numpy as np
import argparse
import subprocess


# ---- Get the arguments ----

parser = argparse.ArgumentParser(description="Script to generate new disk surface")

# Add arguments
parser.add_argument("input_cgns", type=str, help="The baseline cgns surface (.cgns)")
parser.add_argument("dx", type=str, help="x-diplacement")
parser.add_argument("dy", type=str, help="y-diplacement")
parser.add_argument("dz", type=str,  help="z-diplacement")
parser.add_argument("output_xyz", type=str, help="Output Plot3D filename (.xyz)")

# Parse the command-line arguments
args = parser.parse_args()


# ---- Do the translation and Plot3D file generations ----

subprocess.run(["cgns_utils", "translate", args.input_cgns, args.dx, args.dy, args.dz, "temp.cgns"])

subprocess.run(["cgns_utils", "cgns2plot3d", "temp.cgns", args.output_xyz])

subprocess.run(["rm", "temp.cgns"])
