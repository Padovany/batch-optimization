import os
import numpy as np
import argparse
import subprocess

from pyhyp.utils import simpleOCart
from cgnsutilities.cgnsutilities import readGrid, combineGrids


# ---- Get the arguments ----

parser = argparse.ArgumentParser(description="Test script to generate new overset mesh")

# Add arguments
parser.add_argument('prop_mesh', type=str, help="The baseline propeller region mesh (.cgns)")
parser.add_argument("dx", type=str, help="x-diplacement")
parser.add_argument("dy", type=str, help="y-diplacement")
parser.add_argument("dz", type=str,  help="z-diplacement")
parser.add_argument('wing_mesh', type=str, help="The baseline wing volume mesh (.cgns)")
parser.add_argument("output", type=str, help="Output mesh filename (.cgns)")

args = parser.parse_args()

# ---- Generate background mesh ----

# Translate the propeller region mesh
subprocess.run(["cgns_utils", "translate", args.prop_mesh, args.dx, args.dy, args.dz, "temp.cgns"])

# Combine with the wing mesh
subprocess.run(["cgns_utils", "combine", "temp.cgns", args.wing_mesh, "prop_wing.cgns"])

# Generate the background mesh

wingGrid = readGrid("prop_wing.cgns")

dh = 0.01
hExtra = 20*0.64
nExtra = 31
sym = 'z'
mgcycle = 3
backgroundFile = 'background.cgns'

simpleOCart(wingGrid, dh, hExtra, nExtra, sym, mgcycle, backgroundFile)

backgroundGrid = readGrid(backgroundFile)

# Combine background grid with wing meshes
oversetGrid = combineGrids([backgroundGrid, wingGrid], useOldNames=False)
oversetGrid.writeToCGNS(args.output)

subprocess.run(["rm", "temp.cgns", "background.cgns", "prop_wing.cgns"])

