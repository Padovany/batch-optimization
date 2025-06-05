'''
10X scaled PROWIM wing
Tip prop
Prop on
'''

import numpy

# scaling factor
X_PROWIM = 10. #10.
runs = 2

# Flow conditions
mach =      [0.3 for _ in range(runs)]
altitude =  [457.2 for _ in range(runs)]
cl =        [0.3,0.0]
alpha =     [0. for _ in range(runs)]

# Geometry specs
areaRef =   [X_PROWIM*0.24*X_PROWIM*0.64 for _ in range(runs)]
chordRef =  [X_PROWIM*0.24 for _ in range(runs)]

# DV and con related
twist_lower = [-10. for _ in range(runs)]
twist_upper = [10. for _ in range(runs)]
twist_scale = [0.2 for _ in range(runs)]

shape_lower = [X_PROWIM*(-0.0144) for _ in range(runs)]
shape_upper = [X_PROWIM*(0.0144) for _ in range(runs)]
shape_scale = [100./X_PROWIM for _ in range(runs)]

leList = [[[X_PROWIM*0.01, 0, X_PROWIM*0.001], [X_PROWIM*0.01, 0, X_PROWIM*0.629]] for _ in range(runs)]
teList = [[[X_PROWIM*0.239, 0, X_PROWIM*0.001], [X_PROWIM*0.239, 0, X_PROWIM*0.629]] for _ in range(runs)]

# Actuator zone specs
thrust = [27.9 for _ in range(runs)]
##swirlFact = [-1.0,1.0] ### 1 is outboard-up looking along the x-axis; -1 is inboard-up
distribPDfactor = [2*0.85 for _ in range(runs)]

axisPt1 = [X_PROWIM*numpy.array([-.2518,0.,0.64]) for _ in range(runs)]
axisPt2 = [X_PROWIM*numpy.array([-.2018,0.,0.64]) for _ in range(runs)]
mDistribParam = [1. for _ in range(runs)]
nDistribParam = [0.2 for _ in range(runs)] 
innerZeroThrustRadius = [X_PROWIM*0.041 for _ in range(runs)]
propRadius = [X_PROWIM*0.118 for _ in range(runs)]
spinnerRadius = [X_PROWIM*0.018 for _ in range(runs)]
rootDragFactor = [0.25 for _ in range(runs)]

# Multipoint info
isAlphaDV = [False for _ in range(runs)]
nProcs =    [28 for _ in range(runs)]
weight =    [1.0 for _ in range(runs)]  # weight for obj func

# ADflow options
nearWallDist = [0.1*X_PROWIM for _ in range(runs)]
sliceLocations = [X_PROWIM*numpy.array([
0.05*0.64,
0.1*0.64,
0.15*0.64,
0.2*.64,
0.25*0.64,
0.3*0.64,
0.345*0.64,
0.35*0.64,
0.35*0.64*1.02,
0.4*0.64,
0.45*0.64,
0.3,
0.5*0.64,
0.55*0.64,
0.6*0.64,
0.62*0.64,
0.623*0.64,
0.65*0.64,
0.7*0.64,
0.75*0.64,
0.8*0.64,
0.85*0.64,
0.9*0.64
]) for _ in range(runs)]
