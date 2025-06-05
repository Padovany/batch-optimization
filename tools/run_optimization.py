# ==============================================================================
# Import modules
# ==============================================================================
# Standard python modules
from __future__ import print_function
import os ### used for creating directories, saving output etc.
import shutil ### also related to copying and saving files
import argparse ### for input arguments
import numpy
import inspect ### also for input saving files
from importlib.machinery import SourceFileLoader ### load files written as python modules; used here for flight conditions
import warnings
warnings.filterwarnings('ignore')

# External python modules
from pprint import pprint as pp
from mpi4py import MPI

# MDOlab python modules
from baseclasses import *
from multipoint import *
from adflow import ADFLOW
# import repostate # does not work with git
from pyoptsparse import Optimization, OPT
from pygeo import DVGeometry, DVConstraints
from idwarp import USMesh

# ==============================================================================
# Input Information
# ==============================================================================
parser = argparse.ArgumentParser()
# General arguments (-- means they are optional)
parser.add_argument('--task', type=str, default='opt', help='Types: opt, analysis, clsolve, alpha-sweep, clsolve_opt')
parser.add_argument('--hotStart', type=str, default=None)
parser.add_argument('--hist', type=str, default=None)
parser.add_argument('--histKey', type=str, default=None)
parser.add_argument('--output', type=str, default='./out')
parser.add_argument('--procs', type=int, default='28')
# Mesh information
parser.add_argument('--mesh', type=str, default='L2')
# Case information
parser.add_argument('--cruiseset', type=str, default='cruiseset_00.py')
parser.add_argument('--rotation', type=str, default='1')
# Geometric information
parser.add_argument('--ffd', type=str, default='10x10')
parser.add_argument('--disksurf', type=str, default='disk_surf')
parser.add_argument('--twist', type=int, default=0, help='Use twist var')
parser.add_argument('--twistroot', type=int, default=0, help='Twist the root')
parser.add_argument('--shape', type=int, default=0, help='Use shape var')
parser.add_argument('--volume', type=float, default=None, help='Lower volume limit')
parser.add_argument('--thickness', type=float, default=None, help='Lower thickness limit')
parser.add_argument('--area', type=int, default=None, help='Use area con')
parser.add_argument('--monotonic', type=str, default=None, help='Use monotonic con')
args = parser.parse_args()

# Load history file to set design variable *values* from previous cases 
xhist = {}
""" if args.hist is not None:
    db = SqliteDict(args.hist)
    if args.histKey is not None:
        xhist = db[args.histKey]['xuser']
    else:
        xhist = db[db['last']]['xuser']
    db.close() """

# Collect input files to save
files = {}
files['grid'] = args.mesh
files['ffd'] = args.ffd
files['cruiseset'] = args.cruiseset
files['self'] = inspect.getfile(inspect.currentframe())

if MPI.COMM_WORLD.rank == 0:
    # Create a folder called INPUT in the output directory to save input files.
    copyDir = os.path.join(args.output, 'INPUT')
    os.system('mkdir -p %s'%copyDir)
    for key in files:
        try:
            shutil.copy(files[key], copyDir)
        except:
            pass

    # And the hist if it's there
    if args.hist:
        shutil.copy(args.hist, copyDir)

# ==============================================================================
# Set up cases
# ==============================================================================
cruiseSet = SourceFileLoader('cruiseset', files['cruiseset']).load_module()
nCruiseCases = len(cruiseSet.mach)
cruiseProblems = []
for i in range(nCruiseCases):
    evalFuncs = ['cd', 'cl']
    ap = AeroProblem(
        name='cruise%d'%i, 
        mach=cruiseSet.mach[i], 
        alpha=cruiseSet.alpha[i],
        altitude=cruiseSet.altitude[i],
        xRef=0, yRef=0, zRef=0,
        areaRef=cruiseSet.areaRef[i],
        chordRef=cruiseSet.chordRef[i],
        evalFuncs=evalFuncs)
    if cruiseSet.isAlphaDV[i]:
        ap.addDV('alpha', lower=-5., upper=5., scale=0.1)
    cruiseProblems.append(ap)

# ==============================================================================
# MPI Processor Setup
# ==============================================================================

# Create multipoint communication object
MP = multiPointSparse(MPI.COMM_WORLD)
print(f"procs:{args.procs}, nMembers: {nCruiseCases}")
MP.addProcessorSet('cruise', nMembers=nCruiseCases, memberSizes=args.procs/nCruiseCases)

# Create the communicators
comm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
setName = MP.getSetName() # used for output files
ptDirs = MP.createDirectories(args.output)

# Create a directory for all standard output
stdoutDir = os.path.join(args.output, 'stdout')
if MP.gcomm.rank == 0:
    os.system('mkdir -p %s'%stdoutDir)
MP.gcomm.barrier()

#Redirect STDOUT of the roots of the gcomm to a file. Everything else
#we can just let dump into the output file:
""" if comm.rank == 0:
    fName = os.path.join(stdoutDir, '%s_%d.out'%(setName, ptID))
    outFile = open(fName, 'w')
    redirectIO(outFile)

# Echo the argparse options:
if comm.rank == 0:
    print('+----------- Command Line Options ---------------+')
    pp(vars(args))
    print('+------------------------------------------------+') """

# ==============================================================================
# CFD Solver
# ==============================================================================

gridfile = args.mesh

oversetpriority = {}

aeroOptions = {
    # I/O Parameters
    'gridFile': gridfile,
    'outputDirectory':args.output,
    'monitorvariables':['resrho','cl','cd','cpu','resturb'],
    'writeTecplotSurfaceSolution':True,
    
    # Solver Parameters
    'MGCycle':'sg',

    # Physics Parameters
    'equationType':'RANS',

    'volumevariables':['blank', 'cp', 'mach'],
    'surfacevariables':['cp', 'vx', 'vy', 'vz', 'mach', 'blank', 'cf', 'cfx', 'cfy', 'cfz', 'sepsensor', 'yplus'],

    # ANK Solver Parameters
    'useANKSolver':True,
    'ankswitchtol':1e5,
    'anksecondordswitchtol': 1e-4,
    'ankcoupledswitchtol': 1e-5,

    # NK Solver Parameters
    'useNKSolver':True,
    'nkswitchtol': 1e-7,

    # Termination Criteria
    'L2Convergence': 1e-10,
    'nCycles':10000,
    
    # Adjoint Parameters
    'adjointL2Convergence':1e-10,
    'adjointMaxIter' : 2000,

    # Zipper mesh option
    'debugzipper':False,
    'usezippermesh':True,
    'nrefine':10, # number of times to run IHC cycle
    'nearwalldist':cruiseSet.nearWallDist[0],
    'oversetpriority':oversetpriority
}

# Create solver
CFDSolver = ADFLOW(options=aeroOptions, comm=comm)

# Save the total lift distribution
CFDSolver.addLiftDistribution(400, 'z')

CFDSolver.addSlices('z', cruiseSet.sliceLocations[0])

# Actuator region
if cruiseSet.thrust[0] != 0.:
    CFDSolver.addActuatorRegion(
        args.disksurf, 
        'simpleProp', 
        cruiseSet.axisPt1[0], 
        cruiseSet.axisPt2[0], 
        'disk', 
        thrust=cruiseSet.thrust[0], 
        swirlFact=args.rotation, 
        mDistribParam=cruiseSet.mDistribParam[0], 
        nDistribParam=cruiseSet.nDistribParam[0], 
        distribPDfactor=cruiseSet.distribPDfactor[0], 
        innerZeroThrustRadius=cruiseSet.innerZeroThrustRadius[0], 
        propRadius=cruiseSet.propRadius[0], 
        spinnerRadius=cruiseSet.spinnerRadius[0], 
        rootDragFactor=cruiseSet.rootDragFactor[0])

    CFDSolver.writeActuatorRegions('write_act_reg')
else:
    print("PROP IS OFF")

meshOptions = {'gridFile':gridfile, 'useRotations':False}
mesh = USMesh(options=meshOptions, comm=comm)

CFDSolver.setMesh(mesh)

# ==============================================================================
# DVGeometry
# ==============================================================================

# Create DVGeometry object
FFDFile = args.ffd
DVGeo = DVGeometry(FFDFile)

# Create reference axis
nRefAxPts = DVGeo.addRefAxis('wing', xFraction=0.25, alignIndex='k')

# Set up global design variables
if args.task == 'clsolve_opt':
    # !!!!! twist funcs changed for a single "twist"
    print("CL SOLVE W/ OPT W/ ROOT TWIST")
    nTwist = nRefAxPts
    def twist(val, geo):
        for i in range(0, nRefAxPts):
            geo.rot_z['wing'].coef[i] = val[0]
    DVGeo.addGlobalDV(dvName='twist', value=[0], func=twist, lower=cruiseSet.twist_lower[0], upper=cruiseSet.twist_upper[0], scale=cruiseSet.twist_scale[0])
elif args.twist:
    if args.twistroot:
        print("WITH ROOT TWIST")
        nTwist = nRefAxPts
        def twist(val, geo):
            for i in range(0, nRefAxPts):
                geo.rot_z['wing'].coef[i] = val[i]
    else:
        print("NO ROOT TWIST")
        nTwist = nRefAxPts - 1
        def twist(val, geo):
            for i in range(1, nRefAxPts):
                geo.rot_z['wing'].coef[i] = val[i-1]

    DVGeo.addGlobalDV(dvName='twist', value=[0]*nTwist, func=twist, lower=cruiseSet.twist_lower[0], upper=cruiseSet.twist_upper[0], scale=cruiseSet.twist_scale[0])

# Set up local design variables
if args.shape:
    DVGeo.addLocalDV('local', lower=cruiseSet.shape_lower[0], upper=cruiseSet.shape_upper[0], axis='y', scale=cruiseSet.shape_scale[0])

# Add DVGeo object to CFD solver
CFDSolver.setDVGeo(DVGeo)

# ==============================================================================
# DVConstraints
# ==============================================================================

DVCon = DVConstraints()
DVCon.setDVGeo(DVGeo)
DVCon.setSurface(CFDSolver.getTriangulatedMeshSurface())

# Volume constraints
leList = cruiseSet.leList
teList = cruiseSet.teList

if args.volume is not None:
    DVCon.addVolumeConstraint(name='volume', leList=leList, teList=teList,
        nSpan=20, nChord=20, lower=args.volume)

# Thickness constraints
if args.thickness is not None:
    DVCon.addThicknessConstraints2D(name='thickness', leList=leList, teList=teList,
        nSpan=10, nChord=10, lower=args.thickness, upper=2)
    # capcurve = numpy.array([[0.88333, 3.00827, 0], [0.65, 3.02066, 0],
    #     [0.41667, 3.02868, 0], [0.18333, 3.02815, 0], [0.01657, 3.01081, 0]])
    # DVCon.addThicknessConstraints1D(name='capthickness', ptList=capcurve,
    #     axis=[0,0,1], nCon=8, lower=args.thickness, upper=2)

# Projected Area Constraint
if args.area:
    DVCon.addProjectedAreaConstraint(name='planform_area', scaled=True, axis='z',
        lower=1.0, upper=1.0)

# LE/TE Constraints
if args.shape:
    DVCon.addLeTeConstraints(volID=0, faceID='ihigh')
    DVCon.addLeTeConstraints(volID=0, faceID='ilow')

# Monotonic Constraint
if args.monotonic:
    var = args.monotonic
    DVCon.addMonotonicConstraints(key=var, slope=1.0, name='mono_'+var)

if comm.rank == 0:
    fileName = os.path.join(args.output, 'constraints.dat')
    DVCon.writeTecplot(fileName)
    if args.volume is not None:
        print('Initial volume: {}'.format(DVCon.constraints['volCon']['volume'].V0))
    if args.area:
        print('Initial projected area: {}'.format(DVCon.constraints['projAreaCon']['planform_area'].X0))

# ==============================================================================
# Functions:
# ==============================================================================
def cruiseObj(x):
    funcs = {}
    # Echo the variables to the screen and set variables
    if comm.rank == 0:
        pp(x)
    DVGeo.setDesignVars(x)
    DVCon.evalFunctions(funcs)

    for i in range(nCruiseCases):
        if i == ptID: # This makes it so that each processor set only runs its designated case
            cruiseProblems[i].setDesignVars(x)
            CFDSolver(cruiseProblems[i])
            CFDSolver.evalFunctions(cruiseProblems[i], funcs)
            CFDSolver.checkSolutionFailure(cruiseProblems[i], funcs)

    if comm.rank == 0:
        print('\nAeroFuncs')
        pp(funcs)
    return funcs

def cruiseSens(x, funcs):
    funcsSens = {}
    for i in range(nCruiseCases):
        if i == ptID:
            CFDSolver.evalFunctionsSens(cruiseProblems[i], funcsSens)
    DVCon.evalFunctionsSens(funcsSens)

    if comm.rank == 0:
        print('\nAeroSens')
        pp(funcsSens)
    return funcsSens

if args.task == 'clsolve_opt':
    def objCon(funcs, printOK):
        funcs['obj'] = 0.0
        for i in range(nCruiseCases):
            ap = cruiseProblems[i]
            funcs['obj'] += (funcs[ap.name+'_cl'] - cruiseSet.cl[i])**2
        if printOK:
            print('funcs in obj:', funcs)
        return funcs
else:
    def objCon(funcs, printOK):
        funcs['obj'] = 0.0
        for i in range(nCruiseCases):
            ap = cruiseProblems[i]
            funcs['obj'] += funcs[ap.name+'_cd'] * cruiseSet.weight[i]
            funcs['cl_con_'+ap.name] = funcs[ap.name+'_cl'] - cruiseSet.cl[i]
            # if cruiseSet.cmx[i]:
            #     funcs['cmx_con_'+ ap.name] = funcs[ap.name+'_cmx']/3.06 + 0.1069
        if printOK:
            print('funcs in obj:', funcs)
        return funcs

if args.task == 'opt' or args.task == 'clsolve_opt':
    # ==========================================================================
    # Aerodynamic Shape Optimization
    # ==========================================================================
    numpy.set_printoptions(threshold=1000000)

    # Set design variables
    DVGeo.setDesignVars(xhist)
    for ap in cruiseProblems:
        ap.setDesignVars(xhist)

    # --------------------------------------------------------------------------
    # Optimization Problem Setup
    # --------------------------------------------------------------------------
    optProb = Optimization('Aerodynamic shape optimization', MP.obj)

    # -------------
    # Set objective
    # -------------
    optProb.addObj('obj', scale=1e2)

    # --------------------
    # Set design variables
    # --------------------
    DVGeo.addVariablesPyOpt(optProb)

    for ap in cruiseProblems:
        ap.addVariablesPyOpt(optProb)

    # ---------------
    # Set constraints
    # ---------------
    DVCon.addConstraintsPyOpt(optProb)

    #for i in range(nCruiseCases):
    #    ap = cruiseProblems[i]
    #    dvs = ap.DVs.keys() + DVGeo.getValues().keys()
    #    if args.task != 'clsolve_opt':
    #        optProb.addCon('cl_con_'+ap.name, lower=0, upper=0, scale=10., wrt=dvs)
            # if cruiseSet.cmx[i]:
            #     optProb.addCon('cmx_con_'+ap.name, lower=0, upper=0, scale=1, wrt=dvs)

    # --------------------------------------------------------------------------
    # MultiPoint object set-up
    # --------------------------------------------------------------------------
    MP.setProcSetObjFunc('cruise', cruiseObj)
    MP.setProcSetSensFunc('cruise', cruiseSens)
    MP.setObjCon(objCon)
    MP.setOptProb(optProb)

    # --------------------------------------------------------------------------
    # Set up optimizer and run
    # --------------------------------------------------------------------------
    # Create optimizer
    optOptions = { #Setup for SNOPT not being used
        'Major feasibility tolerance':1.0e-5,
        'Major optimality tolerance':1.0e-5,
        'Difference interval':1e-3,
        'Hessian full memory':None,
        'Function precision':1.0e-8,
        'Major iterations limit':200
    }

    opt = OPT('SLSQP', options={})

    # Print Optimization Problem and sparsity
    if comm.rank == 0:
        print(optProb)
    optProb.printSparsity()

    # Run Optimization
    histFile = os.path.join(args.output, 'opt_hist.hst')
    sol = opt(optProb, MP.sens, storeHistory=histFile, hotStart=args.hotStart)

    # Print final solution
    if comm.rank == 0:
            print(sol)

    # Write final FFD to file
    DVGeo.writePlot3d(os.path.join(args.output, 'modifiedFFD.xyz'))

if args.task == 'analysis':
    funcs = cruiseObj(xhist)
    if comm.rank == 0:
        pp(funcs)

if args.task == 'clsolve':
    DVGeo.setDesignVars(xhist)

    funcs = {}
    for i in range(nCruiseCases):
        if i == ptID:
            ap = cruiseProblems[i]
            ap.setDesignVars(x)
            CFDSolver.solveCL(aeroProblem=ap, CLStar=cruiseSet.cl[i],
                alpha0=ap.alpha, delta=0.01, tol=1e-5)
            CFDSolver.evalFunctions(ap, funcs)

    if comm.rank == 0:
        print('\nAeroFuncs')
        pp(funcs)

if args.task == 'alpha-sweep':
    alpha = numpy.linspace(0, 15, 16)
    ap = cruiseProblems[0]
    data = numpy.zeros((len(alpha), 3))
    for i, a in enumerate(alpha):
        xhist = {
            'alpha_'+ap.name: a
        }
        funcs = cruiseObj(xhist)
        data[i,0] = a
        data[i,1] = funcs['cruise0_cl']
        data[i,2] = funcs['cruise0_cd']

    print(data)
