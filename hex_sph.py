#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 22:45:16 2018

@author: mtrawick
"""
#This script puts sphere-forming diblock polymers into an initial
#hexagonal lattice configuration.
import hoomd
import hoomd.md
import numpy as np
import mattreadargs as mra

def make_hex_geom(w):
    #returns points arranged in a hexagonal lattice, L_0=1, centered around 0.
    #Width is given by input parameter w.  
    #Height is calculated for aspect_ratio approximatly 1, rounded up.
    sqr3 = np.sqrt(3)
    unitvec = np.array([0.5, sqr3/2])
    #print(unitvec)

    nh = np.ceil((w/sqr3)/2) * 2 #round up to nearest even number
    #Use np.mgrid to make a list of xy points A in a hexagonal lattice.
    A = np.mgrid[0:w, 0:nh].astype(float)
    A = A.reshape((2,-1)).transpose()

    A[:,1] *= sqr3
    A = np.concatenate((A, A + unitvec))

    #center coordinates around zero:
    A -= (np.max(A[:,0]/2), np.max(A[:,1]/2))
    
    return(A)    


def poly_coords(boxsize, rho, Nab):
    #returns (x,y,z) coordinates of particles for polymers, of density rho,
    #packed within a volume (x,y,z) of boxsize,
    #where each polymer has Nab particles.  
    #Each individual polymer is folded in a zig-zag fashion into a rectangle
    #in the xy plane, with particles arranged in a hexagonal lattice for
    #closer packing.
    
    #Define the coordinates of particles for single polymer
    Nx = int(np.ceil(np.sqrt(Nab)))
    Ny = int(np.ceil(Nab/Nx))
    pol = np.mgrid[0:Nx,0:Ny].astype(float)
    oddrows = np.arange(1,Ny,2)
    pol[1,oddrows] = np.fliplr(pol[1,oddrows])
    pol[1,oddrows] += 0.5
    pol = pol.reshape((2,-1)).transpose()
    pol = pol[0:Nab]
    pol[:,0] *= np.sqrt(3)/2
    pol = np.concatenate((pol,np.zeros(Nab)[np.newaxis].T),axis=1)
    
    #Define the grid of offsets at which each polymer will be placed
    #pol_size = np.max(pol,axis=0) + 1
    pol_size = np.array([Nx * np.sqrt(3)/2, Ny, 1])
    R = (boxsize / pol_size).astype(int)
    offsets = np.mgrid[0:R[0],0:R[1],0:R[2]-1].astype(float)
    offsets = offsets.reshape((3,-1)).T
    offsets[:,2] +=0.5
    offsets *= pol_size.T
    offsets = offsets[np.argsort(offsets[:,2])]

    volume = np.prod(boxsize)
    num_pol = int(volume * rho / Nab)
    #print("packing fraction: ",num_pol / len(offsets))
    if len(offsets) < num_pol:
        print("ERROR in poly_coords. Unable to pack enough polymers into box.")
        print("       Requested density: ", rho)
        print("       Actual density: ", len(offsets) * Nab/ volume)
        num_pol = len(offsets)
    offsets = offsets[0:num_pol]
    
    #Tile the space with the polymer molecules, and return their coordinates.
    particle_offsets = np.repeat(offsets,Nab,axis=0)
    particle_coords = np.tile(pol,(num_pol,1)) + particle_offsets
    
    #particle_coords -= boxsize/2
    particle_coords -= 0.5 * (np.max(particle_coords,axis=0) + 
                              np.min(particle_coords,axis=0))
    return(particle_coords)


def define_initial_hex_snapshot():
    #Define where the templating spheres will be placed, 
    #but don't actually create particles there yet.
    HEXpoints = make_hex_geom(lattice_width)
    box_dims = 2 * (np.max(HEXpoints,axis=0)) + (0.5, np.sqrt(3)/2) 
    HEXpoints *= lattice_constant
    box_dims *= lattice_constant
    box_dims = np.concatenate((box_dims,np.array([thickness])))
    N_HEXpoints = len(HEXpoints)
    HEXpoints = np.concatenate((HEXpoints,np.zeros(N_HEXpoints)[np.newaxis].T),axis=1)
    
    #Define where the actual type A and B particles will be.
    particle_coords = poly_coords(box_dims, rho, Nab)
    N_AB = len(particle_coords) #number of type A and B particles in total
    num_polymers = int(N_AB/Nab)
    
    #Create a snapshot to hold both block copolymers and template particles:
    snapshot = hoomd.data.make_snapshot(N=N_AB + N_HEXpoints,
                    box=hoomd.data.boxdim(Lx=box_dims[0], Ly=box_dims[1], Lz=box_dims[2]),
                    particle_types=['A', 'B', 'T'],
                    bond_types=['polymer1'])
    
    #Define polymer types:
    #   A: majority polymer
    #   B: minority polymer
    #   T: a template point (where we want minority spheres to form) 
    single_poly_type = np.zeros(Nab) #0 is type 'A'
    single_poly_type[Nab-Nb:Nab] = 1 #1 is type 'B'
    snapshot.particles.typeid[0:N_AB] = np.tile(single_poly_type,num_polymers)
    snapshot.particles.typeid[N_AB:N_AB + N_HEXpoints] = 2 #type 'T'
    
    #load location of polymer particles and templating particles.
    snapshot.particles.position[0:N_AB] = particle_coords
    snapshot.particles.position[N_AB:N_AB + N_HEXpoints] = HEXpoints
    snapshot.particles.diameter[N_AB:] = lattice_constant/2 #all template particles
    
    #Create the polymer bonds between neighbors
    neighbor_pairs = np.transpose([np.arange(0,Nab-1),np.arange(1,Nab)])
    neighbor_pairs = np.tile(neighbor_pairs,(num_polymers,1))
    offsets = np.repeat(np.arange(num_polymers),Nab-1)
    offsets = np.stack((offsets,offsets)).T
    offsets *= Nab
    neighbor_pairs += offsets
    snapshot.bonds.resize(len(neighbor_pairs))
    snapshot.bonds.group[:] = neighbor_pairs
    snapshot.bonds.typeid[:] = 0 # 'polymer1'
    
    #Extend the box in z, to avoid particles interacting across the wall
    #(There must be a less clumsy way to do this!)
    Lx = snapshot.box.Lx
    Ly = snapshot.box.Ly
    Lz = snapshot.box.Lz + 2 * lattice_constant
    snapshot.box = hoomd.data.boxdim(Lx = Lx, Ly = Ly, Lz = Lz)
    #print(snapshot.box)
    return(snapshot)
    
def define_walls():
    walls=hoomd.md.wall.group()
    # Walls are placed 0.5 units away from "thickness", since particle centers are 
    # generally 1 unit from the actual wall location.
    walls.add_plane(origin = (0,0,-thickness/2 -0.5), normal = (0,0,1))
    walls.add_plane(origin = (0,0,+thickness/2 +0.5), normal = (0,0,-1)) 
    ljw=hoomd.md.wall.lj(walls, r_cut=r_cut_glob)
    ljw.force_coeff.set('A', epsilon=epsilonwa, sigma=1.0, r_cut=r_cut_glob)
    ljw.force_coeff.set('B', epsilon=epsilonwb, sigma=1.0, r_cut=2**(1./6))
    ljw.force_coeff.set('T', epsilon=0.0, sigma=1.0,r_cut=False)
    
    if do_B_wall: #Different wall potential for type B particles
        #remove previous force on B particles
        ljw.force_coeff.set('B', epsilon=epsilonwb, sigma=1.0, r_cut=False)
        #Create New wall group for purely repulsive yukawa potential
        Bwalls=hoomd.md.wall.group()
        #Note that the r-shift below is different than for the LJ potenetial:
        Bwalls.add_plane(origin = (0,0,-thickness/2 +0.5), normal = (0,0,1))
        Bwalls.add_plane(origin = (0,0,+thickness/2 -0.5), normal = (0,0,-1)) 
        yukw=hoomd.md.wall.yukawa(Bwalls, r_cut=r_cut_glob)
        yukw.force_coeff.set(['A','T'], epsilon=0.1, kappa=1.0,r_cut=False)
        yukw.force_coeff.set('B', epsilon=5.0, kappa=5.0,r_cut=r_cut_glob)
        #With these coefficients (including a shift of the positional axis by 1.0
        #wrt the LJ potential), the +1 kT point on this potential lies about 0.5
        #further out than the LJ potential with epsilon=1.0.

def define_polymer_bonds():
    FENE = hoomd.md.bond.fene()
    #FENE.bond_coeff.set('polymer1', k=60.0, r0=1.6, sigma=1.1, epsilon= 1.0)
    #The FENE parameters were chosen to give a minimum at  about r = 1.0, 
    #With a restoring force similar to the harmonic bond previously used (k=1000)
    FENE.bond_coeff.set('polymer1', k=30.0, r0=1.5, sigma=1.0, epsilon= 0.0)

def define_lj_bonds():
    nl_lj = hoomd.md.nlist.cell()
    nl_lj.reset_exclusions(exclusions = [])
    lj = hoomd.md.pair.lj(r_cut=r_cut_glob, nlist=nl_lj)
    lj.set_params(mode="xplor")
    lj.pair_coeff.set('A', 'A', epsilon=epsilonaa, sigma=1.0, r_cut=r_cut_glob, r_on=2.0)
    lj.pair_coeff.set('A', 'B', epsilon=epsilonab, sigma=1.0, r_cut=r_cut_glob, r_on=2.0)
    lj.pair_coeff.set('B', 'B', epsilon=1.0, sigma=1.0, r_cut=r_cut_glob, r_on=2.0)
    lj.pair_coeff.set(['A','B', 'T'], 'T', epsilon=0.0, sigma=1.0, r_cut=False)

def do_disordering(kT=5.0, steps = 1e4, dump_period=2000):
    gsd_dump.set_period(dump_period) 
    integrator = hoomd.md.integrate.langevin(kT=kT, group=groupAB, seed=inp.se)
    hoomd.run(steps) #6000 is minimum for testing; 2e4 seems okay, with kT=9.0
    integrator.disable()

def do_templating(kT = 4.0, steps = 1e4, cooling_steps = 2e3, dump_period=2000):
    #Turn on the "templating" potential to form spheres in half-loop configuration
    #Initially I had used kT=1.0 and epsilon=-10, which worked fine.
    #But I found it much faster to increase to kT=4.0 and epsilon=-80
    gsd_dump.set_period(dump_period) 
    nl_gauss = hoomd.md.nlist.cell()
    gauss = hoomd.md.pair.gauss(r_cut=lattice_constant/2, nlist=nl_gauss)
    gauss_width = lattice_constant / 4
    gauss.pair_coeff.set('B', 'T', epsilon=-80.0, sigma = gauss_width, r_cut=0.6*lattice_constant)
    gauss.pair_coeff.set(['A','T'], 'T', epsilon=0.0, sigma=2.0, r_cut=False)
    gauss.pair_coeff.set(['A','B'], ['A','B'], epsilon=0.0, sigma=2.0, r_cut=False)
    gauss.set_params(mode="xplor")
    integrator = hoomd.md.integrate.langevin(kT=kT, group=groupAB, seed=inp.se)
    hoomd.run(steps) #1e4 looks complete.
    #This may not be necessary, but this gives time for Temperature to equilibrate
    #before turning off the Gaussian templating potential
    integrator.set_params(kT=inp.kT)
    hoomd.run(cooling_steps) 
    gauss.disable()
    integrator.disable()


# MAIN part of program begins here-------------------------------------------

hoomd.context.initialize()

#The following line parses commmand for the inputs in quotes.
#Entries for items (like "kT") are stored as "inp.kT".
#Values are also appended onto the output .gsd filename.
inp, filestring = mra.readargs(["base", "rho", "Bw", "epwa", "kT", "per", "se", "ts"])

#Set various parameters for simulation, either directly or from command line.
filestringbase = ''  #'hex_sph_'
lattice_width = 10 #width of box, in lattice constants, for templating particles
lattice_constant = 11.5 #spacing between templating particles.
                        #(use natural spacing between spherical domains)
thickness = 10 #How thick to make diblock film, (for single layer of spheres)
Nab = 20 #polymer length
Nb = 2 #polymer number of minority block
epsilonaa = 1.0
epsilonab = 0.3
epsilonwa = 3.5 #I've often used 3.5
epsilonwb = 1.0 #set very high, but cuttoff is r_cut=2^(1./6)
rho = inp.rho #particle density, generally around 0.8 or 0.9
r_cut_glob = 3.0
do_B_wall = inp.Bw
final_dump_period = inp.per
restart_period = final_dump_period

restart_filename = filestringbase + filestring + '_restart' + '.gsd'
try:
    system = hoomd.init.read_gsd(filename = restart_filename)
    restarting_from_file = True
except:
    system = hoomd.init.read_snapshot(define_initial_hex_snapshot())
    restarting_from_file = False

define_walls()
define_polymer_bonds()
define_lj_bonds()

#Define various hoomd GROUPS
groupall = hoomd.group.all()
groupA = hoomd.group.type(name="groupA", type='A')
groupB = hoomd.group.type(name="groupB", type='B')
groupAB = hoomd.group.union(name="groupAB", a=groupA, b=groupB)
groupT = hoomd.group.type(name="groupT", type='T')

integrator_mode = hoomd.md.integrate.mode_standard(dt=0.005)

if not restarting_from_file:
    gsd_dump = hoomd.dump.gsd(filestringbase + filestring + '.gsd',
                              period=2000, group=groupAB, overwrite=True)
    do_disordering(kT = 5.0, steps = 1e4, dump_period=2000)
    do_templating(kT = 4.0, steps = 1e4, cooling_steps=2000, dump_period=2000)
    gsd_dump.disable()
setup_steps = 22000 #sum of all steps in the previous four lines. Automate?

#This is the final "annealing" phase.
gsd_dump = hoomd.dump.gsd(filestringbase + filestring + '.gsd',
                          period=final_dump_period, 
                          phase = (setup_steps  -1 ) % final_dump_period + 1,
                          group=groupAB, dynamic=['momentum'], overwrite=False)
restart_dump = hoomd.dump.gsd(restart_filename, 
                          period=restart_period, 
                          phase = (setup_steps  -1 ) % restart_period + 1,
                          group = groupall, dynamic=['momentum'], truncate=True)
if not restarting_from_file:
    gsd_dump.write_restart()
    restart_dump.write_restart()
integrator = hoomd.md.integrate.langevin(kT=inp.kT, group=groupAB, seed=inp.se)
hoomd.run_upto(setup_steps + inp.ts + 1)
