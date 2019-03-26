#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#Created on Thu Nov 15 21:02:24 2018
#
#
# This module (actually just one function currently) is designed to 
# parse command line arguments for a hoomd script, and produce a meaningful
# string to use for a resulting output filename.
#
# A typical use for the function might be:
#       inp, filestring = mra.readargs(["uh", "uw", "kT", "se", "ts"])
#
# In this case, the function readargs does 4 things:
# (1) set up the "argparse" parser to read ONLY those five arguments
# from the command line.  The command line has to be given as something like:
#       --user="--uh=19 --uw=100 --se=43 --kT=0.3 --ts=300000"
# because argments not given in the --user= string are parsed by hoomd instead.
#
# (2) Read those arguments from the command line, raising errors if something
# isn't right.
#
# (3) Returns values of the parameters in "inp", to be accessed in the
# hoomd script as inp.uh, inp.uw, inp.kT, etc.
#
# (4) Returns a meaningful string like "uh19_uw100_se43_kT0.3_ts300000" in 
# "filestring", which can be used as part of an output file name if desired.
#
# Note that one can load up the definition of readargs, below, with a ton of
# different parameters that you MIGHT use, but you can decide to only use a
# small subset of them when you call the function.  This means you don't have
# to rewrite readargs for every little test you do.
#
# Also, although I've used 2-letter codes below to help me remember them,
# there's no restriction on what you can use as a key.

import hoomd
import argparse

def readargs(arglist):
    parser = argparse.ArgumentParser(description=
        'Script for messing with lennard-jones particles bonded as polymers')
    
    if "base" in arglist:
        parser.add_argument('basename', action='store', nargs='?',
                            default = 'output',
                            help='String for base of filename')
    if "per" in arglist:
        parser.add_argument('--per', dest='per', action='store',
                            default=10000, type=int,
                            help='gsd dump period for final anneal stage')
    if "wi" in arglist:
        parser.add_argument('--wi', dest='wi', action='store',
                            default=100, type=int,
                            help='width of simulation box')
    if "rho" in arglist:
        parser.add_argument('--rho', dest='rho', action='store',
                            default=0.826, type=float,
                            help='Volume density of polymer')
    if "Bw" in arglist:
        parser.add_argument('--Bw', dest='Bw', action='store',
                            default=0, type=int,
                            help='Boolean: extra wall for B-type particles')
    if "th" in arglist:
        parser.add_argument('--th', dest='th', action='store',
                            default=10, type=int,
                            help='Thickness of film')
    if "epwa" in arglist:
        parser.add_argument('--epwa', dest='epwa', action='store',
                            default=2.0, type=float,
                            help='LJ parameter epsilon between walls and type A particles')
    if "epab" in arglist:
        parser.add_argument('--epab', dest='epab', action='store',
                            default=0.4, type=float,
                            help='LJ parameter epsilon between types A and B particles')
    if "Nab" in arglist:
        parser.add_argument('--Nab', dest='Nab', action='store',
                            default=20, type=int,
                            help='Total number of A and B particles in polymer')
    if "uh" in arglist:
        parser.add_argument('--uh', dest='uh', action='store',
                            default=9, type=int,
                            help='Height of half-loop')
    if "uw" in arglist:
        parser.add_argument('--uw', dest='uw', action='store',
                            default=50, type=int,
                            help='Width of half-loop, or more like "length"')
    if "se" in arglist:
        parser.add_argument('--se', dest='se', action='store',
                            default=4, type=int,
                            help='random seed')
    if "kT" in arglist:
        parser.add_argument('--kT', dest='kT', action='store',
                            default=1.0, type=float,
                            help='Final temperature kT')
    if "ts" in arglist:
        parser.add_argument('--ts', dest='ts', action='store',
                            default=10000, type=int,
                            help='Number of timesteps to run')
        

    inputs = parser.parse_args(hoomd.option.get_user())

    argdict = dict(vars(inputs))  #dict makes a copy
    argdict.pop('basename', None)
    argdict.pop('ts', None)
    argdict.pop('per', None)
    
    filestring = str(argdict)
    filestring = filestring.replace(" ", "").replace("'", "")
    filestring = filestring.replace("{", "").replace("}", "")
    filestring = filestring.replace(":", "").replace(",", "_")
    if "base" in arglist:
        filestring = inputs.basename + '_' + filestring
    return(inputs, filestring)