#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 22:59:18 2018

@author: mtrawick
"""
import numpy as np
import matplotlib.pyplot as plt

def plot_Ugeom(P):
    fig, ax = plt.subplots()
    plt.ion
    plt.autoscale(enable=True,axis='both')
    ax.plot(P[:,0], P[:,1], 'ro', markersize=2)
    plt.show()

def print_counts_array(c):
    print('     Vertices of outer lattice (A) removed from rectangle: ', c[0])
    print('         "     "    "      "    "  removed from semicircle: ', c[1])
    print('     Vertices of inner lattice (B) replaced into rectangle: ', c[2])
    print('         "     "    "      "    "  replaced into semicircle: ', c[3])
    
def make_known_Ugeom(h, w, uh, uw, **kwargs):
    #This routine is now somewhat deprecated.  
    #The idea is that I ran make_Ugeom() a bunch of times with different
    #parameters (main routine at the end of this file) and found some good
    #combinations that worked well.  The problem is that it turns out that
    #the solution also depends on the length of the U-shape, not just how
    #wide it is. So no I recommend using make_optimal_Ugeom, below.
    known_parms = np.array([[6.0, 0, 0.25],
                            [9.5, 0, 0],
                            [13.0, 0, 0.25],
                            [15.5, 0.25, 0],
                            [18.0, 0, 0],
                            [19.0, 0.25, 0.25],
                            [21.5, 0, 0],
                            [22.5, 0.25, 0],
                            [25.0, 0, 0.25],
                            [28.0, 0, 0],
                            [31.0, 0.25, 0.25],
                            ])
    closest_uh_idx = np.argmin(abs(uh - known_parms[:,0]))
    UH = known_parms[closest_uh_idx, 0]
    Aoff = (0, known_parms[closest_uh_idx, 1])
    Boff = (0, known_parms[closest_uh_idx, 2])
    return(make_Ugeom(h, w, UH, uw, A_offset=Aoff, B_offset=Boff, **kwargs))

def make_optimal_Ugeom_known_widths(h, w, uh, uw, **kwargs_in):
    # As of 11/16/2018, this is my best solution to the problem of finding
    # good parameters for creating the U-shaped, "half-loop" geometry.
    # The problem is that with rotated lattices, only some widths are
    # sufficiently commensurate with the two lattice orientations to work well.
    # (If they don't work well, then the density of particles is way off.)
    # Essentially, this routine produces a geometry that's CLOSE to the with
    # uh requested, but optimizes uh as well as the offsets to produce a
    # geometry with a uniform density of spheres.
    
    #Below are widths (values of uh) that have produced good results before.
    known_good_widths = np.array([6, 9.5, 13, 15.5, 18, 19, 21.5, 22.5, 25.0, 
                                  28.0, 30.0])
    closest_uh_idx = np.argmin(abs(uh - known_good_widths))
    UH = known_good_widths[closest_uh_idx]

    #The kwargs theta1 and theta2 will get passed on to make_Ugeom.
    #But the kwarg "counts" gets replaced by my local arry cc, because I use
    #it in this function. 
#    if "counts" in kwargs_in:
#        print("counts is", kwargs_in["counts"])
    cc = np.zeros(4)
    kwargs_out = dict(kwargs_in, counts=cc)

    #Now try with different offsets to find the best possible set of parameters
    min_diff = 100000000
    bestargs = (0,0,0)
    for Ay in (0, 0.25):
        for By in (0, 0.25):
            for Bx in np.arange(0,0.7,0.2):
                args = (Ay, By, Bx)
                make_Ugeom(h, w, UH, uw, 
                           A_offset = (0, Ay), B_offset = (Bx, By),
                           **kwargs_out)
                diff = abs(cc[0] - cc[2]) + abs(cc[1] - cc[3])
                if diff < min_diff:
                    min_diff = diff
                    bestargs = (Ay, By, Bx)
                #print(args)
                #print_counts_array(cc)
                
    #Call make_Ugeom one more time, returning the result to the original call
    #of this function.
    Ay = bestargs[0]
    By = bestargs[1]
    Bx = bestargs[2]
    A_off = (0,Ay)
    B_off = (Bx, By) 
    #print(h, w, UH, uw, A_off, B_off)
    return(make_Ugeom(h, w, UH, uw, A_offset = A_off, B_offset = B_off, **kwargs_in))

def make_optimal_Ugeom(h, w, uh, uw, uh_tolerance=1.0, **kwargs):
    # As of 11/16/2018, this is my best solution to the problem of finding
    # good parameters for creating the U-shaped, "half-loop" geometry.
    # The problem is that with rotated lattices, only some widths are
    # sufficiently commensurate with the two lattice orientations to work well.
    # (If they don't work well, then the density of particles is way off.)
    # Essentially, this routine produces a geometry that's CLOSE to the with
    # uh requested, but optimizes uh as well as the offsets to produce a
    # geometry with a uniform density of spheres.
    
    #The kwargs theta1 and theta2 will get passed on to make_Ugeom.
    cc = np.zeros(4, dtype = int)

    # Now try with slightly different widths and offsets to find the best 
    # possible set of parameters
    min_diff = 100000000
    for UH in np.arange(uh-uh_tolerance, uh+uh_tolerance + 0.5, 0.5):
        for Ay in (0, np.sqrt(3)/4):
            for By in (0, 0.25):
                for Bx in np.arange(0,0.7,0.1):
                    make_Ugeom(h, w, UH, uw, 
                               A_offset = (0, Ay), B_offset = (Bx, By),
                               counts = cc, **kwargs)
                    diff = abs(cc[0] - cc[2]) + abs(cc[1] - cc[3])
                    if diff < min_diff:
                        min_diff = diff
                        bestargs = (UH, Ay, By, Bx)
                    #print(args)
                    #print_counts_array(cc)
                
    #Call make_Ugeom one more time, returning the result to the original call
    #of this function.
    UH = bestargs[0]
    Ay = bestargs[1]
    By = bestargs[2]
    Bx = bestargs[3]
    A_off = (0, Ay)
    B_off = (Bx, By)
    print("make_optimal_Ugeom: requested width ", uh, "returning", UH)
    P = make_Ugeom(h, w, UH, uw, A_offset = A_off, B_offset = B_off, 
                   counts=cc, **kwargs)
    print_counts_array(cc)
    print(" ")
#    #print("           scores:", UH,":   ",abs(cc[0] - cc[2]), abs(cc[1] - cc[3]), 
#          abs(cc[0] - cc[2]) + abs(cc[1] - cc[3]))
    return(P)    
                
def try_widths():
    for uh in np.arange(6, 30, 2):
        make_optimal_Ugeom(uh+20,100+12+int(uh/2),uh,100, uh_tolerance = 1)
    
    
def min_pair_distance(P):
    #for set of xy points P, return the minimum distance between any two pairs.
    mins = np.zeros(len(P)-1)
    for shift in range(1,len(P)):
        disps = P - np.roll(P, shift, axis=0)
        mins[shift -1] = np.min(np.linalg.norm(disps, axis=1))
    return(np.min(mins))

def edge_points(P, tolerance=1.0):
    w = np.where(((P[:,0]) - np.min(P[:,0]) <= tolerance) |
                 ((np.max(P[:,0]) - P[:,0]) <= tolerance) |
                 ((P[:,1]) - np.min(P[:,1]) <= tolerance) |
                 ((np.max(P[:,1]) - P[:,1]) <= tolerance) )
    return(w[0])
    

def rotate_coords(a, theta):
    theta_r = np.radians(theta)
    rot = np.array([[np.cos(theta_r),-np.sin(theta_r)],
                    [np.sin(theta_r), np.cos(theta_r)]])
    return(rot.dot(a.T).T)

def p_in_rect(p, rect, buffer=0):
    inside = ((p[:,0] >= rect[0] - buffer) &
              (p[:,0] <= rect[1] + buffer) &
              (p[:,1] >= rect[2] - buffer) &
              (p[:,1] <= rect[3] + buffer) )
    return(inside)

def p_in_circle(p, circle, buffer=0):
    #print(p)
    disp_from_cent = p - np.array([circle[0],circle[1]])
    dist_from_cent = np.linalg.norm(disp_from_cent, axis=1)
    #print(dist_from_cent)
    inside = (dist_from_cent <= circle[2] + buffer)
    return(inside)

def make_Ugeom(h, w, uh, uw, theta1=0, theta2=30, 
               A_offset = 0, B_offset = 0, 
               counts=np.zeros(4)):
    # This function creates a set of points in the half-loop geometry.
    # Required inputs:
    # h, w: height and width of whole arrangement
    # uh, uw, height and width of U shape, from edge to center of circle.
    # Optional inputs:
    # theta1, theta2: rotation of lattices A and B in degrees
    # A_offset, B_offsets: offsets for the two lattices, as ndarrays.
    # counts: an array that will hold 4 numbers: [a, b, c, d], where
    # a = number of spheres in lattice A removed from the rectrangle
    # b = number of spheres in lattice A removed from the half-circle
    # c = number of spheres in lattice B put back in the rectrangle
    # d = number of spheres in lattice B put back in the half-circle
    
    sqr3 = np.sqrt(3)
    unitvec = np.array([0.5, sqr3/2])
    #print(unitvec)

    #Use np.mgrid to make a list of xy points A in a hexagonal lattice.
    #Make the list bigger than we need.
    big = max([h,w]) * 3
    N = 2*big + 1
    A = np.mgrid[-big:big+1:1, -big:big+1:1]
    A = A.reshape(2,N*N).transpose()
    A = A.astype(float)
    A[:,1] *= np.sqrt(3)
    A = np.concatenate((A, A + unitvec))
    
    #Rotate and offset two lattices:
    B = rotate_coords(A, theta2)
    B += B_offset
    A = rotate_coords(A, theta1)
    A += A_offset

    #Remove the ones that are NOT within the outer boundaries (w x h rectangle)   
    edge_rect = (0, w, -h/2, h/2)
    A = A[np.where(p_in_rect(A, edge_rect))]
    B = B[np.where(p_in_rect(B, edge_rect, buffer=0.0001))]
    
    #Now we remove points from lattice A that are within the U-shape...
    Urect = (0, uw, -uh/2, uh/2)
    w1 = np.where(~p_in_rect(A, Urect))
    counts[0] = len(A) - len(w1[0])
    A = A[w1]

    Ucircle = (uw, 0, uh/2)
    w2 = np.where(~p_in_circle(A, Ucircle))
    counts[1] = len(A) - len(w2[0])
    A = A[w2]
    

    #...and replace them with points from lattice B that within the U-shape
    w3 = np.where(p_in_rect(B, Urect, buffer=0.0001))
    notw3 = np.where(~p_in_rect(B, Urect, buffer=0.0001))
    Binrect = B[w3]
    counts[2] = len(Binrect)
    Bnotinrect = B[notw3]
    
    w4 = np.where(p_in_circle(Bnotinrect, Ucircle))
    Binhalfcircle = Bnotinrect[w4]
    counts[3] = len(Binhalfcircle)

    B = np.concatenate((Binrect, Binhalfcircle)) 
    
    AandB = np.concatenate((A,B))
    return(AandB)


if __name__ == '__main__':

    #This main procedure cycles through different widths and offsets
    #To find numbers for which the lattices are reasonably commensurate in
    #terms of the numbers of spheres removed and replaced.
    UW = 60
    min_border = 10
    
    c = np.zeros(4)
    for UH in np.arange(24,32,0.5):
        H = UH + min_border*2
        W = UW + min_border + int(UH/2)
        for Ay_offset in (0, 0.25):
            for By_offset in (0, 0.25):
                Ashift = np.array([0, Ay_offset])
                Bshift = np.array([0, By_offset])
                print("UH, Ay_offset, By_offset = ", UH, Ay_offset, By_offset)
                print("    H, W, UH, UW = ", H, W, UH, UW, Ay_offset, By_offset)
                p_arr = make_Ugeom(H, W, UH, UW, 
                    A_offset=Ashift, B_offset=Bshift, theta2 = 30, counts=c)
                print("    counts from make_Ugeom", c, min_pair_distance(p_arr))
                if (abs(c[2] - c[0]) <= UH/2 and abs(c[3] - c[1]) <= 3):
                    plot_Ugeom(p_arr)
                #(6, 0.25), (9,0), (13, 0.25)
                #with A offset by sqr3/4, (7.2,0) works well too

 