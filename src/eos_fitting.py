#!/usr/bin/env python3

"""
This code calcualtes the equation of state (EoS).
The input file should be volume (V) vs. energy (E).
V and E can be in different units, the default being ang**3 and eV.
Pressure (P) vs. V is not present in birch and p3.
"""

import os
import sys
import time
import numpy as np

import eos

sys.path.append(os.getcwd())


# constants
Ry2eV=13.605698066
ang32au=6.74833449394997


# Input and Output functions
def readev(evfile):
    """
    read Energy-Volume data from evfile
    """
    En=[]
    Vol=[]
    with open(evfile,'r') as f:
        f.readline()
        for line in f:
            tmp=line.strip().split()
            En.append(float(tmp[1]))
            Vol.append(float(tmp[0]))
    return np.array(En), np.array(Vol)


def write_to_file(filename,text):
    with open(filename, "a") as f:
        f.write(text)
    return


if __name__ == '__main__':
    start_time_total = time.time()
    # all parameters from thermo_config
    os.remove('eos_fitted_data.txt') if os.path.exists('eos_fitted_data.txt') else None
    
    E,V=readev("input")
       
    eos.saveEOSfitdata(V,E)
    #eos.mergeplots()
 
    # show the running time
    end_time_total = time.time()
    time_elapsed = end_time_total-start_time_total
    print("time elapsed: " + '{:8.2f}'.format(time_elapsed) + " seconds")
