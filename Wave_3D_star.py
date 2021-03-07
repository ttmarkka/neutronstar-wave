import numpy as np
import matplotlib.pyplot as plt
import csv
from random import seed
from random import random
import os
from time import time

location = "./data/"

'''###############################################
### Initialization ###############################
##################################################'''


pi    = 3.14159265359
Mp    = 2.435*10**(18)				     # in GeV
seed(1)

# Theory parameters #
xi    = 100
la 	  = 10**-20
m 	  = 0

# NS parameters #
rn    = 5*10**(19)						 # radius in (GeV)**-1
M	  = 1.6*10**(57)					 # mass in GeV
ro	  = 2.5*rn							 # orbital radius
To	  = 2*pi*(32*pi*Mp**2*ro**3/M)**(1/2)# orbital period

# Simulation parameters #
boost = 5								 # Increase velocity, 1 = no boost

L	  =  8*ro							 # size of box
g	  =  50								 # damping parameter
T	  = (1/boost)*(12/4)*To				 # duration
Td1   = (1/boost)*(1/2)*To			 	 # start of damping
Td2   = (1/boost)*(4/4)*To				 # end of damping
Ts	  = (1/boost)*(4/4)*To				 # start of rotation
Dt	  = To*5*10**-4					 	 # stepsize
Dx    = rn*0.2							 # stepsize
rs1	  = 7*ro							 # inner radius of area where energy is not evaluated
rs2	  = 9*ro							 # outer radius of area where energy is not evaluated
k	  = 120								 # Number of saved grids

# Lattice #
nX	  = int(L/Dx)
nY	  = nX
nZ	  = nX
nT	  = int(T/Dt)
nTo	  = int(To/Dt)
nTs	  = int(Ts/Dt)
nTd1  = int(Td1/Dt)
nTd2  = int(Td2/Dt)
nn	  = int(rn/Dx)
ns1	  = int(rs1/Dx)
ns2	  = int(rs2/Dx)
nro	  = int(ro/Dx)
dt	  = Dt/To
dx	  = Dx/rn
dy    = dx
dz	  = dx

# Derived constants
C	 = (Dt/Dx)**2
B    = 96*pi**2*ro**3/rn**3
mm	 = m**2*(128*pi**3*Mp**2*ro**3/M)
Phim = (B*xi/(2*la))**(1/2)

# Arrays #
P		 = np.array([0.0]*(nX+3)*(nY+3)*(nZ+3)*(3)).reshape(
	(3),(nZ+3),(nY+3),(nX+3)) # Generates a 3D array of floats for 3 time slices
DPDPdtdt = np.array([0.0]*(nX+3)*(nY+3)*(nZ+3)).reshape(
	(nZ+3),(nY+3),(nX+3))	# RHS of the wave equation
noE		 = np.array([1]*(nX+3)*(nY+3)*(nZ+3)).reshape(
	(nZ+3),(nY+3),(nX+3))	# Volume where energy is measured
Sg		 = [int(i*nT/k) for i in range(1,k+1)]	# Time coordinates of saved grids,
# spread evenly in t

'''###############################################
####### Boundary Conditions & initialization #####
##################################################'''


# Zeroes on the boundary
# Initially shot noise
# (exponentilly decaying
# when close to the edges)
# Initial time t = 1

for nz in range(1,nZ+2):
	for ny in range(1,nY+2):
		for nx in range(1,nX+2):
			if ((nX/2+1-nx)**2 + (nY/2+1-ny)**2 + (nZ/2+1-nz)**2 < ns1**2) or\
				((nX/2+1-nx)**2 + (nY/2+1-ny)**2 + (nZ/2+1-nz)**2 > ns2**2):
				noE[nz,ny,nx] = 0
			if ny > int(nY/2+1):
				P[1,nz,ny,nx] =  (1+random())*np.exp(
					-((nX/2+1-nx)**2 + (nY/2+1-ny)**2 + (nZ/2+1-nz)**2)/nro**2)
			else:
				P[1,nz,ny,nx] =  (1+random())*np.exp(
					-((nX/2+1-nx)**2 + (nY/2+1-ny)**2 + (nZ/2+1-nz)**2)/nro**2)
P[1] = (10**-2)*Phim*P[1]	# normalize peak to maximum times small number

'''From the initial condition DP[1,z,y,x]/dt = f[x]
and the wave equation solve for the phantom point
(t = 0) with symmetric derivative
(better accuracy than forward)
DP[1,z,y,x]/dt = (P[2,z,y,x] - P[0,y,z,x])/(2dt)'''

# Static peak => DP[1,z,y,x]/dt = 0

for nz in range(1,nZ+2):
	for ny in range(1,nY+2):
		for nx in range(1,nY+2):
			DPDPdtdt[nz,ny,nx] = C*(P[1,nz,ny,nx+1]-2*P[1,nz,ny,nx]+P[1,nz,ny,nx-1])\
			 + C*(P[1,nz,ny+1,nx]-2*P[1,nz,ny,nx]+P[1,nz,ny-1,nx])\
			 + C*(P[1,nz+1,ny,nx]-2*P[1,nz,ny,nx]+P[1,nz-1,ny,nx])\
			 	- la*dt**2*P[1,nz,ny,nx]**3
			P[2,nz,ny,nx] =  P[1,nz,ny,nx] + (1/2)*DPDPdtdt[nz,ny,nx]
P[0] = P[2]

'''###############################################
#### Solver for the 3D Wave equation #############
#### on a background with a neutron star binary ##
##################################################'''

start = time()
with open(location + "energy.dat", "w") as data:
	writer = csv.writer(data, delimiter='\t')
	for nt in range(2,nT+2):
		P[0] = P[1]
		P[1] = P[2]
		e    = 0																 # Energy variable
		V    = dx*dy*dz*np.sum(noE)												 # Volume of region (lattice units)
		G    = g*np.heaviside(nt-nTd1,0)*(1-np.heaviside(nt-nTd2,0))			 # Damping on nTd1, off at nTd2
		W    = boost*((2*pi)/nTo)*np.heaviside(nt-nTs,0)						 # Rotation at Ts,
		ns1x = 1+nX/2 + nro*np.sin(W*(nt-nTs))	 					  			 # Center of star 1
		ns1y = 1+nY/2 + nro*np.cos(W*(nt-nTs))									 # Center of star 1
		ns1z = 1+nZ/2  															 # Center of star 1
		ns2x = 1+nX/2 - nro*np.sin(W*(nt-nTs))									 # Center of star 2
		ns2y = 1+nY/2 - nro*np.cos(W*(nt-nTs))									 # Center of star 2
		ns2z = 1+nZ/2 															 # Center of star 2
		for nz in range(1,nZ+2):
			for ny in range(1,nY+2):
				for nx in range(1,nX+2):
					# Scalar curvature
					R  = B*np.exp(-((ns1x-nx)**2 + (ns1y-ny)**2 + (ns1z-nz)**2)/nn**2) \
					+ B*np.exp(-((ns2x-nx)**2 + (ns2y-ny)**2 + (ns2z-nz)**2)/nn**2)
					# EOM
					DPDPdtdt[nz,ny,nx] = C*(P[1,nz,ny,nx+1]-2*P[1,nz,ny,nx]\
					+P[1,nz,ny,nx-1]) + C*(P[1,nz,ny+1,nx]-2*P[1,nz,ny,nx]+P[1,nz,ny-1,nx])\
					+ C*(P[1,nz+1,ny,nx]-2*P[1,nz,ny,nx]+P[1,nz-1,ny,nx])\
					+ dt**2*(-mm + R*xi)*(P[1,nz,ny,nx]) - dt**2*la*P[1,nz,ny,nx]**3\
					- G*dt*(P[1,nz,ny,nx] - P[0,nz,ny,nx])# Last term gives damping
					P[2,nz,ny,nx] =  2*P[1,nz,ny,nx] - P[0,nz,ny,nx] + DPDPdtdt[nz,ny,nx]
					# Energy, given in units of rn**3*To**-4 #
					if noE[nz,ny,nx] > 0:
						e = e + dx*dy*dz*((1/2)*(P[2,nz,ny,nx] - P[0,nz,ny,nx])**2/(2*dt)**2)\
						+ dx*dy*dz*(C*(1/2)*(P[1,nz,ny,nx+1] - P[1,nz,ny,nx-1])**2/(2*dt)**2)\
						+ dx*dy*dz*(C*(1/2)*(P[1,nz,ny+1,nx] - P[1,nz,ny-1,nx])**2/(2*dt)**2)\
						+ dx*dy*dz*(C*(1/2)*(P[1,nz+1,ny,nx] - P[1,nz-1,ny,nx])**2/(2*dt)**2)\
						+ dx*dy*dz*((1/2)*(mm - R*xi)*(P[1,nz,ny,nx])**2\
						+ (1/4)*la*P[1,nz,ny,nx]**4)
		energy = (dx*dy*dz)*(nX*nY*nZ)*(e/V)	 # Energy of the particles in the box
		writer.writerow([dt*nt,energy])
		if nt in Sg:
			np.save(location + "phi_" + str(Sg.index(nt)) + ".npy", P[1])

		# Print percent done
		print("  "+str(int(100*nt/(nT+1)))+"%", end="\r")
data.close()

with open(location + "dims.dat", "w") as data:
	writer = csv.writer(data, delimiter='\t')
	writer.writerow([int((nT+1)/(100/k)),
		nX,nY,nZ,nTo,nT,nn,nro,ns1,ns2,xi,la,Phim,k])
data.close()

#os.system("systemctl suspend")
