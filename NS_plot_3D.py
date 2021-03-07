import numpy as np
import matplotlib.pyplot as plt

location = "./data/"	# location of file

nT   = int(np.loadtxt(location + "dims.dat")[0])	# total time
nX   = int(np.loadtxt(location + "dims.dat")[1])	# X length
nY   = int(np.loadtxt(location + "dims.dat")[2])	# Y length
nZ   = int(np.loadtxt(location + "dims.dat")[3])	# Z length
k	 = int(np.loadtxt(location + "dims.dat")[13])	# number of grids
Phim = int(np.loadtxt(location + "dims.dat")[12])

'''###############################################
##### Animation ##################################
###############################################'''

fig, (ax11,ax12,ax13) = plt.subplots(1, 3)

for nt in range(0,k):
	P  = np.load(location + "phi_" + str(nt) + ".npy")
	ax11.imshow(P[int(1+nZ/2),:,:], cmap='gist_stern', interpolation='none',
	vmin = 0 ,vmax = Phim)
	ax11.set_axis_off()
	ax12.imshow(P[:,int(1+nY/2),:], cmap='gist_stern', interpolation='none',
	vmin = 0 ,vmax = Phim)
	ax12.set_axis_off()
	ax13.imshow(P[:,:,int(1+nX/2)], cmap='gist_stern', interpolation='none',
	vmin = 0 ,vmax = Phim)
	ax13.set_axis_off()
	if nt < 10:
		plt.savefig(location + "phi_00" + str(nt) + ".jpg")
	elif nt < 100:
		plt.savefig(location + "phi_0" + str(nt) + ".jpg")
	else:
		plt.savefig(location + "phi_" + str(nt) + ".jpg")
