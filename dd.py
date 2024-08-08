from quspin.basis import spinful_fermion_basis_general
from quspin.operators import hamiltonian
import numpy as np
#
###### define model parameters ######
Lx, Ly = 3, 3 # linear dimension of spin 1 2d lattice
N_2d = Lx*Ly # number of sites for spin 1
t=1           # 最近邻跃迁强度
t1=0.3*t
delta_t=0.9*t1  # 次邻跃迁的差值
tR=t1+delta_t  # 次邻跃迁强度
tL=t1-delta_t  # 次邻跃迁强度
U=3.5 # onsite interaction
###### setting up user-defined BASIC symmetry transformations for 2d lattice ######
s = np.arange(N_2d) # sites [0,1,2,...,N_2d-1] in simple notation
x = s % Lx # x positions for sites
y = s // Lx # y positions for sites
T_x = (x+1)%Lx + Lx*y # translation along x-direction
T_y = x + Lx*((y+1)%Ly) # translation along y-direction
T_xy1 = ((x + 1) % Lx) + Lx * ((y + 1) % Ly)  # x 和 y 方向分别向右向下平移一个格点的次近邻
T_xy2 = ((x - 1) % Lx) + Lx * ((y + 1) % Ly)  # x 和 y 方向分别向左向下平移一个格点的次近邻
#S = -(s+1) # fermion spin inversion in the simple case
###### setting up bases ######
basis_2d=spinful_fermion_basis_general(N_2d,Nf=(5,4),double_occupancy=True)
					#kxblock=(T_x,0),kyblock=(T_y,0),kxy1block=(T_xy1,0),kxy2block=(T_xy2,0))
print(basis_2d)
###### setting up hamiltonian ######
# setting up site-coupling lists for simple case
hopping_left = [[-t,i,T_x[i]] for i in range(N_2d)] + [[-t,i,T_y[i]] for i in range(N_2d)]
hopping_right = [[+t,i,T_x[i]] for i in range(N_2d)] + [[+t,i,T_y[i]] for i in range(N_2d)]
interaction=[[U,i,i] for i in range(N_2d)]
#
static=[["+-|",hopping_left], # spin up hops to left
		["-+|",hopping_right], # spin up hops to right
		["|+-",hopping_left], # spin down hopes to left
		["|-+",hopping_right], # spin down hops to right
		["n|n",interaction]] # spin up-spin down interaction
#
# 初始化一个长度为N_2d的数组，表示所有格点属于B子格
subgrid = np.zeros(N_2d, dtype=int)
# 将偶数索引的格点标记为1，表示属于A子格
subgrid[::2] = 1
for i in range(N_2d):
	if subgrid[i] == 1:
		# A格点到A格点的跃迁
		hopping_pm1 = [[+tR,i,T_xy1[i]] for i in range(N_2d)]
		hopping_pm2 = [[+tL,i,T_xy2[i]] for i in range(N_2d)]
		hopping_mp1 = [[-tR,i,T_xy1[i]] for i in range(N_2d)]
		hopping_mp2 = [[-tL,i,T_xy2[i]] for i in range(N_2d)]
	else:
		# B格点到B格点的跃迁
		hopping_pm1 = [[+tL,i,T_xy1[i]] for i in range(N_2d)]
		hopping_pm2 = [[+tR,i,T_xy2[i]] for i in range(N_2d)]
		hopping_mp1 = [[-tL,i,T_xy1[i]] for i in range(N_2d)]
		hopping_mp2 = [[-tR,i,T_xy2[i]] for i in range(N_2d)]
		
static.append(["+-|", hopping_pm1]) 
static.append(["-+|", hopping_mp1]) 
static.append(["+-|", hopping_pm2])
static.append(["-+|", hopping_mp2])
static.append(["|+-", hopping_pm1])
static.append(["|-+", hopping_mp1])
static.append(["|+-", hopping_pm2]) 
static.append(["|-+", hopping_mp2])
# build hamiltonian
H=hamiltonian(static,[],basis=basis_2d,dtype=np.float64)
# compute GS of H
E_GS, psi_GS=H.eigsh(k=1,which='SA')
print(E_GS)
