import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from astropy import constants as const
from astropy import units as u


########## Part 1

def dSstate_dr(r, S):
    '''
    Args:

    r (array-like): dependant variable times
    S (array-like): state vector [p, m]
   
    Returns:

    dSstate (array): state vector [dp/dr, dm/dr]
    '''
    dSstate = [0,0] # ie [dp/dr, dm/dr]
    p = S[0]
    m = S[1]

    x = np.sign(p) * (np.abs(p)) ** (1/3) # numpy throws warning for fractional powers of negative numbers
    gamma = (x**2)/(3*((1+x**2)**0.5))
 
    #derivatives 
    dSstate[0] = -(m*p)/(gamma*r**2)
    dSstate[1] = (r**2)*p
    
    return dSstate

# This is the function that returns the thing we want to 'zero' - events in solve ivp makes this equal to 0
# In this case, the density will go to 0 at our desired max radius
def maxR(r, S):
    '''
    Args:

    r (array-like): dependant variable times
    S (array-like): state vector [p, m]
   
    Returns:

    S[0]: event for which we want to find the root of; returns S[0] = 0 for small enough values of S[0]
    '''
    if S[0] < 1e-5: # events detects a 0 via a sign change; since the density will not be negative, we set a close enough bound to prevent infinite iteration
        return 0
    else:
        return S[0]

########## Part 2

pc = np.logspace(-1, np.log(2.5e6), 10, endpoint=True)
ue = 2
R0 = (7.72*10**8/ue)
M0 = 5.67*10**33/(ue**2)
p0 = 9.74*10**5*ue*const.g0

fig, ax = plt.subplots() 

maxR.terminal = True # Will terminate once event has been reached. Value of interest will be the singular value in sol.t_events/y_events or the last value in sol.t/y
for i in range(len(pc)):
    sol = sc.integrate.solve_ivp(dSstate_dr, (1e-8, 1e5), np.array([pc[i],0]), method = 'RK45', events = maxR)
    radius = ((sol.t_events[0][0]*R0)*u.cm).to(u.solRad) # converting radius to solar radii
    mass = ((sol.y_events[0][0][1]*M0)*u.g).to(u.solMass) # converting mass to solar mass
    ax.plot(mass, radius, 'bo')

ax.set_title('White Dwarf: Radius vs Mass')  # title and axis labels
ax.set_xlabel('Mass (solar mass units)') 
ax.set_ylabel('Radius (solar radii)')
plt.axvline(mass.value, color = 'r') # set a vertical line at the last tested value; is close to radius = 0  
plt.grid(True)  # adding grid
plt.savefig('TsouSonya_Project3_Fig1_RvMplot.png') # save to file

print(f"Estimated Chandrasekhar limit: {mass}")
print(f"Kippenhahn & Weigert (1990) MCh: {5.836/(ue**2)*u.solMass} ")
print(f"Estimated mass limit is about {round(((mass.value-5.836/(ue**2))/(5.836/(ue**2)))*100, 3)}% more")
plt.show()

########## Part 3

import pandas as pd #for simpler table comparaison

pc = np.logspace(-1, np.log(2.5e6), 3, endpoint=True)
methods = ['RK45', 'BDF']

vals3 = np.zeros((len(pc), len(methods)*3))

for i in range(len(pc)):
    for method in methods:
        sol = sc.integrate.solve_ivp(dSstate_dr, (1e-8, 1e5), np.array([pc[i],0]), method = method, events = maxR)
        rad = (sol.t_events[0][0]*R0) 
        mass = (sol.y_events[0][0][1]*M0)

        vals3[i, methods.index(method)] = rad
        vals3[i, methods.index(method)+3] = mass 

vals3[:,2] = (np.abs(vals3[:,0]-vals3[:,1])/((vals3[:,0]+vals3[:,1])*0.5))*100 # 3rd col is percent diff between the radii calculated for each method
vals3[:,5] = (np.abs(vals3[:,3]-vals3[:,4])/((vals3[:,3]+vals3[:,4])*0.5))*100 # 5th col is percent diff between the mass calculated for each method

vals3[:,0:2] = (vals3[:,0:2]*u.cm).to(u.solRad) #converting units
vals3[:,3:5] = (vals3[:,3:5]*u.g).to(u.solMass)

print("\nPart 3")
print(pd.DataFrame(vals3, columns=['Radius: RK45', 'Radius: BDF', '% Diff', 'Mass: RK45', 'Mass: BDF', '% Diff'], index = pc))
# Results are less than or about 1% different at most



########## Part 4

data = np.loadtxt('TsouSonya_Project3/wd_mass_radius.csv', delimiter = ',', skiprows=1) 
M_Msun = data[:, 0]  
M_unc = data[:, 1]  
R_Rsun = data[:, 2]
R_unc = data[:, 3]

# Making interpolation function. This will allow checking if the computed ODE is within range of an observed radius/mass with error    
radius_list = []
mass_list=[]
pc = np.logspace(-1, np.log(2.5e6), 1000, endpoint=True)

for i in range(len(pc)):
    sol = sc.integrate.solve_ivp(dSstate_dr, (1e-8, 1e5), np.array([pc[i],0]), method = 'RK45', events = maxR)
    radius = ((sol.t_events[0][0]*R0)*u.cm).to(u.solRad) # converting radius to solar radii
    mass = ((sol.y_events[0][0][1]*M0)*u.g).to(u.solMass) # converting mass to solar mass
    radius_list.append(radius.value)
    mass_list.append(mass.value)

f = sc.interpolate.interp1d(mass_list, radius_list)

plt.errorbar(x=M_Msun, y=R_Rsun, yerr = R_unc, xerr = M_unc, fmt='o')
plt.plot(mass_list[::2], f(mass_list[::2]), '-')
plt.xlabel('Mass (solar mass units)')
plt.ylabel('Radius (solar radii)')
plt.grid(True)
plt.show()
