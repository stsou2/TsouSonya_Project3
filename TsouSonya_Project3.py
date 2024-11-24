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

def solve_pc(pc, method = 'RK45', solar = True):
    '''
    Args:
    pc (array-like): List of pc values to test
    method (string): solve_ivp method to use, defaults to the solve_ivp default RK45
    solar (boolean): Converts radius and mass to solar units if True, else leaves in cgs

    Returns:
    radius_arr (array): list of maximum radius for given pc values
    mass_arr (array): list of maximum mass for given pc values
    '''
    radius_arr = np.zeros(len(pc))
    mass_arr = np.zeros(len(pc))
    maxR.terminal = True # Will terminate once event has been reached. Value of interest will be the singular value in sol.t_events/y_events or the last value in sol.t/y

    for i in range(len(pc)):
        sol = sc.integrate.solve_ivp(dSstate_dr, (1e-8, 1e5), np.array([pc[i],0]), method = method, events = maxR)
        radius = sol.t_events[0][0]*R0*u.cm # converting dimensionless radius to cgs
        mass = sol.y_events[0][0][1]*M0*u.g # converting dimensionless mass to cgs
        if solar == True: # convert cgs to solar units
            radius = radius.to(u.solRad)
            mass = mass.to(u.solMass)

        radius_arr[i] = radius.value # np array does not allow storage of dimensional quantities
        mass_arr[i] = mass.value
  
    return radius_arr, mass_arr



########## Part 2

# Declare constants
ue = 2
R0 = (7.72*10**8/ue)
M0 = 5.67*10**33/(ue**2)
p0 = 9.74*10**5*ue*const.g0

# Testing for 10 values of pc
pc = np.logspace(-1, np.log10(2.5e6), 10, endpoint=True)
radius_arr, mass_arr = solve_pc(pc)

# Plotting
fig, ax = plt.subplots() 
ax.scatter(mass_arr, radius_arr)
ax.set_title('White Dwarf: Radius vs Mass')  # title and axis labels
ax.set_xlabel('Mass (solar mass units)') 
ax.set_ylabel('Radius (solar radius)')
plt.axvline(mass_arr[-1], color = 'r') # set a vertical line at the last tested value; is very close to radius = 0  
plt.grid(True)  # adding grid
plt.savefig('TsouSonya_Project3_Fig1_RvMplot.png') # save to file

print(f"Estimated Chandrasekhar limit: {mass_arr[-1]*u.solMass}")
print(f"Kippenhahn & Weigert (1990) MCh: {5.836/(ue**2)*u.solMass} ")
print(f"Estimated mass limit is about {round(((mass_arr[-1]-5.836/(ue**2))/(5.836/(ue**2)))*100, 3)}% more")
#plt.show()


########## Part 3

import pandas as pd # just for simpler table making with headers

# Testing default RK45 and different BDF solve_ivp methods for 3x pc values
pc = np.logspace(-1, np.log10(2.5e6), 3, endpoint=True)
radRK45, massRK45 = solve_pc(pc, method = 'RK45')
radBDF, massBDF = solve_pc(pc, method = 'BDF')

# Finding percent differences using |A-B|/((A+B)/2) * 100 percent diff formula
diff_rad = np.abs(radRK45-radBDF)/((radRK45+radBDF)*0.5) * 100
diff_mass = np.abs(massRK45-massBDF)/((massRK45+massBDF)*0.5) * 100

print("\nPart 3")
print(pd.DataFrame(data = [pc, radRK45, radBDF, diff_rad, massRK45, massBDF, diff_mass], index=['Pc Value', 'Radius: RK45', 'Radius: BDF', '% Diff', 'Mass: RK45', 'Mass: BDF', '% Diff']).T)
# Results are less than or about 1% different at most



########## Part 4

# Loading data. I have the csv file stored in my github repo folder within my open folder hence the load name.
data = np.loadtxt('TsouSonya_Project3/wd_mass_radius.csv', delimiter = ',', skiprows=1) 
M_Msun = data[:, 0]  
M_unc = data[:, 1]  
R_Rsun = data[:, 2]
R_unc = data[:, 3]

# Making interpolation function to plot computed ODE solution with
pc = np.logspace(-1, np.log10(2.5e6), 1000, endpoint=True)
radius_arr, mass_arr = solve_pc(pc)
f = sc.interpolate.interp1d(mass_arr, radius_arr)

# Plotting
fig, ax2 = plt.subplots()
ax2.errorbar(x=M_Msun, y=R_Rsun, yerr = R_unc, xerr = M_unc, fmt='o', label = 'Observed') # plotting given data with error
ax2.plot(mass_arr, f(mass_arr), '-', label = 'Computed') # plotting computed solution
ax2.set_xlabel('Mass (solar mass units)')
ax2.set_ylabel('Radius (solar radius)')
ax2.set_title('White Dwarf: Radius vs Mass, Observed and Computed')
plt.legend()
plt.grid(True)
plt.savefig('TsouSonya_Project3_Fig2_Obsvplot.png') # save to file

plt.show()

# Plot matches terribly with observational data, but that does appear to be normal for astronomy
