import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from astropy import constants as const
from astropy import units as u


# Part 1


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

pc = np.logspace(-1, 6.4, 10, endpoint=True)
ue = 2
R0 = (7.72*10**8/ue)
M0 = 5.67*10**33/(ue**2)
p0 = 9.74*10**5*ue*const.g0

fig, ax = plt.subplots() 

for i in range(len(pc)):
    maxR.terminal = True # Will terminate once event has been reached. Value of interest will be the singular value in sol.t_events/y_events or the last value in sol.t/y
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
print(f"Estimated mass limit is about {round(np.abs(((mass.value-5.836/(ue**2))/(5.836/(ue**2)))*100), 3)}% less")
plt.show()
