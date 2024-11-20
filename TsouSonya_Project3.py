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
    S (array-like): state vector [p, m, dp/dr, dm/dr]
   
    Returns:

    dSstate (array): state vector [dp/dr, dm/dr, d2p/dr2, d2m/dr2] 
    '''
    dSstate = [0,0,0,0] # ie [dp/dr, dm/dr, d2p/dr2, d2m/dr2]
    p = S[0]
    m = S[1]

    x = p**(1/3)
    gamma = (x**2)/(3*((1+x**2)**0.5))
 
    #derivatives 
    dSstate[0] = S[2] 
    dSstate[1] = S[3]
    dSstate[2] = -(m*p)/(gamma*r**2)
    dSstate[3] = (r**2)*p
    
    return dSstate

# This is the function that returns the thing we want to 'zero' - events in solve ivp makes this equal to 0
def maxR(r, S):
        return dSstate_dr(r,S)[0]

pc = np.linspace(0.1, 2.5*10**6, 10, endpoint=True)

fig, ax = plt.subplots() 

for i in range(len(pc)):
    sol = sc.integrate.solve_ivp(dSstate_dr, (0.01, 30000), np.array([pc[0],0,0,0]), t_eval=np.linspace(0.01, 30000, 1000), events = maxR)
    ax.plot(sol.t, sol.y[0], label = f'pc={pc[i]}')

  # "y is an array of arrays, so we use y[0]"
# ax.set_title('Solution of the undamped, undriven ODE')  # title and axis labels
# ax.set_xlabel('Time (s)') 
# ax.set_ylabel('theta(t)')  
ax.legend()
plt.grid(True)  # adding grid
# plt.savefig('TsouSonya_Lab5_Fig1.png') # save to file
plt.show()
