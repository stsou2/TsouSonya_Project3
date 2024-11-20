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

    #x = p**(1/3)
    x = np.sign(p) * (np.abs(p)) ** (1/3)
    gamma = (x**2)/(3*((1+x**2)**0.5))
 
    #derivatives 
    dSstate[0] = -(m*p)/(gamma*r**2)
    dSstate[1] = (r**2)*p
    
    return dSstate

# This is the function that returns the thing we want to 'zero' - events in solve ivp makes this equal to 0
def maxR(r, S):
        if S[0] < 1e-5:
            return 0
        else:
             return S[0]

pc = np.linspace(0.1, 2.5*10**6, 10)
ue = 2
R0 = (7.72*10**8/ue)
M0 = 5.67*10**33/(ue**2)
p0 = 9.74*10**5*ue*const.g0

fig, ax = plt.subplots() 

for i in range(len(pc)):
    sol = sc.integrate.solve_ivp(dSstate_dr, (0.01, 10), np.array([pc[i],0]), events = maxR)
    ax.plot(sol.t_events[0][0]*R0/6.957e10, sol.y_events[0][0][1]*M0/2e33, 'o') #t is r and y[0] is p, y[1] is m
    print(i, 'done')

#ax.plot(sol.t*R0, sol.y[0]*p0, label = f'pc={pc[1]}') #t is r and y[0] is p, y[1] is m

ax.set_title('r vs m')  # title and axis labels
ax.set_xlabel('m') 
ax.set_ylabel('r')  
ax.legend()
plt.grid(True)  # adding grid
# plt.savefig('TsouSonya_Lab5_Fig1.png') # save to file
plt.show()
