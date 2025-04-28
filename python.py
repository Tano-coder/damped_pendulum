#Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

#CONSTANTS
#Declare the constant variables, these shouldn't be changed
C = 0.48 #Drag coefficient
P = 1.293 #Air pressure
PI = np.pi #Pi = 3.141...
G = 9.81 #Gravitational accelaration
M = 1 #Weight of the pendulum bob
L = 10 #Length of the pendulum
R = 0.05 #Radius of the bob
A = R**2 * PI #Area of the bob facing air resistance (Transverse direction)

#VARIABLES - These should be changed
#Declare the initial values for angle and velocity which can be changed
theta0 = -1
v0 = 2
initial_values = (theta0, v0) #Stores the initial values

#Setup the time variable, in this case the animation/simulation will run for 30 seconds
t = np.linspace(0, 30, 900)
t_span = (0, 30)

#ANIMATING THE SWINGING PENDULUM
#Setup the equation for the pendulum's motion, this will be in terms of angle and angular velocity
#The differential equation is impossible to solve analytically, therefore I am solving it numerically using python
def movement(t, variables):
    theta, omega = variables
    dtheta_dt = omega
    domega_dt = - (C * P * A * L * omega * np.abs(omega) / (2 * M)) - (G * np.sin(theta) / L)

    return (dtheta_dt, domega_dt)

#Solve the equation numerically and store in an array
solution = solve_ivp(movement, t_span, initial_values, t_eval = t)

#Create arrays storing x and y coordinates for the swinging pendulum
x = L * np.sin(solution.y[0])
y = -L * np.cos(solution.y[0])

#Create velocity vector for each instance
v_vector = []
length = len(solution.y[1])
angle = 0
for i in range(length):
    v_vector.append([x[i], y[i], 0.5 * solution.y[1][i] * np.cos(solution.y[0][i]), 0.5 * solution.y[1][i] * np.sin(solution.y[0][i])])

#Setup the animation
fig, ax = plt.subplots()
ax.set_aspect('equal')
ax.set_xlim(-L * 3/2, L * 3/2)
ax.set_ylim(-L * 3/2, L * 3/2)
line, = ax.plot([], [], 'o-', color='black')
bob, = ax.plot([], [], 'o', color='blue')
arrow = ax.quiver(v_vector[0][0], v_vector[0][1], v_vector[0][2], v_vector[0][3], width = 0.01, scale = 4, color = 'blue')
bob.set_markersize(R*200)

#Create function for animation frames
def update(frame):
    x_pos = x[frame]
    y_pos = y[frame]

    line.set_data([0, x_pos], [0, y_pos])
    bob.set_data([x_pos, x_pos], [y_pos, y_pos])
    
    global arrow
    arrow.remove()
    arrow = ax.quiver(v_vector[frame][0], v_vector[frame][1], v_vector[frame][2], v_vector[frame][3], width = 0.01, scale = 4, color = 'blue')

    return line, bob, arrow

#Create the swinging pendulum animation and show
pendulum_Animation = FuncAnimation(fig, update, frames=len(t), interval = 30, blit = False)
plt.title('Swinging Pendulum')
plt.show()

#SIMULATING THE VECTOR FIELD
#Declare variables to be used in creating the vector field
c = 64
theta_lim = 2*PI * 4
omega_lim = c / L
vector_field = [0] * c
initial_theta = -theta_lim
initial_omega = omega_lim

#Creating the vector field by solving trajectories with different starting conditions
for k in range(0, c):
    if (k == c // 2):
        initial_theta = -initial_theta

    
    initial_values = [initial_theta, initial_omega]
    initial_omega -= (2*omega_lim) / c

    instance = solve_ivp(movement, t_span, initial_values, t_eval = t)
    vector_field[k] = instance

#Setup the plot
figfield, axfield = plt.subplots()
axfield.set_xlim(-theta_lim-1, theta_lim+1)
axfield.set_ylim(-omega_lim-1, omega_lim+1)
pendulum_field, = axfield.plot([],[], 'b-', color = 'blue', lw = 1)

#Plot the different vectors
for vector in vector_field:
    for k in range(len(vector.y[0])):
        if k > 0 and k % 25 == 0:
            axfield.quiver(vector.y[0][k], vector.y[1][k], vector.y[0][k] - vector.y[0][k-1], vector.y[1][k] - vector.y[1][k-1]
                           , width=0.001, scale = 4, color='grey')

x_field = solution.y[0]
y_field = solution.y[1]

#Create the function to update frames for the pendulum from the first part moving through the field
def update_pendulum(frame):
    pendulum_field.set_data(x_field[:frame], y_field[:frame])

    return [pendulum_field]

#Animate and show
pendulum_Field_Animation = FuncAnimation(figfield, update_pendulum, frames = len(t), interval = 30, blit = True)
plt.show()







