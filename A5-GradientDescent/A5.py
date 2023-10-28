# Set up the imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define Problem 1: f(x) = x^2 + 3x + 8
def f1(x):
    return x ** 2 + 3 * x + 8

def df1_dx(x):
    return 2 * x + 3

# Define Problem 3: f(x, y) = x^4 - 16x^3 + 96x^2 - 256x + y^2 - 4y + 262
def f3(x, y):
    return x**4 - 16*x**3 + 96*x**2 - 256*x + y**2 - 4*y + 262

def df3_dx(x, y):
    return 4*x**3 - 48*x**2 + 192*x - 256

def df3_dy(x, y):
    return 2*y - 4

# Define Problem 4: f(x, y) = exp(-(x - y)^2) * sin(y)
def f4(x, y):
    return np.exp(-(x - y)**2) * np.sin(y)

def df4_dx(x, y):
    return -2 * np.exp(-(x - y)**2) * np.sin(y) * (x - y)

def df4_dy(x, y):
    return np.exp(-(x - y)**2) * np.cos(y) + 2 * np.exp(-(x - y)**2) * np.sin(y)*(x - y)

# Define Problem 5: f(x) = cos(x)^4 - sin(x)^3 - 4sin(x)^2 + cos(x) + 1
def f5(x):
    return np.cos(x)**4 - np.sin(x)**3 - 4*np.sin(x)**2 + np.cos(x) + 1

def df5_dx(x):
    return (-4)*((np.cos(x))**3)*(np.sin(x)) - (3 * ((np.sin(x))**2) * np.cos(x)) - (8*np.sin(x)*np.cos(x)) - np.sin(x)

# Gradient descent function for 1D problems
def grad_desc(f, df, start, end, step, n, guess):
    # Ensure the guess is within the search range
    guess = max(start, min(end, guess))
    x = guess
    for _ in range(n):
        grad = df(x)
        x = x - (step * grad)
        # Ensure that x remains in the search range
        x = max(start, min(end, x))
        
    return x

# Gradient descent function for 2D problems
def grad_desc_2d(f, dfx, dfy, start_x, end_x, start_y, end_y, guess_x, guess_y, step, n):
    x = guess_x
    y = guess_y
    
    # Create a 3D plot to visualize the optimization path
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    x_range = np.linspace(start_x, end_x)
    y_range = np.linspace(start_y, end_y)
    
    X, Y = np.meshgrid(x_range, y_range)
    Z = f(X, Y)
    
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.4)
    
    for _ in range(n):
        grad_x = dfx(x, y)
        grad_y = dfy(x, y)
        x = x - (step * grad_x)
        y = y - (step * grad_y)
        # Ensure the updated values of x and y remain within the search range
        x = max(start_x, min(x, end_x))
        y = max(start_y, min(y, end_y))
        ax.scatter(x, y, f(x, y), c='red')
        
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.title('Gradient Descent Path')
    plt.show()
        
    return x, y, f(x, y)

# ----- Problem 1: Optimize f(x) = x^2 + 3x + 8 -------

start_range = -5
end_range = 5
learning_rate = 0.1
iterations = 100
initial_guess = 4

# Perform gradient descent for Problem 1
op_x = grad_desc(f1, df1_dx, start_range, end_range, learning_rate, iterations, initial_guess)

# Print the optimal solution for Problem 1
print('Optimal x:', op_x)
print('Optimized value of f1(x) is', f1(op_x))

# Create the axis and function for Problem 1
xbase = np.linspace(-5, 5, 500)
ybase = f1(xbase)

# Create a figure and axis for the animation of Problem 1
bestcost = 100000
bestx = 4
rangemin, rangemax = -5, 5 
fig, ax = plt.subplots()
ax.plot(xbase, ybase)
xall, yall = [], []
lnall,  = ax.plot([], [], 'ro')
lngood, = ax.plot([], [], 'go', markersize=10)
lr = 0.1

# Function to animate gradient descent for Problem 1
def onestepderiv(frame):
    global bestcost, bestx, lr
    x = bestx - df1_dx(bestx) * lr 
    bestx = x
    y = f1(x)
    lngood.set_data(x, y)
    xall.append(x)
    yall.append(y)
    lnall.set_data(xall, yall)

ani = FuncAnimation(fig, onestepderiv, frames=range(100), interval=100, repeat=False)
ani.save("problem1_animation.gif", writer="pillow")
plt.show()

# ----- Problem 2: Optimize f3(x, y) -------

# Define the search range and initial parameters for Problem 2
xlim3 =  [-10, 10]
ylim3 =  [-10, 10]
initial_guess_x3 = 5
initial_guess_y3 = 5
learning_rate3 = 0.1
num_iterations3 = 1000

# Perform gradient descent for Problem 2
optimal_x3, optimal_y3, opz = grad_desc_2d(f3, df3_dx, df3_dy, xlim3[0], xlim3[1], ylim3[0], ylim3[1], initial_guess_x3, initial_guess_y3, learning_rate3, num_iterations3)

# Save the figure for Problem 2
plt.savefig("problem2_figure.png")

# Print the optimal solution for Problem 2
print("Optimal x:", optimal_x3, "Optimal y:", optimal_y3)
print("Optimized value of f3(x, y) is", f3(optimal_x3, optimal_y3))

# ----- Problem 3: Optimize f4(x, y) -------

# Define the search range and initial parameters for Problem 3
xlim4 = [-np.pi, np.pi]
ylim4 = [-np.pi, np.pi]
initial_x4 = 0
initial_y4 = 0
learning_rate4 = 0.1
num_iterations4 = 100

# Perform gradient descent for Problem 3
optimal_x4, optimal_y4, opz = grad_desc_2d(f4, df4_dx, df4_dy, xlim4[0], xlim4[1], ylim4[0], ylim4[1], initial_x4, initial_y4, learning_rate4, num_iterations4)

# Save the figure for Problem 3
plt.savefig("problem3_figure.png")

# Print the optimal solution for Problem 3
print("Optimal x:", optimal_x4, "Optimal y:", optimal_y4)
print("Optimized value of f4(x, y) is", f4(optimal_x4, optimal_y4))

# ----- Problem 4: Optimize f5(x) -------

# Define the search range and initial parameters for Problem 4
range_x5 = [0, 2 * np.pi]
initial_x5 = 3
learning_rate5 = 0.05
num_iterations5 = 1000

# Perform gradient descent for Problem 4
optimal_x5 = grad_desc(f5, df5_dx, range_x5[0], range_x5[1], learning_rate5, num_iterations5, initial_x5)

# Print the optimal solution for Problem 4
print("Optimal x:", optimal_x5)
print("Optimized value of f5(x) is", f5(optimal_x5))

# Create the axis and function for Problem 4
xbase = np.linspace(0, 2*np.pi, 200)
ybase = f5(xbase)

# Create a figure and axis for the animation of Problem 4
bestcost = 100000
bestx = 3
rangemin, rangemax = 0, 2*np.pi 
fig, ax = plt.subplots()
ax.plot(xbase, ybase)
xall, yall = [], []
lnall,  = ax.plot([], [], 'ro')
lngood, = ax.plot([], [], 'go', markersize=10)
lr = 0.05

# Function to animate gradient descent for Problem 4
def onestepderiv(frame):
    global bestcost, bestx, lr
    x = bestx - df5_dx(bestx) * lr 
    bestx = x
    y = f5(x)
    lngood.set_data(x, y)
    xall.append(x)
    yall.append(y)
    lnall.set_data(xall, yall)

ani = FuncAnimation(fig, onestepderiv, frames=range(1000), interval=500, repeat=False)
ani.save("problem4_animation.gif", writer="pillow")
plt.show()
