# # Model Representation
# - Learn to implement the model $f_{w,b}$ for linear regression with one variable

# ## Tools
# - NumPy, a popular library for scientific computing
# - Matplotlib, a popular library for plotting data

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

# x_train is the input variable (size in 1000 square feet)
# y_train is the target (price in 1000s of dollars)
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")

# m is the number of training examples
print(f"x_train.shape: {x_train.shape}")
m = x_train.shape[0]
print(f"Number of training examples is: {m}")

# Can also use the Python `len()` function

# m is the number of training examples
m = len(x_train)
print(f"Number of training examples is: {m}")



i = 0 # Change this to 1 to see (x^1, y^1)

x_i = x_train[i]
y_i = y_train[i]
print(f"(x^({i}), y^({i})) = ({x_i}, {y_i})")

# ### Plotting the data

# Plot these two points using the `scatter()` function in the `matplotlib` library. 
# The function arguments `marker` and `c` show the points as red crosses (the default is blue dots).

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r')
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()


# **Note: The model's w and b parameters can be adjusted manually to test the output**

w = 100
b = 100
print(f"w: {w}")
print(f"b: {b}")

# fw,b(x) = wx + b
# for $x^{(0)}$, `f_wb = w * x[0] + b` 
# for $x^{(1)}$, `f_wb = w * x[1] + b`

def compute_model_output(x, w, b):
    """
    Computes the prediction of a linear model
    Args:
      x (ndarray (m,)): Data, m examples 
      w,b (scalar)    : model parameters  
    Returns
      f_wb (ndarray (m,)): model prediction
    """
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
        
    return f_wb

# Vall the `compute_model_output` function and plot the output.

tmp_f_wb = compute_model_output(x_train, w, b,)

# Plot model prediction
plt.plot(x_train, tmp_f_wb, c='b',label='Our Prediction')

# Plot the data points
plt.scatter(x_train, y_train, marker='x', c='r',label='Actual Values')

# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.legend()
plt.show()



# ### Prediction
# Now that we have a model, we can use it to make our original prediction. Let's predict the price of a house with 1200 sqft. Since the units of $x$ are in 1000's of sqft, $x$ is 1.2.
# 

w = 200                         
b = 100    
x_i = 1.2
cost_1200sqft = w * x_i + b    

print(f"${cost_1200sqft:.0f} thousand dollars")







