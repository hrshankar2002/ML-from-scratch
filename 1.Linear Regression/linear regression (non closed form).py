'''
Linear regression using non closed form (gradient descent)
'''

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("../assets/student_scores.csv")
 
# loss function

def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Hours
        y= points.iloc[i].Scores
        total_error += (y-(m*x + b)) ** 2
        
    return total_error / float(len(points))

# gradient descent function

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    
    n = len(points)
    
    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
        
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    
    return m, b
    
# linear regression implementation

m = 0 
b = 0
L = 0.0001 # learning rate
epochs = 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch:{i}")
    m, b = gradient_descent(m, b, data, L)

plt.scatter(data.Hours, data.Scores, color = "black")
plt.plot(list(range(2,10)), [m * x + b for x in range(2, 10)], color = "red")
plt.show()