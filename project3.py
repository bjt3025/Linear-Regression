import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import random


# Import the dataset.
training_df = pd.read_csv(filepath_or_buffer="/Users/benthomas/241/project 3/train.csv")


# Function for Prediction
def prediction(features,weights):
    
    pred = []
    
    
    #print("Shape of the array features:", features.shape)
    #rint("Shape of the array weights:", weights.shape)
    #remove price and ID coloumn 
    # Iterate through each row 
    
        # Calculate the sum of values in the row
    pred = np.dot(weights,features.T)
    return pred

# Function for Mean Squared Error (MSE) Loss
def calculate_mse_loss(predictions, actual_prices):
    m = len(actual_prices)
    mse_loss = np.sum((predictions - actual_prices)**2) / m
    return mse_loss

def calculate_gradient(features, predictions, actual_prices): 
    m = len(actual_prices)
    ydif = predictions - actual_prices
    grad = 2 * np.dot(features.T, ydif.T) / m  #does the fact I transpose error instead of features mess up my code 
    return grad

def update_weights(weight, alpha, gradient):
    
    Wt_plus_1 = weight - (alpha * gradient)
    return Wt_plus_1

def plot(mse):
    # Create a line plot
    plt.plot(mse)

    # Add labels and title
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.title('Simple Line Plot')
    # Show the plot
    plt.show()  

    

    
#---------givens-----------
features = training_df.iloc[:, 1:-1]#values.reshape((25, 818)) #shape (25,818)

#creates array of weights
weights = [0]*25#np.random.randn(25).reshape((1, 25)) #shape (1,25)
#np.ones(25).reshape((25, 1)), np.random.randn(25)

alpha = 10e-10

iterations = 500

MSE_values = []
iterations_array = []

#-------code-----------

for i in range(iterations+1):

    #5. Implement function pred that calculates the predicted value of the price based on the current weights and feature values. 
    prediction_array = prediction(features,weights)
    #print("prediction_array", prediction_array)

    #6. calculates MSE: loss function
    mse = calculate_mse_loss(prediction_array, training_df['Price'].values)
    print("mse_loss", mse)
    #print("MSE Loss: ", mse_loss)


    #7. Calculate gradient
    grad = calculate_gradient(features, prediction_array, training_df['Price'].values)
    #print("Gradient: ", grad)

    #8. Update weight
    weights = update_weights(weights, alpha, grad)

    #append MSE_values and iterations to a list to be plotted
    MSE_values.append(mse)
    iterations_array.append(i)

    print("done")

plot(MSE_values)

