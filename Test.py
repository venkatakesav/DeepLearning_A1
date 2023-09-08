import numpy as np
import pandas as pd

# Load the Data
data = pd.read_csv('./Concrete_Data.csv', header=None)
# Remove the First Row
data = data.iloc[1:]
# Print the Shape of the Data
print(data.shape)

# Target Variable 
target_variable = data.iloc[:, -1]
# Input Variables
input_variables = data.iloc[:, 0:8]

# Convert both to numpy arrays
target_variable = np.array(target_variable, dtype=np.float32)
input_variables = np.array(input_variables, dtype=np.float32)

print(target_variable.shape)
print(input_variables.shape)

# Preprocess the Data
########################

########################

# Split the Data into Train and Test
# Randomly choose 70% and 30% of the data
Total_samples = target_variable.shape[0]
Train_percentage = 0.7

Train_Samples = 0.7*Total_samples
Test_Samples = 0.3*Total_samples

# Shuffle The Data
indices = np.arange(Total_samples)
np.random.shuffle(indices)

# Split this data
Train_indices = indices[:int(Train_Samples)]
Test_indices = indices[int(Train_Samples):]

X_Train = input_variables[Train_indices]
Y_Train = target_variable[Train_indices]
x_test = input_variables[Test_indices]
y_test = target_variable[Test_indices]


# Define the tanh activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)**2

# Define the Mean Squared Error (MSE) loss function
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Initialize neural network parameters
input_size = 8
hidden_units = 25  # Change this value for different variations
output_size = 1
learning_rate = 1e-3
epochs = 1000  # Change this for the desired number of epochs

# Initialize weights and biases for the neural network
np.random.seed(0)
weights_input_hidden = np.random.randn(input_size, hidden_units)
bias_hidden = np.zeros((1, hidden_units))
weights_hidden_output = np.random.randn(hidden_units, output_size)
bias_output = np.zeros((1, output_size))

# Load your training and testing data using Pandas
# Assume you have 'X_Train', 'Y_Train', 'X_test', and 'y_test' DataFrames

print("X_Train", X_Train.shape)

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_input = np.dot(X_Train, weights_input_hidden) + bias_hidden
    hidden_output = tanh(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
    final_output = final_input
    
    # Compute the loss
    loss = mse_loss(Y_Train, final_output)

    # Reshape the loss value into a 721*1 matrix
    Y_Train = Y_Train.reshape(-1,1)
    # Backpropagation
    d_loss = 2 * (final_output - Y_Train) / len(X_Train)
    # The Diff wrto The Final Output
    d_final_input = d_loss
    print("d_final_input", d_final_input.shape)
    # Differntiation wrto The Weights of Layer 2 = (H_Layer * d_final_input) -> 25*721 * 721*1 = 25*1
    d_weights_hidden_output = np.dot(hidden_output.T, d_final_input)
    # Differntiation wrto The Bias of Layer 2 = (d_final_input)
    d_bias_output = np.sum(d_final_input, axis=0, keepdims=True)
    # Differntiation wrto The Hidden Layer = (d_final_input * weights_hidden_output) (Because of Hidden Output : No.of Samples*No.of Hidden Units)
    d_hidden_output = np.dot(weights_hidden_output, d_final_input.T) # 25*721
    d_hidden_input = d_hidden_output.T * tanh_derivative(hidden_input)
    d_weights_input_hidden = np.dot(X_Train.T, d_hidden_input) # 8*721 * 721*25 = 8*25
    d_bias_hidden = np.sum(d_hidden_input, axis=0, keepdims=True)

    # Update weights and biases using gradient descent
    weights_input_hidden -= learning_rate * d_weights_input_hidden
    bias_hidden -= learning_rate * d_bias_hidden
    weights_hidden_output -= learning_rate * d_weights_hidden_output
    bias_output -= learning_rate * d_bias_output
    
    # Print loss for every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Testing
hidden_input = np.dot(x_test, weights_input_hidden) + bias_hidden
hidden_output = tanh(hidden_input)
final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = final_input

test_loss = mse_loss(y_test, final_output)
print(f"Test Loss: {test_loss:.4f}")
