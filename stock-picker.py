import numpy
import pandas
import yfinance
import keyboard
import os

class logistic_regresion():
  def __init__(self):
    pass
    
  def sigmoid(z):
    return 1 / 1 + (np.exp(-z))

  def initialize_parameters(n):
    weights = np.random.rand(n, 1) * 0.01
    bias = np.random.rand() * 0.01
    return weights, bias

  def calculate_loss(y_prediction, y_actual):
    training_rotations = len(y_actual)
    loss = -(1/training_rotations) * np.sum(y_actual * np.log(y_prediction) + (1 - y_actual) * log(1 - y_predicted))
    return loss

  def compute_gradients(x, y_actual, y_predicted):
    m = x.shape[0]
    dw = (1/m) * np.dot(x.T, (y_predicted - y_actual))
    db = (1/m) * np.sum(y_predicted - y_actual)
    return dw, db

  def update_parameters(w, b, dw, db, learning_rate):
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

  def train(x, y, num_iterations, learning_rate):
    n_features = X.shape[1]
    w, b = initialize_parameters(n_features)
    
    for i in range(num_iterations):
      # Predict
      y_hat = model(X, w, b)
        
      # Compute loss
      loss = compute_loss(y, y_hat)
        
      # Compute gradients
      dw, db = compute_gradients(X, y, y_hat)
        
      # Update parameters
      w, b = update_parameters(w, b, dw, db, learning_rate)
        
      if i % 100 == 0:
          print(f"Iteration {i}: Loss = {loss}")
    
    return w, b


  def predict(x, w, b):
    y_actual = model(x, w, b)
    y_predicted = [1 if i > 0.5 else 0 for i in y_hat]
    return np.array(y_pred)

