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

