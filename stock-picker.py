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

    loss = -1/training_rotations ( 
    
