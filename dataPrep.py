import pandas as pd
import schedule
import seaborn as sns
import keyboard
import numpy

def prep_data_for_graph(AP_TYPE, nameOfAI, neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, timetookrunHours, timetookrunMins, timetookrunSecs, timeranHours, timeranMins, timeranSecs, stocksymbol, interval):
  if AP_TYPE == "dt":
      return nameOfAI, neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, (f"{timetookrunHours};{timetookrunMins}:{timetookrunSecs}"), (f"{timeranHours};{timeranMins}:{timeranSec}"), stocksymbol, interval
  elif AP_TYPE == "csv":
    return nameOfAI, neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, timetookrunHours, timetookrunMins, timetookrunSecs, timeranHours, timeranMins, timeranSecs, stocksymbol, interval

def add_row_to_table(nameOfAI, neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, timetookrun, timeran, stocksymbol, interval)
  with open("data/table.md", "a") as data:
    data.append(f"| {nameOfAI} | {neuronsone} | {neuoronstwo} | {denseamt} | {dense1} | {dense2} | {dense3} | {dense4} | {dense5} | {epochsize} | {batchsize} | {profit} | {profitprcntg} | {startingmoney} | {timetookrun} | {timeran} | {stocksymbol} | {interval} |")

def add_data_to_csv(nameOfAI, neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, timetookrun, timeran, stocksymbol, interval)
  with open("data/data.csv", "a") as data:
    data.append(f"{nameOfAI}, {neuronsone}, {neuoronstwo}, {denseamt}, {dense1}, {dense2}, {dense3}, {dense4}, {dense5}, {epochsize}, {batchsize}, {profit}, {profitprcntg}, {startingmoney}, {timetookrun}, {timeran}, {stocksymbol}, {interval}")


class graph():
  def __init__(self):
    pass

  def scatter_plot(data_file):
    sns.scatterplot(data_file)

  def line_plot(x, y):
    pass

  def bar_graph():
    pass

  def ddd_graph():
    pass

  def render_graph():
    pass
