import pandas
import schedule
import matplotlib.pyplot as mpl
import keyboard
import numpy

def prep_data_for_graph():
  pass

def add_row_to_table(neruonsone, neuronstwo, denseamt, dense1, dense2, dense3, dense4, dense5, epochsize, batchsize, profit, profitprcntg, startingmoney, timetookrun, timeran, stocksymbol, interval)
  with open("data/data.md", "a") as data:
    data.append(f"| {neuronsone} | {neuoronstwo} | {denseamt} | {dense1} | {dense2} | {dense3} | {dense4} | {dense5} | {epochsize} | {batchsize} | {profit} | {profitprcntg} | {startingmoney} | {timetookrun} | {timeran} | {stocksymbol} | {interval} |")

class gragh():
  def __init__(self):
    pass

  def scatter_plot(x, y):
    mpl.scatter(x, y)

  def line_plot(x, y):
    mpl.plot(x, y)

  def bar_graph():
    pass

  def ddd_graph():
    pass

  def render_graph():
    mpl.show()
