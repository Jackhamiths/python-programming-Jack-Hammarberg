import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# https://www.w3schools.com/python/pandas/pandas_csv.asp 
# https://www.geeksforgeeks.org/python/get-current-directory-python/
currDir = os.path.abspath(os.path.dirname(__file__))
trainingSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/datapoints.csv")
testSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/testpoints.csv")

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
df = pd.DataFrame(trainingSet,
    columns=['width', 'height', 'pokemons'])

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
# pichus are blue and pikachus are red
df.plot.scatter(
    x='width',
    y='height',
    c=df['pokemons'].map({0: 'blue', 1: 'red'})
)


plt.show()

# https://www.geeksforgeeks.org/python/calculate-the-euclidean-distance-using-numpy/ 
d = np.sqrt(np.sum((trainingSet - testSet)**2))
print(d)

