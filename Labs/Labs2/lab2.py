import pandas as pd
import matplotlib.pyplot as plt
import os

# https://www.w3schools.com/python/pandas/pandas_csv.asp 
# https://www.geeksforgeeks.org/python/get-current-directory-python/
currDir = os.path.abspath(os.path.dirname(__file__))
trainingSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/datapoints.csv")
testSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/testpoints.csv")

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
df = pd.DataFrame(trainingSet,
    columns=['width', 'height', 'label'])

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
# pichus are blue and pikachus are red
df.plot.scatter(
    x='width',
    y='height',
    c=df['label'].map({0: 'blue', 1: 'red'})
)


plt.show()

