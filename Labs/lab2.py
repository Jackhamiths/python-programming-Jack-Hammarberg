import pandas as pd
import matplotlib.pyplot as plt

# https://www.w3schools.com/python/pandas/pandas_csv.asp 
trainingSet = pd.read_csv(r"C:\Users\Jack\python-programming-Jack-Hammarberg\Labs\Lab2-data-and-testpoints\datapoints.csv")
testSet = pd.read_csv(r"C:\Users\Jack\python-programming-Jack-Hammarberg\Labs\Lab2-data-and-testpoints\testpoints.csv")

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

