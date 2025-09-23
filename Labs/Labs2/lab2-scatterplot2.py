import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# https://www.w3schools.com/python/pandas/pandas_csv.asp 
# https://www.geeksforgeeks.org/python/get-current-directory-python/
currDir = os.path.abspath(os.path.dirname(__file__))
trainingSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/datapoints.csv")
testSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/testpoints.csv")

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
# https://www.geeksforgeeks.org/data-visualization/visualizing-multiple-datasets-on-the-same-scatter-plot/
train_df = pd.DataFrame(trainingSet,
    columns=['width', 'height', 'pokemons'])

test_df = pd.DataFrame(testSet,
    columns=['width', 'height', 'pokemons'])

combined_data = pd.concat([train_df, test_df])


# this code nolonger works as it relies on having the pokemons column with the value 2 which is not optimal 
# this was jsut me testing some stuff for fun xD
sns.scatterplot(data=combined_data,
    x='width',
    y='height',
    c=combined_data['pokemons'].map({0: 'blue', 1: 'red',2: 'green'})
)

plt.title('Scatter Plot of Training and Test set')
plt.legend(title='Pokemons')

# Display the plot
plt.show()

# https://www.geeksforgeeks.org/python/calculate-the-euclidean-distance-using-numpy/ 
d = np.sqrt(np.sum((trainingSet - testSet)**2))
print(d)


