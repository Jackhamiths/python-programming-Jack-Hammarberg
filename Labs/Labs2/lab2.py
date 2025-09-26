import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# https://www.w3schools.com/python/pandas/pandas_csv.asp 
# https://www.geeksforgeeks.org/python/get-current-directory-python/
currDir = os.path.abspath(os.path.dirname(__file__))
trainingSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/datapoints.csv")
testSet = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/testpoints.csv")

# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
train_df = pd.DataFrame(trainingSet,
    columns=['width', 'height', 'pokemons'])


# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.scatter.html
# pichus are blue represented by 0 and pikachus are red represented by 1
train_df.plot.scatter(
    x='width',
    y='height',
    c=train_df['pokemons'].map({0: 'blue', 1: 'red'})
)


plt.show()


# https://www.geeksforgeeks.org/python/calculate-the-euclidean-distance-using-numpy/ 
d = np.sqrt(np.sum((trainingSet - testSet)**2, axis=0))
print(d)

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
# Prints out the class the testpoints are closest to in the datapoints-
# as 1 for pikachu and 0 for pichu instead of yes for pichu and no for pikachu
pokemon_dict = {1:'Pikachu',0:'Pichu'}
test = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/testpoints.csv")
train = pd.read_csv(f"{currDir}/Lab2-data-and-testpoints/datapoints.csv")
X = train[['width','height']]
y = train['pokemons']
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X, y)
print(neigh.predict(test))

try:
    user_width = float(input("Enter width: "))
    user_height = float(input("Enter height: "))
    
    if user_width < 0 or user_height < 0:
        raise ValueError("Negative numbers are not allowed.")
    
    # https://stackoverflow.com/questions/69326639/sklearn-warning-valid-feature-names-in-version-1-0
    # didn't manage to solve this myself using the fix above so i used chat gpt to get the correct syntax
    feature_names = pd.DataFrame([[user_width, user_height]], columns=["width", "height"]) 

    prediction = neigh.predict(feature_names)[0]
    print(f"The Pokémon is likely a {pokemon_dict[prediction]}")  

except ValueError as e:
    print(f"Stop right there criminal scum you violated the law! {e}")

neigh10 = KNeighborsClassifier(n_neighbors=10)
neigh10.fit(X, y)
predictions = neigh10.predict(test)
print(neigh10.predict(test))

# All of the code below is copypasted from chat gpt 
# with promt Dela in ursprungsdatan slumpmässigt så att: 100 är träningsdata (50 Pikachu, 50 Pichu) 50 är testdata (25 Pikachu, 25 Pichu här e min kod 
# samt att jag copypasta min kod som chat gpt kunde utgå ifrån
# Function for confusion matrix and accuracy
def evaluate(train_X, train_y, test_X, test_y, k=1):
    preds = []
    for i in range(len(test_X)):
        pred = knn_predict(train_X, train_y, test_X.iloc[i].values, k)
        preds.append(pred)

    preds = np.array(preds)
    TP = np.sum((preds == 1) & (test_y == 1))
    TN = np.sum((preds == 0) & (test_y == 0))
    FP = np.sum((preds == 1) & (test_y == 0))
    FN = np.sum((preds == 0) & (test_y == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, (TP, FP, FN, TN)

def knn_predict(train_X, train_y, test_point, k=1):
    # Calculate euclidean distance to all trainingpoints
    distances = np.sqrt(np.sum((train_X - test_point) ** 2, axis=1))
    # Find index on K closest
    k_indices = distances.argsort()[:k]
    # majority vote
    votes = train_y.iloc[k_indices]
    return votes.mode()[0]  # Most frequent class

# repeat 10 times
accuracies = []
for i in range(10):
    # Split trainingset to pikachu and pichu
    pikachu = trainingSet[trainingSet['pokemons'] == 1].sample(frac=1, random_state=i)
    pichu = trainingSet[trainingSet['pokemons'] == 0].sample(frac=1, random_state=i)

    # 50 train + 25 for each class
    pikachu_train, pikachu_test = pikachu.iloc[:50], pikachu.iloc[50:75]
    pichu_train, pichu_test = pichu.iloc[:50], pichu.iloc[50:75]

    # put togheter train and testpoints
    train = pd.concat([pikachu_train, pichu_train])
    test = pd.concat([pikachu_test, pichu_test])

    X_train, y_train = train[['width', 'height']], train['pokemons']
    X_test, y_test = test[['width', 'height']], test['pokemons']

    acc, cm = evaluate(X_train, y_train, X_test, y_test, k=10)
    accuracies.append(acc)

# Plot result
print("Mean accuracy:", np.mean(accuracies))
plt.plot(range(1, 11), accuracies, marker='o')
plt.ylim(0,1)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("KNN  Accuracy över 10 körningar")
plt.show()


