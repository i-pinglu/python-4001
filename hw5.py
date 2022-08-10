import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, 0].values.reshape(-1,1)
Y = dataset.iloc[:, 1].values

train_x, test_x, train_y, test_y  = train_test_split(X, Y, test_size=0.2, random_state=10)
regression = LinearRegression()
regression.fit(train_x, train_y)
y_p = regression.predict(test_x)

print(f"evaluation MSE: {mean_squared_error(test_y, y_p)}")

plt.scatter(X[:, 0], Y, s = 3)
plt.plot(test_x,y_p, color = 'red')
plt.show()