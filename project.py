import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Génération de données (simulation ventes)
np.random.seed(42)
days = np.arange(1, 101)

sales = 50 + 0.5 * days + np.random.normal(0, 5, size=100)

data = pd.DataFrame({
    'day': days,
    'sales': sales
})

# Features et target
X = data[['day']]
y = data['sales']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
predictions = model.predict(X_test)

# Affichage
plt.scatter(X_test, y_test, label="Real")
plt.scatter(X_test, predictions, label="Predicted")
plt.legend()
plt.title("Sales Prediction")
plt.show()

# Score
score = model.score(X_test, y_test)
print("Model accuracy:", score)
