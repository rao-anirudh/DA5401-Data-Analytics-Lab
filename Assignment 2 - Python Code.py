# Importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# Defining SSE
def sse(y1, y2):
    return np.sum((y1 - y2) ** 2)


# Loading the data into a Pandas DataFrame
data = open("Assignment2.data", mode="r").readlines()
df = pd.DataFrame([x.split() for x in data][:-1])
df.columns = df.iloc[0]
df = df[1:]
df.index = [x - 1 for x in df.index]
df["SpringPos"] = df["SpringPos"].astype(float)
df["StockPrice"] = df["StockPrice"].astype(float)

# TASK 1

print("\nTASK 1\n")

X = np.array(df.index).reshape(len(df), 1)
y = np.array(df["StockPrice"])

# 1.1 - OLS solution

m = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
print(f"Slope from OLS solution: {m[0]}")
y_pred = m * X
y_pred = y_pred.reshape(1, len(df))
print(f"SSE of OLS solution: {sse(y, y_pred):.3f}\n")

plt.figure(dpi=150)
plt.scatter(df.index, df["StockPrice"], label="True stock price")
plt.scatter(df.index, y_pred, label="y = mx (OLS)")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()

# 1.2 - tan(θ) linear search

thetas = list(range(0, 61, 5))
slopes = [np.tan(x * np.pi / 180) for x in thetas]
errors = []
for slope in slopes:
    y_pred = slope * X
    y_pred = y_pred.reshape(1, len(df))
    errors.append(sse(y, y_pred))

plt.figure(dpi=150)
plt.plot(thetas, errors)
plt.xlabel("θ")
plt.ylabel("SSE")
plt.show()

print(f"The SSE is minimised at θ = {thetas[errors.index(min(errors))]}°, with an SSE of {min(errors):.3f}\n")

plt.figure(dpi=150)
plt.scatter(df.index, df["StockPrice"], label="True stock price")
plt.scatter(df.index, X * slopes[errors.index(min(errors))], label=f"y = tan({thetas[errors.index(min(errors))]}°)x")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()

# 1.3 - sklearn

from sklearn.linear_model import LinearRegression

linear_model = LinearRegression(fit_intercept=False)
linear_model.fit(X, y)

plt.figure(dpi=150)
plt.scatter(df.index, df["StockPrice"], label="True stock price")
plt.scatter(df.index, linear_model.predict(X), label="Predicted stock price (sklearn)")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()

print(f"SSE of sklearn solution: {sse(y, linear_model.predict(X)):.3f}\n")

# 1.4 - Comparison

m1 = m[0]
m2 = slopes[errors.index(min(errors))]
m3 = linear_model.coef_[0]
print(f"m_OLS = {m1:.3f}\nm_tan = {m2:.3f}\nm_sklearn = {m3:.3f}\n")

# TASK 2

print("TASK 2\n")

X = np.array(df.index)
y = np.array(df["StockPrice"])

# 2.1 - Train-test-eval split

random.seed(5401)

# Interpolation

X_inter_train = random.sample(list(X), int(len(df) * 0.7))
X_inter_eval = random.sample(list(set(X) - set(X_inter_train)), int(len(df) * 0.15))
X_inter_test = random.sample(list(set(X) - set(X_inter_train) - set(X_inter_eval)), int(len(df) * 0.15))

y_inter_train = df["StockPrice"].loc[X_inter_train]
y_inter_eval = df["StockPrice"].loc[X_inter_eval]
y_inter_test = df["StockPrice"].loc[X_inter_test]

# Extrapolation

X_extra_train = list(X)[:int(len(df) * 0.7)]
X_extra_eval = random.sample(list(set(X) - set(X_extra_train)), int(len(df) * 0.15))
X_extra_test = random.sample(list(set(X) - set(X_extra_train) - set(X_extra_eval)), int(len(df) * 0.15))

y_extra_train = df["StockPrice"].loc[X_extra_train]
y_extra_eval = df["StockPrice"].loc[X_extra_eval]
y_extra_test = df["StockPrice"].loc[X_extra_test]

X_inter_test_copy = X_inter_test.copy()
X_extra_test_copy = X_extra_test.copy()


# 2.2 - Feature transformation


def transformation(x):
    return 0.119 * x + 5 * np.sin(0.1 * x)


plt.figure(dpi=150)
plt.scatter(df.index, y, label="True stock price")
plt.scatter(df.index, [transformation(t) for t in X], label="Transformed feature")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()

X_inter_train = np.array([transformation(t) for t in X_inter_train]).reshape(-1, 1)
X_inter_eval = np.array([transformation(t) for t in X_inter_eval]).reshape(-1, 1)
X_inter_test = np.array([transformation(t) for t in X_inter_test]).reshape(-1, 1)

X_extra_train = np.array([transformation(t) for t in X_extra_train]).reshape(-1, 1)
X_extra_eval = np.array([transformation(t) for t in X_extra_eval]).reshape(-1, 1)
X_extra_test = np.array([transformation(t) for t in X_extra_test]).reshape(-1, 1)

# 2.3 - Interpolation

interpolation_model = LinearRegression(fit_intercept=False)
interpolation_model.fit(X_inter_train, y_inter_train)
y_inter_eval_pred = interpolation_model.predict(X_inter_eval)
y_inter_test_pred = interpolation_model.predict(X_inter_test)
print(f"Interpolation eval error: {sse(y_inter_eval, y_inter_eval_pred):.3f}")
print(f"Interpolation test error: {sse(y_inter_test, y_inter_test_pred):.3f}\n")

plt.figure(dpi=150)
plt.scatter(X_inter_test_copy, y_inter_test, label="True stock price (test data)")
plt.scatter(X_inter_test_copy, y_inter_test_pred, label="Predicted stock price")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.title("Interpolation")
plt.legend()
plt.show()

# 2.4 - Extrapolation

extrapolation_model = LinearRegression(fit_intercept=False)
extrapolation_model.fit(X_extra_train, y_extra_train)
y_extra_eval_pred = extrapolation_model.predict(X_extra_eval)
y_extra_test_pred = extrapolation_model.predict(X_extra_test)
print(f"Extrapolation eval error: {sse(y_extra_eval, y_extra_eval_pred):.3f}")
print(f"Extrapolation test error: {sse(y_extra_test, y_extra_test_pred):.3f}\n")

plt.figure(dpi=150)
plt.scatter(X_extra_test_copy, y_extra_test, label="True stock price (test data)")
plt.scatter(X_extra_test_copy, y_extra_test_pred, label="Predicted stock price")
plt.title("Extrapolation")
plt.xlabel("Time")
plt.ylabel("Stock price")
plt.legend()
plt.show()

# TASK 3

print("TASK 3\n")

X = np.array(df.index)
y = np.array(df["SpringPos"])

# 3.1 - Train-test-eval split

random.seed(5401)

# Interpolation

X_inter_train = random.sample(list(X), int(len(df) * 0.7))
X_inter_eval = random.sample(list(set(X) - set(X_inter_train)), int(len(df) * 0.15))
X_inter_test = random.sample(list(set(X) - set(X_inter_train) - set(X_inter_eval)), int(len(df) * 0.15))

y_inter_train = df["SpringPos"].loc[X_inter_train]
y_inter_eval = df["SpringPos"].loc[X_inter_eval]
y_inter_test = df["SpringPos"].loc[X_inter_test]

# Extrapolation

X_extra_train = list(X)[:int(len(df) * 0.7)]
X_extra_eval = random.sample(list(set(X) - set(X_extra_train)), int(len(df) * 0.15))
X_extra_test = random.sample(list(set(X) - set(X_extra_train) - set(X_extra_eval)), int(len(df) * 0.15))

y_extra_train = df["SpringPos"].loc[X_extra_train]
y_extra_eval = df["SpringPos"].loc[X_extra_eval]
y_extra_test = df["SpringPos"].loc[X_extra_test]

X_inter_test_copy = X_inter_test.copy()
X_extra_test_copy = X_extra_test.copy()


# 3.2 - Feature transformation


def transformation(x):
    return max(df["SpringPos"]) * np.exp(-0.005 * x) * np.sin(0.1 * x)


plt.figure(dpi=150)
plt.scatter(df.index, y, label="True spring position")
plt.scatter(df.index, [transformation(t) for t in X], label="Transformed feature")
plt.xlabel("Time")
plt.ylabel("Spring position")
plt.legend()
plt.show()

X_inter_train = np.array([transformation(t) for t in X_inter_train]).reshape(-1, 1)
X_inter_eval = np.array([transformation(t) for t in X_inter_eval]).reshape(-1, 1)
X_inter_test = np.array([transformation(t) for t in X_inter_test]).reshape(-1, 1)

X_extra_train = np.array([transformation(t) for t in X_extra_train]).reshape(-1, 1)
X_extra_eval = np.array([transformation(t) for t in X_extra_eval]).reshape(-1, 1)
X_extra_test = np.array([transformation(t) for t in X_extra_test]).reshape(-1, 1)

# 3.3 - Interpolation

interpolation_model = LinearRegression(fit_intercept=False)
interpolation_model.fit(X_inter_train, y_inter_train)
y_inter_eval_pred = interpolation_model.predict(X_inter_eval)
y_inter_test_pred = interpolation_model.predict(X_inter_test)
print(f"Interpolation eval error: {sse(y_inter_eval, y_inter_eval_pred):.3f}")
print(f"Interpolation test error: {sse(y_inter_test, y_inter_test_pred):.3f}\n")

plt.figure(dpi=150)
plt.scatter(X_inter_test_copy, y_inter_test, label="True spring position (test data)")
plt.scatter(X_inter_test_copy, y_inter_test_pred, label="Predicted spring position")
plt.xlabel("Time")
plt.ylabel("Spring position")
plt.title("Interpolation")
plt.legend()
plt.show()

# 3.4 - Extrapolation

extrapolation_model = LinearRegression(fit_intercept=False)
extrapolation_model.fit(X_extra_train, y_extra_train)
y_extra_eval_pred = extrapolation_model.predict(X_extra_eval)
y_extra_test_pred = extrapolation_model.predict(X_extra_test)
print(f"Extrapolation eval error: {sse(y_extra_eval, y_extra_eval_pred):.3f}")
print(f"Extrapolation test error: {sse(y_extra_test, y_extra_test_pred):.3f}")

plt.figure(dpi=150)
plt.scatter(X_extra_test_copy, y_extra_test, label="True spring position (test data)")
plt.scatter(X_extra_test_copy, y_extra_test_pred, label="Predicted spring position")
plt.title("Extrapolation")
plt.xlabel("Time")
plt.ylabel("Spring position")
plt.legend()
plt.show()
