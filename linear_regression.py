import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("datasets/StudentPerformance.csv")
data.drop(
    [
        "Previous Scores",
        "Extracurricular Activities",
        "Sleep Hours",
        "Sample Question Papers Practiced",
    ],
    axis=1,
    inplace=True,
)


def gradient_descent(w_cur, b_cur, points, lr):
    w_grad = 0
    b_grad = 0
    N = len(points)

    for i in range(N):
        x = points.loc[i, "Hours Studied"]
        y = points.loc[i, "Performance Index"]

        w_grad += -(2 / N) * x * (y - (w_cur * x + b_cur))
        b_grad += -(2 / N) * (y - (w_cur * x + b_cur))

    w_new = w_cur - lr * w_grad
    b_new = b_cur - lr * b_grad

    return w_new, b_new


w = 0
b = 0
L = 0.01
epochs = 1000

for i in range(epochs):
    w, b = gradient_descent(w, b, data, L)

print(w, b)


# Now the real solution with sklearn
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(data[["Hours Studied"]], data["Performance Index"])

print(model.coef_, model.intercept_)

plt.scatter(data["Hours Studied"], data["Performance Index"])
plt.plot(
    data["Hours Studied"],
    model.coef_[0] * data["Hours Studied"] + model.intercept_,
    color="black",
)
plt.plot(data["Hours Studied"], w * data["Hours Studied"] + b, color="red")
plt.savefig("plot.png")
