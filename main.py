import numpy as np
import matplotlib.pyplot as plt


count = int(input("Number of comparisons : "))
learning_rates = []
iterations_list = []

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2.1, 4.2, 5.8, 8.3, 10.1, 12.2, 13.9, 16.4, 18.1])


for _ in range(count):
    learning_rates.append(float(input("learning_rate : ")))
    iterations_list.append(int(input("iteration : ")))


def train_model(x, y, learning_rate, iterations):
    w = 0
    b = 0

    history_w = []
    history_b = []
    history_mse = []
    history_iteration = []

    for iteration in range(1, iterations + 1):

        y_pred = w * x + b
        error = y_pred - y

        gradient_w = (2 / len(x)) * np.sum(x * error)
        gradient_b = (2 / len(x)) * np.sum(error)

        w -= learning_rate * gradient_w
        b -= learning_rate * gradient_b

        y_pred = w * x + b
        error = y_pred - y

        mse = np.mean(error ** 2)

        history_iteration.append(iteration)
        history_w.append(w)
        history_b.append(b)
        history_mse.append(mse)

    return {
        "learning_rate": learning_rate,
        "iterations": iterations,
        "history_iteration": history_iteration,
        "history_w": history_w,
        "history_b": history_b,
        "history_mse": history_mse,
        "final_w": w,
        "final_b": b
    }


experiments = []

for i in range(count):
    result = train_model(
        x,
        y,
        learning_rates[i],
        iterations_list[i]
    )

    experiments.append(result)

    print(f"Experiment {i + 1} completed!")

for i, experiment in enumerate(experiments, start=1):
    print(f"Experiment {i}: w={experiment['final_w']:.4f}, b={experiment['final_b']:.4f}")


figure, axes = plt.subplots(2, 2)

for i, experiment in enumerate(experiments, start=1):
    axes[0, 0].plot(
        experiment["history_iteration"],
        experiment["history_w"],
        label=f"Experiment {i}"
    )

for i, experiment in enumerate(experiments, start=1):
    axes[0, 1].plot(
        experiment["history_iteration"],
        experiment["history_b"],
        label=f"Experiment {i}"
    )

for i, experiment in enumerate(experiments, start=1):
    axes[1, 0].plot(
        experiment["history_iteration"],
        experiment["history_mse"],
        label=f"Experiment {i}"
    )

for i, experiment in enumerate(experiments, start=1):
    axes[1, 1].plot(
        x,
        experiment["final_w"] * x + experiment["final_b"],
        label=f"Experiment {i}"
    )

axes[0, 0].set_xlabel("Iteration")
axes[0, 0].set_ylabel("w")
axes[0, 0].set_title("w")
axes[0, 0].legend()

axes[0, 1].set_xlabel("Iteration")
axes[0, 1].set_ylabel("b")
axes[0, 1].set_title("b")
axes[0, 1].legend()

axes[1, 0].set_xlabel("Iteration")
axes[1, 0].set_ylabel("MSE")
axes[1, 0].set_title("MSE")
axes[1, 0].legend()

axes[1, 1].scatter(x, y, color="black", label="Data")
axes[1, 1].set_xlabel("X")
axes[1, 1].set_ylabel("Y / Prediction")
axes[1, 1].set_title("Regression")
axes[1, 1].legend()

plt.tight_layout()
plt.show()