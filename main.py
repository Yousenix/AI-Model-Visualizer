import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([2.1, 4.2, 5.8, 8.3, 10.1, 12.2, 13.9, 16.4, 18.1])

w = 0
b = 0

learning_rate = 0.00001

history_w = []
history_b = []
history_mse = []
history_iteration = []
iteration = 0

plt.ion()

figure, axes = plt.subplots(2, 2)


axes[1, 1].remove()

line1, = axes[0, 0].plot([], [] , color = "blue")
line2, = axes[0, 1].plot([], [] , color = "green")
line3, = axes[1, 0].plot([], [] , color = "red")

axes[0, 0].set_title("w")
axes[0, 1].set_title("b")
axes[1, 0].set_title("MSE")


for i in range(100000):
    iteration += 1
    history_iteration.append(iteration)

    y_pred = w * x + b

    error = y_pred - y

    gradient_w = (2 / len(x)) * np.sum(x * error)
    gradient_b = (2 / len(x)) * np.sum(error)

    w -= learning_rate * gradient_w
    b -= learning_rate * gradient_b

    history_w.append(w)
    history_b.append(b)
    history_mse.append(np.mean(error ** 2))


    if iteration % 100 == 0:

        line1.set_data(history_iteration, history_w)
        line2.set_data(history_iteration, history_b)
        line3.set_data(history_iteration, history_mse)

        for ax in axes.flat:
            if ax.has_data():
                ax.relim()
                ax.autoscale_view()

        plt.pause(0.001)

plt.ioff()
plt.show()