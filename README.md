# AI Model Visualizer 🤖📊

A simple machine learning visualization project built with **Python, NumPy, and Matplotlib**.

The project visualizes how a **Linear Regression** model learns its parameters using **Gradient Descent**.

Instead of only seeing the final prediction, you can watch the model learn over time.

## ✨ Features

* 📈 Linear Regression from scratch
* 🧠 Gradient Descent
* ⚙️ Adjustable learning rate
* 🔢 Adjustable number of iterations
* 📊 Live visualization during training
* `w` (weight) visualization
* `b` (bias) visualization
* 📉 MSE (Mean Squared Error) visualization
* 📐 Live regression line

## 🧠 How It Works

The model uses the following equation:

```text
ŷ = wx + b
```

Where:

* `w` — Weight
* `b` — Bias
* `x` — Input
* `ŷ` — Model prediction

During training, Gradient Descent repeatedly updates `w` and `b` to reduce the model's error.

The project also calculates **Mean Squared Error (MSE)** to track how well the model is learning.

## 📊 Visualizations

The application displays four graphs:

### 1. Weight (`w`)

Shows how the weight changes during training.

### 2. Bias (`b`)

Shows how the bias changes during training.

### 3. MSE

Shows the model's error during training.

A decreasing MSE generally means that the model is getting better at fitting the data.

### 4. Regression

Shows:

* ⚫ The original dataset
* 🟣 The model's current regression line

The regression line moves as the model updates `w` and `b`.

## 🚀 Usage

Install the required libraries:

```bash
pip install numpy matplotlib
```

Run the program:

```bash
python main.py
```

The program asks for:

```text
Iterations : 10000
Learning rate : 0.0001
```

You can experiment with different values and observe how they affect training.

## 🛠️ Built With

* Python
* NumPy
* Matplotlib
* Linear Regression
* Gradient Descent

## 📚 Purpose

This project was created as a learning project to better understand:

* How Linear Regression works
* How Gradient Descent updates model parameters
* The relationship between `w`, `b`, and predictions
* How MSE changes during training
* How machine learning models learn from data

The model is implemented from scratch without using machine learning libraries such as **scikit-learn**.

## 🔮 Future Plans

Possible future improvements:

* [ ] Display current `w`, `b`, and MSE directly on the graph
* [ ] Training speed controls
* [ ] Multiple datasets
* [ ] Compare different learning rates
* [ ] Better visualization controls
* [ ] Additional machine learning models

## 📄 License

This project is open source and available under the MIT License.
