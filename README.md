# Ex.No: 08 MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
Date:
## AIM:
To implement Moving Average Model and Exponential smoothing Using Python.

## ALGORITHM:
Import necessary libraries
Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of the dataset
Set the figure size for plots
Suppress warnings
Plot the first 50 values of the 'Value' column
Perform rolling average transformation with a window size of 5
Display the first 10 values of the rolling mean
Perform rolling average transformation with a window size of 10
Create a new figure for plotting,Plot the original data and fitted value
Show the plot
Also perform exponential smoothing and plot the graph
## PROGRAM:
```
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')

# Read the Gold dataset from a CSV file
data = pd.read_csv('Gold Price Prediction.csv')  # Replace with the correct file path

# Display the shape and the first 20 rows of the dataset
print("Shape of the dataset:", data.shape)
print("First 20 rows of the dataset:")
print(data.head(20))

# Set the figure size for plots
plt.rcParams['figure.figsize'] = [10, 6]

# Plot the first 50 values of the 'Price Today' column
plt.plot(data['Price Today'][:50])
plt.title('First 50 Values of "Price Today"')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.grid(True)
plt.show()

# Perform rolling average transformation with a window size of 5 on 'Price Today'
rolling_mean_5 = data['Price Today'].rolling(window=5).mean()

# Display the first 10 values of the rolling mean
print("First 10 values of the rolling mean (window size 5):")
print(rolling_mean_5.head(10))

# Perform rolling average transformation with a window size of 10 on 'Price Today'
rolling_mean_10 = data['Price Today'].rolling(window=10).mean()

# Plot the original data and fitted value (rolling mean with window size 10)
plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(rolling_mean_10, label='Rolling Mean (window=10)', color='orange')
plt.title('Original Data and Rolling Mean (window=10)')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Perform exponential smoothing on 'Price Today' and plot the graph
alpha = 0.3  # Smoothing factor
exp_smooth = data['Price Today'].ewm(alpha=alpha).mean()

plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(exp_smooth, label='Exponential Smoothing', color='red')
plt.title('Original Data and Exponential Smoothing')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Implement the Moving Average (MA) Model
# Set the order (p, d, q) to (0, 0, q) for a pure MA model. Let's try q = 1 first.
q = 1
ma_model = ARIMA(data['Price Today'], order=(0, 0, q))
ma_model_fit = ma_model.fit()

# Display the summary of the fitted model
print(ma_model_fit.summary())

# Forecast the next 10 data points using the fitted model
forecast = ma_model_fit.forecast(steps=10)

# Plot the original data and the forecasted values
plt.figure()
plt.plot(data['Price Today'], label='Original Data')
plt.plot(range(len(data), len(data) + 10), forecast, label='MA Forecast', color='green', marker='o')
plt.title('Original Data and Moving Average Forecast')
plt.xlabel('Index')
plt.ylabel('Price Today')
plt.legend()
plt.grid(True)
plt.show()

# Print the forecasted values
print("Forecasted values for the next 10 points:")
print(forecast)
``` 
OUTPUT:
data:

![image](https://github.com/user-attachments/assets/6278de5f-dfe9-4f27-8e9d-1fc49c293fb8)

![image](https://github.com/user-attachments/assets/eb7c6b6c-072c-4bfb-82f0-1d33bed00d6c)


Price Today

![image](https://github.com/user-attachments/assets/5def2ea8-3773-481e-aa9f-d03bc8cc9064)

Original Data and Rolling Mean

![image](https://github.com/user-attachments/assets/37376e78-71b9-4f58-a809-6912da9f26b9)


Exponential Smoothing

![image](https://github.com/user-attachments/assets/59c10b33-5cd3-4500-b599-1fb650347590)

Moving average

![image](https://github.com/user-attachments/assets/678f2ee2-084d-4bc5-a219-4a2fa0e78504)


## RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
