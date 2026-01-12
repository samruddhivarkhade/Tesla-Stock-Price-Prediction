import matplotlib.pyplot as plt
import numpy as np

'''
def plot_predictions(actual_prices, predicted_prices, title="Stock Price Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color="blue", label="Actual Price")
    plt.plot(predicted_prices, color="red", label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.show()
'''
'''
def plot_predictions(actual_prices, predicted_prices, title="Stock Price Prediction"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(actual_prices, color="blue", label="Actual Price")
    ax.plot(predicted_prices, color="red", label="Predicted Price")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price (USD)")
    ax.legend()
    ax.grid(True)
    return fig 
    '''
def plot_predictions(actual_prices, predicted_prices, title="Stock Price Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Price")
    plt.plot(predicted_prices, label="Predicted Price")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Stock Price (USD)")
    plt.legend()
    plt.grid(True)
    return plt.gcf()
