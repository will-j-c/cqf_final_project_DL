import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-v0_8')
from umap import UMAP

# Plot price and volatility 
def plot_price_vol(df, close_col='Close', vol_col='Volume'):
    # Credit to this article for the basic flow for this method https://medium.com/analytics-vidhya/visualizing-historical-stock-price-and-volume-from-scratch-46029b2c5ef9
    fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    fig.tight_layout(pad=3)
    close = df[close_col]
    vol = df[vol_col]
    # Set the axis
    plot_price = axes[0]
    plot_vol = axes[1]
    # Plot the charts
    plot_price.plot(close)
    plot_vol.bar(vol.index, vol, width=15)
    # Move the ticks to the right
    plot_price.yaxis.tick_right()
    plot_vol.yaxis.tick_right()
    # Set the labels
    plot_price.yaxis.set_label_position('right')
    plot_price.set_ylabel('Close')
    plot_vol.set_ylabel('Volume')
    plot_vol.yaxis.set_label_position('right')
    # Set the grid lines
    plot_price.grid(axis='y', linestyle='-', linewidth=0.5)
    plot_vol.grid(axis='y', linestyle='-', linewidth=0.5)
    # Remove the top left borders
    for plot in [plot_price, plot_vol]:
        plot.spines['top'].set_visible(False)
        plot.spines['left'].set_visible(False)
    plt.show()

# Plot train and validation metrics
def plot_metrics(metric_values, val_metric_values, type):
    epochs = range(1, len(metric_values) + 1)
    plt.plot(epochs, metric_values, 'o', label=f'Training {type}')
    plt.plot(epochs, val_metric_values, '', label=f'Validation {type}')
    plt.title(f'Training and validation {type}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{type.title()}')
    plt.legend()
    plt.show()
    
def plot_umap():
    n_neighbors_arr = [5, 10, 20]
    min_dists = [0.05, 0.1, 0.3]
    fig, ax = plt.subplots(3, 3)
    for i, min_dist in enumerate(min_dists):
        print(i, min_dist)
        for j, n in enumerate(n_neighbors_arr):
            print(j, n)    
            reducer = UMAP(min_dist=min_dist, n_neighbors=n)
            embedding = reducer.fit_transform(X_train)
            params = reducer.get_params()
            n_neighbors = params['n_neighbors']
            min_dist = params['min_dist']
            ax[i][j].scatter(embedding[:, 0], embedding[:, 1], s=3)
            ax[i][j].set_title(f'min_dist: {min_dist} n_neighbors: {n_neighbors}', fontsize=8)
            ax[i][j].tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)
    plt.show()