import matplotlib.pyplot as plt
import numpy as np
plt.style.use('seaborn-v0_8')


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

def plot_scatter(x, y, x_ticks_entries, title):
    plt.scatter(x, y)
    plt.xticks(x_ticks_entries)
    # Plot trend line
    plt.title(title)
    plt.show()