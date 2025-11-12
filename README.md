# Utilities

A Python package containing utility functions for data analysis, plotting, and signal processing.

## Features

- **Time-series plotting**: Enhanced matplotlib functions for plotting time-based data
- **Data visualization**: Histograms, bar charts, Bland-Altman plots
- **Signal processing**: Fourier transform visualization using Welch's method or FFT
- **DataFrame utilities**: Convenient functions for working with pandas DataFrames
- **Interactive plotting**: Keyboard shortcuts for figure manipulation

## Installation

### From Source

Clone the repository and install:

```bash
git clone https://github.com/robetraks/Utilities.git
cd Utilities
pip install .
```

### Development Installation

For development, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

```python
from utilities import make_figure, plot_with_time, my_hist, plotFourierTransform

# Create a figure with custom layout
axs, fig = make_figure(name='My Plot', nrows=2, ncols=1)

# Plot time-series data
plot_with_time(time_data, values, ax=axs[0], title='Time Series')

# Create a histogram
my_hist(data, bins=20, title='Distribution')

# Analyze frequency spectrum
plotFourierTransform(signal, fs=100, window_size=256, overlap=128)
```

## Available Functions

### Plotting Functions
- `make_figure()`: Create customizable matplotlib figures with keyboard shortcuts
- `plot_with_time()`: Plot time-series data with formatted time axes
- `plot_dataframe()`: Plot all columns of a DataFrame in subplots
- `my_hist()`: Enhanced histogram with percentage/absolute options
- `my_bar()`: Bar chart with automatic labeling
- `plot_bland_altman()`: Bland-Altman plot for method comparison

### Signal Processing
- `plotFourierTransform()`: Compute and plot frequency spectrum

### Data Utilities
- `df_betweem_time()`: Filter DataFrame by time range
- `remove_bookends_timeseries()`: Remove start/end portions of time series
- `lat_long_semicircles_to_degree()`: Convert GPS coordinates

### Helper Functions
- `my_round()`: Round numbers to specified decimal places
- `draw_line()`: Draw horizontal reference lines
- `draw_line_yaxis()`: Draw vertical reference lines

## Interactive Features

When using `make_figure()`, the following keyboard shortcuts are available:
- **t**: Apply tight layout to the figure

## Requirements

- Python >= 3.7
- matplotlib >= 3.0.0
- pandas >= 1.0.0
- numpy >= 1.18.0
- scipy >= 1.4.0
- Pillow >= 7.0.0

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

Akshay Jain
