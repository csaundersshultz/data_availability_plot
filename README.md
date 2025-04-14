
![data_avail_2](https://github.com/user-attachments/assets/84e815d3-039d-4e7f-89c0-200e8ebfb70f)


# 📈 Data availability visualizer for SQLite (rover) databases

This Python tool visualizes the **data availability percentage of multiple stations** over time using horizontal bar charts. It queries a local sqlite file, intended to be one connected to a local rover database. 

---

## ✨ Features

- Horizontal bar plots with time on the x-axis and stations on the y-axis.
- Color-coded bars based on uptime percentages in custom intervals (default is daily)
- Supports Network, Station, Location, and Channel query parameters
- Built using `matplotlib` and `pandas`.

---

## 🛠 USAGE
```python
from data_availability_plot import data_availability_plot
import matplotlib.pyplot as plt

sqlite_path = "/path/to/rover/datarepo/timeseries.sqlite"

fig, ax = availability_plot(sqlite_path, network="A*", station="", location="", channel="HDF", interval_days=1, max_chunk_days=365)
plt.show()
plt.close()
```

---

--- 
## ⬇️ Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/me/data_availability_plot.git
cd data_availability_plot
pip install -e .
```

---

## 📦 Requirements

Requires Obspy==1.4.1
Requires Numpy, Matplotlib, Pandas, 
Requires a valid and accessible sqlite file with a tsindex column (created by mseedindex / rover)





