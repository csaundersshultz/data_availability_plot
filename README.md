
![data_avail_2](https://github.com/user-attachments/assets/84e815d3-039d-4e7f-89c0-200e8ebfb70f)


# 📈 Data availability visualizer for SQLite (rover) databases

This Python tool visualizes the **data availability percentage of multiple stations** over time using horizontal bar charts. It queries a local sqlite file, intended to be one connected to a local rover database. Note, this does not pull the full data availability from IRIS, it only shows what is downloaded locally and available in a rover database. Hopefully in the future this can be updated to accommodate other obspy clients (earthworm / fdsn).


---

## 🛠 USAGE
```python
from data_availability_plot import availability_plot, NullPoolTSIndexDatabaseHandler
import matplotlib.pyplot as plt

sqlite_path = "/path/to/rover/datarepo/timeseries.sqlite"

fig, ax = availability_plot(sqlite_path, network="A*", station="", location="", channel="HDF", interval_days=1, max_chunk_days=200)
plt.show()
plt.close()
```

---

--- 
## ⬇️ Installation

Clone the repository and move into the project directory:

```bash
git clone https://github.com/csaundersshultz/data_availability_plot
```

---

## 📦 Requirements

Requires Obspy==1.4.1
Requires Numpy, Matplotlib, Pandas, 
Requires a valid and accessible sqlite file with a tsindex column (created by mseedindex / rover)





