
![data_availability_plot](https://github.com/user-attachments/assets/70a4b08e-bf56-4379-969f-4cbee6ef8ec0)

# 📈 Data availability visualizer for SQLite (rover) databases

This Python tool visualizes the **data availability percentage of multiple stations** over time using horizontal bar charts. It queries a local sqlite file, intended to be one connected to a local rover database. 

---

## ✨ Features

- Horizontal bar plots with time on the x-axis and stations on the y-axis.
- Color-coded bars based on uptime percentages in custom intervals (default is daily)
- Supports Network, Station, Location, and Channel query parameters
- Built using `matplotlib` and `pandas`.

---

## 📦 Requirements

Requires Obspy==1.4.1
Requires Numpy, Matplotlib, Pandas, 
