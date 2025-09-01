import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed so results are repeatable
np.random.seed(34)

# Number of rows (e.g., production shifts)
n = 200

# Possible machines and shifts
machines = [
    "West Blg.",
    "North Blg.",
    "East Blg.",
    "South Blg.",
]

shift = ["Morning", "Afternoon", "Night"]

# Generate random data
data = {
    "Machine Location": np.random.choice(machines, n),
    "Shift": np.random.choice(shift, n),
    "Cookies Produced": np.random.randint(200, 700, n),        # between 200–700 cookies
    "Downtime Minutes": np.random.randint(0, 60, n),         # 0–60 minutes downtime
    "Defected Cookies": np.random.randint(0, 200, n),                 # 0–200 defective cookies
    "Energy Used in kWh": np.random.randint(500, 700, n)        # 500–700 kWh per shift
}

# Put into DataFrame
cookie_df = pd.DataFrame(data)

print(cookie_df.head(50))   # show first 50 rows

cookie_df.to_csv("manufacturing_data.csv", index=False)

cookie_df["DefectRate"] = cookie_df["Defected Cookies"] / cookie_df["Cookies Produced"]
cookie_df["DowntimeRate"] = cookie_df["Downtime Minutes"] / 480   # assuming 8h shifts
cookie_df["EnergyPerUnit"] = cookie_df["Energy Used in kWh"] / cookie_df["Cookies Produced"]


#Cookies Produced Per location
plt.figure(figsize=(10,5))
sns.barplot(x="Machine Location", y="Cookies Produced", data=cookie_df, estimator="mean", errorbar=None)
plt.xticks(rotation=45)
plt.title("Average Cookie Production per Location")
plt.show()

#Defect Rate by Machine
plt.figure(figsize=(10,5))
sns.barplot(x="Machine Location", y="DefectRate", data=cookie_df, estimator="mean", errorbar=None)
plt.xticks(rotation=45)
plt.title("Average Defect Rate per Location")
plt.show()

# Grouped bar: Energy per Unit by Machine & Shift
plt.figure(figsize=(12,6))
sns.barplot(x="Machine Location", y="EnergyPerUnit", hue="Shift",
            data=cookie_df, estimator="mean", errorbar=None)
plt.ylabel("kWh per Cookie")
plt.title("Energy per Unit by Machine and Shift")
plt.legend(title="Shift")
plt.show()

# Grouped bar: Downtime Rate by Machine & Shift
plt.figure(figsize=(12,6))
sns.barplot(x="Machine Location", y="DowntimeRate", hue="Shift",
            data=cookie_df, estimator="mean", errorbar=None)
plt.ylabel("Downtime Rate (per shift)")
plt.title("Downtime Rate by Machine and Shift")
plt.legend(title="Shift")
plt.show()

# Pivot for heatmap (Energy per Unit)
energy_pivot = cookie_df.pivot_table(
    index="Machine Location", columns="Shift",
    values="EnergyPerUnit", aggfunc="mean"
)

plt.figure(figsize=(8,6))
sns.heatmap(energy_pivot, annot=True, fmt=".3f", cmap="YlOrRd")
plt.title("Energy per Unit (kWh per Cookie) by Machine and Shift")
plt.show()

# Pivot for heatmap (Downtime Rate)
downtime_pivot = cookie_df.pivot_table(
    index="Machine Location", columns="Shift",
    values="DowntimeRate", aggfunc="mean"
)

plt.figure(figsize=(8,6))
sns.heatmap(downtime_pivot, annot=True, fmt=".3f", cmap="Blues")
plt.title("Downtime Rate by Machine and Shift")
plt.show()

