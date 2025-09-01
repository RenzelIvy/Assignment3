import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# -----------------------------
# Format dates for the charts
# -----------------------------
sns.set_style("whitegrid")
sns.set_palette("Set2")

# Helper function to format dates
def format_dates(dates):
    return [pd.to_datetime(d).strftime("%b %d") for d in dates]

# -----------------------------
# Generate Synthetic Dataset (Hourly for 1 Month)
# -----------------------------
np.random.seed(42)

n_days = 30 #The whole month of June
hours_per_day = 24 # hours per day
n_rows = n_days * hours_per_day  # 720 rows

product_lines = ["Chocolate", "Dairy", "Coffee", "Frozen Foods"]
statuses = ["Normal", "Boiler High", "Chiller High", "Chiller Off", "Idle"]

# Create hourly datetime range
date_rng = pd.date_range(start="2025-06-01", periods=n_rows, freq="h")

data = {
    "Date": date_rng,  # keep datetime for plotting
    "Shift": np.where(
        date_rng.hour < 8, "Night",
        np.where(date_rng.hour < 16, "Morning", "Afternoon")
    ),
    "Boiler_Load_kg_hr": np.random.randint(3500, 5200, n_rows),
    "Boiler_Steam_ton_hr": np.round(np.random.uniform(3.2, 4.5, n_rows), 1),
    "Boiler_Fuel_L": np.random.randint(450, 650, n_rows),
    "Chiller_Load_kW": np.random.randint(200, 350, n_rows),
    "Chiller_RunTime_hr": np.round(np.random.uniform(3, 9, n_rows), 1),
    "Product_Line": np.random.choice(product_lines, n_rows),
    "Energy_Cost_USD": np.random.randint(1000, 1400, n_rows),
    "Status": np.random.choice(statuses, n_rows)
}

df = pd.DataFrame(data)

# -----------------------------
# Introduce Inefficiency Events
# -----------------------------
idle_boiler_indices = np.random.choice(df.index, size=20, replace=False)
df.loc[idle_boiler_indices, ["Boiler_Load_kg_hr", "Boiler_Steam_ton_hr"]] = 0
df.loc[idle_boiler_indices, "Boiler_Fuel_L"] = np.random.randint(400, 500, len(idle_boiler_indices))
df.loc[idle_boiler_indices, "Status"] = "Boiler Waste"
df.loc[idle_boiler_indices, "Energy_Cost_USD"] += 200

idle_chiller_indices = np.random.choice(df.index, size=15, replace=False)
df.loc[idle_chiller_indices, "Chiller_Load_kW"] = 0
df.loc[idle_chiller_indices, "Chiller_RunTime_hr"] = np.random.uniform(4, 7, len(idle_chiller_indices))
df.loc[idle_chiller_indices, "Status"] = "Chiller Waste"
df.loc[idle_chiller_indices, "Energy_Cost_USD"] += 150

spike_indices = np.random.choice(df.index, size=10, replace=False)
df.loc[spike_indices, "Energy_Cost_USD"] += 300
df.loc[spike_indices, "Status"] = "Scheduling Issue"

# -----------------------------
# Export Dataset
# -----------------------------
df.to_csv("test.csv", index=False)

# -----------------------------
# Summary Aggregates
# -----------------------------
daily_summary = df.groupby(df["Date"].dt.date).agg(
    Total_Energy_Cost=("Energy_Cost_USD", "sum"),
    Avg_Energy_Cost=("Energy_Cost_USD", "mean"),
    Avg_Boiler_Load=("Boiler_Load_kg_hr", "mean"),
    Avg_Chiller_Load=("Chiller_Load_kW", "mean"),
    Waste_Events=("Status", lambda x: (x.isin(["Boiler Waste", "Chiller Waste", "Scheduling Issue"])).sum())
).round(2)

waste_daily = df[df["Status"].isin(["Boiler Waste","Chiller Waste","Scheduling Issue"])].groupby(df["Date"].dt.date).size()

# -----------------------------
# 1. Status Distribution Pie Chart with Legend
# -----------------------------
plt.figure(figsize=(10,6))
status_counts = df['Status'].value_counts()
plt.pie(status_counts, labels=None, autopct='%1.1f%%', startangle=90)
plt.title("Status Distribution (June 2025)")
plt.legend(status_counts.index, title="Status", loc="center left", bbox_to_anchor=(1, 0.5))
plt.show()

# -----------------------------
# 2. Average Energy Cost per Shift (Bar Chart)
# -----------------------------
plt.figure(figsize=(8,5))
sns.barplot(data=df, x="Shift", y="Energy_Cost_USD", errorbar=None)

# Calculate the average energy cost per shift
avg_energy_cost_per_shift = df.groupby("Shift")["Energy_Cost_USD"].mean()

# Add the average values as text below the chart
y_offset = 30  # Offset to place the text slightly below the plot
for i, (shift, avg_cost) in enumerate(avg_energy_cost_per_shift.items()):
    plt.text(i, y_offset, f'{avg_cost:.2f}', ha='center', fontsize=12, fontweight='bold')

plt.title("Average Energy Cost per Shift")
plt.ylabel("Avg Cost (USD)")
plt.xlabel("")
plt.show()

# -----------------------------
# 3. Daily Waste Events (Line Chart)
# -----------------------------
waste_daily = df[df["Status"].isin(["Boiler Waste","Chiller Waste","Scheduling Issue"])]
waste_daily_count = waste_daily.groupby(waste_daily["Date"].dt.date).size()

plt.figure(figsize=(12,6))
sns.lineplot(
    x=format_dates(waste_daily_count.index),
    y=waste_daily_count.values,
    marker="o",
    color="crimson"
)
plt.title("Daily Waste Events (June 2025)")
plt.xlabel("Date")
plt.ylabel("Number of Waste Events")
plt.xticks(rotation=45)
plt.show()

# -----------------------------
# 4. Daily Energy Cost by Product Line (Line Chart)
# -----------------------------
product_daily = df.groupby([df["Date"].dt.date, "Product_Line"])["Energy_Cost_USD"].sum().unstack()
plt.figure(figsize=(12,6))
for product in product_daily.columns:
    plt.plot(
        format_dates(product_daily.index),
        product_daily[product],
        marker='o',
        label=product
    )
plt.title("Daily Energy Cost by Product Line")
plt.xlabel("Date")
plt.ylabel("Energy Cost (USD)")
plt.xticks(rotation=45)
plt.legend(title="Product Line")
plt.show()

# -----------------------------
# 5. Daily Energy Cost by Shift (Line Chart)
# -----------------------------
shift_daily = df.groupby([df["Date"].dt.date, "Shift"])["Energy_Cost_USD"].sum().unstack()
plt.figure(figsize=(12,6))
for shift in shift_daily.columns:
    plt.plot(
        format_dates(shift_daily.index),
        shift_daily[shift],
        marker='o',
        label=shift
    )
plt.title("Daily Energy Cost by Shift")
plt.xlabel("Date")
plt.ylabel("Energy Cost (USD)")
plt.xticks(rotation=45)
plt.legend(title="Shift")
plt.show()


# -----------------------------
# 6. Boiler vs Chiller Load (Colored by Status)
# -----------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    data=df, x="Boiler_Load_kg_hr", y="Chiller_Load_kW",
    hue="Status", palette="Set2", alpha=0.7
)
plt.title("Boiler Load vs Chiller Load (Colored by Status)")
plt.xlabel("Boiler Load (kg/hr)")
plt.ylabel("Chiller Load (kW)")
plt.legend(bbox_to_anchor=(.9, .5))
plt.show()
