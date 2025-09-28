# Creating a simulated marketing dataset for 3 cruise products (Caribbean, Mediterranean, Fjords)
# Weekly data for 3 years (156 weeks). This code generates:
# - weekly media spend by channel (TV, Social, Search, Display, Email)
# - adstocked and saturated channel effects
# - product-level sales (bookings) with seasonality, price promotions, and economic index
# - saves CSV to /mnt/data/cruise_marketing_simulated.csv and displays the first rows
#
# The generated CSV can be downloaded from the returned path below.
# %%
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt

np.random.seed(42)

# %%

# Parameters
start_date = "2019-01-07"  # Monday
weeks = 156  # ~3 years of weekly data
dates = pd.date_range(start=start_date, periods=weeks, freq='W-MON')


channels = ["TV", "Social", "Search", "Display", "Email"]
products = ["Caribbean", "Mediterranean", "Fjords"]

# Channel-level base weekly spends (in thousands of dollars)
base_spend = {"TV": 300, "Social": 80, "Search": 120, "Display": 40, "Email": 10}

# %%

# Channel seasonality patterns (multiplier by week of year)
def week_seasonality_multiplier(week_of_year):
    # Stronger spend in Jan-Feb (weeks 1-8), and Apr-Jun (weeks 14-26)
    if 1 <= week_of_year <= 8:
        return 1.25  # Strong Jan-Feb
    if 14 <= week_of_year <= 26:
        return 1.15  # Strong Apr-Jun
    # Weaker in December (weeks 48-52)
    if 48 <= week_of_year <= 52:
        return 0.75  # Very weak in December due to Christmas
    if 36 <= week_of_year <= 44:
        return 0.9   # Weaker in autumn
    return 1.0

# Generate raw channel spends with noise and occasional campaign spikes
channel_spend = {c: [] for c in channels}
for i, date in enumerate(dates):
    w = date.isocalendar()[1]
    season_mult = week_seasonality_multiplier(w)
    for c in channels:
        base = base_spend[c] * season_mult
        # random week-to-week variation
        spend = np.random.normal(loc=base, scale=base*0.15)
        # occasional campaign spikes (5% chance)
        if np.random.rand() < 0.05:
            spend *= np.random.uniform(1.8, 3.5)
        spend = max(spend, 0.0)
        channel_spend[c].append(spend)

channel_df = pd.DataFrame(channel_spend, index=dates).rename_axis('date').reset_index()

# %%

# Economic index (normalized around 1.0 with small variations and occasional shocks)
econ = []
val = 1.0
for i in range(weeks):
    np.random.seed(63+i)  # different seed per week for reproducibility
    # Add small random fluctuation to the economic index each week; occasionally, apply a negative "shock" (rare, larger drop)
    val += np.random.normal(0, 0.005)
    if np.random.rand() < 0.02:
        val += np.random.normal(-0.005, 0.003)
    econ.append(max(val, 0.7))
econ = np.array(econ)

plt.figure(figsize=(10, 4))
plt.plot(econ, label="Economic Index", color="blue")
plt.title("Simulated Economic Index Over Weeks")
plt.xlabel("Week")
plt.ylabel("Index Value")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.show()

# %%

# Price promotions per product (binary indicator & discount level)
promo_indicator = {p: [] for p in products}
promo_discount = {p: [] for p in products}
for i in range(weeks):
    for p in products:
        if np.random.rand() < 0.08:  # ~8% of weeks have a promo for a product
            promo_indicator[p].append(1)
            promo_discount[p].append(np.random.uniform(0.05, 0.25))  # 5% to 25% discount
        else:
            promo_indicator[p].append(0)
            promo_discount[p].append(0.0)
            
# print("Keys:", promo_discount.keys())
# print("Values:", promo_discount.values())

            
# %%

# Adstock and saturation functions
def apply_adstock(spend_series, decay=0.5):
    # geometric adstock: adstock_t = spend_t + decay * adstock_{t-1}
    adstock = np.zeros_like(spend_series)
    for t in range(len(spend_series)):
        if t == 0:
            adstock[t] = spend_series[t]
        else:
            # Geometric adstock: adstock_t = spend_t + decay * adstock_{t-1}
            adstock[t] = spend_series[t] + decay * adstock[t-1]
    return adstock

def hill_saturation(x, alpha=1.0, gamma=1.0, saturation_half=100.0):
    # Hill function: effect = alpha * x^gamma / (x^gamma + saturation_half^gamma)
    return alpha * (x**gamma) / (x**gamma + saturation_half**gamma)


# %%

# Product-channel elasticities (how responsive each product is to each channel)
# expressed as base effect multipliers (per 1000$ of spend)
elasticities = {
    "Caribbean": {"TV": 0.18, "Social": 0.28, "Search": 0.22, "Display": 0.10, "Email": 0.08},
    "Mediterranean": {"TV": 0.22, "Social": 0.18, "Search": 0.26, "Display": 0.12, "Email": 0.07},
    "Fjords": {"TV": 0.12, "Social": 0.10, "Search": 0.30, "Display": 0.08, "Email": 0.06}
}

# Adstock decay per channel (some channels have longer carryover)
adstock_decay = {"TV": 0.7, "Social": 0.5, "Search": 0.4, "Display": 0.45, "Email": 0.2}
# Saturation half points per channel (higher means takes more spend to saturate)
saturation_half = {"TV": 400, "Social": 100, "Search": 120, "Display": 80, "Email": 20}

# Base weekly demand per product (bookings baseline)
base_weekly = {"Caribbean": 180.0, "Mediterranean": 140.0, "Fjords": 90.0}

# %%

# Build full dataset
rows = []
# precompute adstocked & saturated channel series
channel_spend_arr = {c: np.array(channel_df[c]) for c in channels}
channel_adstock = {c: apply_adstock(channel_spend_arr[c], decay=adstock_decay[c]) for c in channels}
channel_saturated = {c: hill_saturation(channel_adstock[c], alpha=1.0, gamma=1.0, saturation_half=saturation_half[c])
                     for c in channels}


# Define a minimum Poisson lambda for clarity
min_poisson_lambda = 0.1

for i, date in enumerate(dates):
    row = {"date": date, "week": i+1, "econ_index": econ[i]}
    # attach raw channel spend
    for c in channels:
        row[f"spend_{c}"] = channel_spend_arr[c][i]
        row[f"adstock_{c}"] = channel_adstock[c][i]
        row[f"saturated_{c}"] = channel_saturated[c][i]
    # promos
    for p in products:
        row[f"promo_{p}"] = promo_indicator[p][i]
        row[f"promo_discount_{p}"] = promo_discount[p][i]
    # product-level sales generation
    for p in products:
        # start with base demand
        demand = base_weekly[p]
        # seasonality multiplier (peak in summer weeks)
        w = date.isocalendar()[1]
        season_mult = 1.0
        if 14 <= w <= 36:
            season_mult = 1.35 if p != "Fjords" else 1.05  # Caribbean & Med peak in summer, Fjords less so
        if 48 <= w <= 52 or 1 <= w <= 6:
            season_mult = 0.85 if p != "Fjords" else 1.05  # Fjords may peak in Northern summer (different weeks)
        demand *= season_mult
        # economic multiplier (people buy more when econ_index >1)
        demand *= econ[i]
        # promo multiplier: increases bookings during promo
        if row[f"promo_{p}"] == 1:
            demand *= (1 + 1.2 * row[f"promo_discount_{p}"])  # promotions boost demand modestly
        # add media effects from saturated channel adstock
        media_effect = 0.0
        for c in channels:
            # elasticity scaled by spend (thousands). channel spend is in unit $ (we set base as thousands though)
            # convert spend to thousands for elasticity interpretation
            spend_thousands = row[f"adstock_{c}"]  # adstock in same units as spend (thousands)
            # saturated value is between 0 and 1 roughly, scale by channel-specific strength
            sat = row[f"saturated_{c}"]
            # product-specific elasticity: (bookings per 1000$); multiply by sat*elasticity
            media_effect += elasticities[p][c] * sat / 100.0  # divide 100 to scale effect into booking units
        # total expected bookings + gaussian noise
        expected = demand + demand * media_effect
        observed = np.random.poisson(lam=max(expected, min_poisson_lambda))  # counts: use Poisson to mimic bookings
        row[f"bookings_{p}"] = observed
    rows.append(row)

df = pd.DataFrame(rows)
# Reorder columns for readability: date, econ, spends, adstocks, saturated, promos, bookings
cols_order = ["date", "week", "econ_index"] + \
             [f"spend_{c}" for c in channels] + [f"adstock_{c}" for c in channels] + [f"saturated_{c}" for c in channels] + \
             sum([[f"promo_{p}", f"promo_discount_{p}"] for p in products], []) + \
             [f"bookings_{p}" for p in products]
df = df[cols_order]

# %%

## Plot Saturated Channel Effect for all channels
plt.figure(figsize=(14, 10))
saturated_cols = [f"saturated_{c}" for c in channels]

for i, col in enumerate(saturated_cols, 1):
    plt.subplot(3, 2, i)
    plt.plot(df['date'], df[col], label=col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.title(f'{col} vs Date')
    plt.legend()
    plt.tight_layout()

plt.show()
# %%

# Save CSV
out_path = "data/cruise_marketing_simulated.csv"
#df.to_csv(out_path, index=False)

# Display first rows to the user via the notebook frontend
# from caas_jupyter_tools import display_dataframe_to_user
# display_dataframe_to_user("Simulated Cruise Marketing Dataset (first 20 rows)", df.head(20))

# out_path

# %%
df.head(20)
# %%
