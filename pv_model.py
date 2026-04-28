"""PV generation profile model."""

"""
pv_model.py
-----------
Generates a 24-hour PV output profile using pvlib clearsky model.
Location: Xi'an, China (34.3°N, 108.9°E).

Returns:
  - hourly PV power [kW] for the total installed capacity (PV_TOTAL_KW)
  - per-bus PV array via distribute_pv_output() in network_setup.py

Two profiles are available:
  - clearsky  : ideal sunny day (summer solstice by default)
  - cloudy    : clearsky scaled by a simple cloud factor (optional scenario)
"""

import numpy as np
import pandas as pd
import pvlib
import matplotlib.pyplot as plt

# ── Location: Xi'an, Shaanxi, China ──────────────────────────────────────────
LATITUDE   =  34.3
LONGITUDE  = 108.9
ALTITUDE   = 400       # metres above sea level
TIMEZONE   = 'Asia/Shanghai'

# ── PV system parameters ──────────────────────────────────────────────────────
PV_TOTAL_KW    = 500.0    # total installed capacity across the network [kW]
PANEL_EFF      = 0.20     # panel efficiency (mono-Si, ~20%)
PR             = 0.80     # performance ratio (inverter + wiring + temp losses)
# Effective yield factor: W_output / W_irradiance = efficiency × PR

# ── Simulation date ───────────────────────────────────────────────────────────
# Summer solstice → maximum solar resource → worst-case PV surplus scenario.
# Change to e.g. '2024-12-21' for winter (lower generation, different dynamics).
SIM_DATE = '2024-06-21'


def get_pv_profile(date: str = SIM_DATE,
                   cloud_factor: float = 1.0) -> np.ndarray:
    """
    Compute hourly PV output [kW] for a full 24-hour day.

    Parameters
    ----------
    date         : ISO date string, e.g. '2024-06-21'
    cloud_factor : 1.0 = clear sky, 0.5 = 50% cloud cover, etc.

    Returns
    -------
    pv_kw : np.ndarray shape (24,), one value per hour starting at 00:00.
            Slot 0 = 00:00, slot 6 = 06:00, slot 12 = 12:00, etc.
    """
    location = pvlib.location.Location(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        altitude=ALTITUDE,
        tz=TIMEZONE
    )

    # Hourly timestamps for the chosen day
    times = pd.date_range(
        start=date,
        periods=24,
        freq='h',
        tz=TIMEZONE
    )

    # Clearsky irradiance (Ineichen model — accurate for mid-latitude sites)
    clearsky = location.get_clearsky(times)   # columns: ghi, dni, dhi [W/m²]
    ghi = clearsky['ghi'].values              # Global Horizontal Irradiance

    # Convert GHI → AC power output
    # P_ac [kW] = GHI [W/m²] × (efficiency × PR) × (PV_TOTAL_KW / 1000 W/m² ref)
    # The "1000" normalises against the STC irradiance reference
    pv_kw = (ghi / 1000.0) * PV_TOTAL_KW * PR * cloud_factor

    return pv_kw   # shape (24,)


def get_pv_surplus(pv_kw: np.ndarray,
                   base_load_kw: np.ndarray) -> np.ndarray:
    """
    Compute PV surplus: generation minus base load, floored at zero.
    Surplus > 0 means the grid has excess energy available for EV charging.

    Parameters
    ----------
    pv_kw        : shape (24,) PV output [kW]
    base_load_kw : shape (24,) total network base load [kW]

    Returns
    -------
    surplus_kw : shape (24,) excess PV [kW], zero when load > generation
    """
    return np.maximum(pv_kw - base_load_kw, 0.0)


def align_to_sim_start(arr: np.ndarray, sim_start_h: int = 6) -> np.ndarray:
    """
    Re-index a 24-element array so that index 0 corresponds to sim_start_h.
    Matches the time axis used by simulate_g2v / simulate_v2g in ev_model.py.

    Example: sim_start_h=6 → slot 0 = 06:00, slot 6 = 12:00, slot 18 = 00:00
    """
    return np.roll(arr, -sim_start_h)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pv_raw = get_pv_profile(date=SIM_DATE, cloud_factor=1.0)
    pv_sim = align_to_sim_start(pv_raw, sim_start_h=6)

    print(f"PV profile — Xi'an, {SIM_DATE}")
    print(f"Total daily yield : {pv_raw.sum():.1f} kWh")
    print(f"Peak output       : {pv_raw.max():.1f} kW at {pv_raw.argmax():02d}:00")
    print()

    # ── Tabella oraria ────────────────────────────────────────────────────────
    print(f"{'Hour':>6}  {'PV [kW]':>9}  {'Profile':}")
    print("-" * 40)
    for h in range(24):
        bar  = "█" * int(pv_raw[h] / 10)
        mark = " ← peak" if h == int(pv_raw.argmax()) else ""
        print(f"  {h:02d}:00  {pv_raw[h]:7.1f}  {bar}{mark}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    hours = np.arange(24)
    pv_cloudy = get_pv_profile(date=SIM_DATE, cloud_factor=0.5)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(hours, pv_raw,    alpha=0.4, color='gold',       label='Clear sky (CF=1.0)')
    ax.fill_between(hours, pv_cloudy, alpha=0.4, color='steelblue',  label='Cloudy (CF=0.5)')
    ax.plot(hours, pv_raw,    color='orange', linewidth=2)
    ax.plot(hours, pv_cloudy, color='steelblue', linewidth=2)

    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Hour of day')
    ax.set_ylabel('PV Output [kW]')
    ax.set_title(f'PV Generation Profile — Xi\'an {SIM_DATE} (total {PV_TOTAL_KW:.0f} kW installed)')
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h:02d}:00' for h in hours], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/pv_profile.png', dpi=150)
    plt.show()
    print("\nPlot saved → results/pv_profile.png")
