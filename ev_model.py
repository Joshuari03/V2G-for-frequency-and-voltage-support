"""EV fleet model and state of charge logic."""

import numpy as np
import pandas as pd

# Physical parameters
N_EV         = 500       # number of EVs in the fleet
E_BAT        = 60.0      # battery capacity [kWh]
P_CHARGE     = 7.4       # AC wallbox charge power [kW] (Mode 2)
P_DISCHARGE  = 5.0       # V2G discharge power [kW] (slightly lower)
SOC_MIN      = 0.20      # minimum SoC to protect battery
SOC_MAX      = 0.95      # maximum SoC (avoid 100 %)
EFFICIENCY   = 0.92      # round-trip charge/discharge efficiency
SIM_START_H  = 6         # simulation starts at 06:00


def generate_fleet(seed=42):
    """
    Create a DataFrame of 500 EVs with random arrival/departure times and arrival SoC.
    Returns columns: arrival_h, departure_h, soc_arrival, soc_current.
    """
    rng = np.random.default_rng(seed)

    arrival     = rng.normal(loc=18.0, scale=1.5, size=N_EV)
    departure   = rng.normal(loc=8.0,  scale=1.0, size=N_EV)
    soc_arrival = rng.uniform(low=SOC_MIN, high=0.65, size=N_EV)

    arrival   = np.clip(arrival,   15.0, 23.0)
    departure = np.clip(departure,  6.0, 10.0)

    fleet = pd.DataFrame({
        'arrival_h'  : arrival,
        'departure_h': departure,
        'soc_arrival': soc_arrival,
        'soc_current': soc_arrival.copy()
    })
    return fleet


def _precharge_to_sim_start(fleet: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-compute SoC at SIM_START_H (06:00), simulating overnight charging
    from arrival_h until 06:00 next morning.

    After this, soc_current is near SOC_MAX for most EVs.
    The simulation then correctly shows near-zero load in the morning.
    When EVs depart and return in the evening, soc_current is reset to
    soc_arrival (post-trip value), producing the realistic evening peak.
    """
    fleet = fleet.copy()
    for i in range(len(fleet)):
        arr = fleet.at[i, 'arrival_h']
        soc = fleet.at[i, 'soc_arrival']

        hours_charging = int((24 - arr) + SIM_START_H)   # e.g. arr=18 -> 12 h

        for _ in range(hours_charging):
            if soc >= SOC_MAX:
                break
            energy_needed  = (SOC_MAX - soc) * E_BAT
            energy_charged = min(P_CHARGE * EFFICIENCY, energy_needed)
            soc += energy_charged / E_BAT

        fleet.at[i, 'soc_current'] = np.clip(soc, SOC_MIN, SOC_MAX)

    return fleet


def _is_connected(h: float, arr: float, dep: float) -> bool:
    """Return True if the EV is plugged in at hour h.
    dep < arr always (departure is next morning after overnight charge).
    """
    return (h >= arr) or (h < dep)


def simulate_g2v(fleet: pd.DataFrame) -> np.ndarray:
    """
    Uncontrolled G2V: every EV charges at full power as soon as it arrives.

    Key logic:
    - At sim start (06:00), EVs are nearly full after overnight charge.
    - They depart in the morning (disconnected while away).
    - When they RECONNECT in the evening, soc_current is reset to soc_arrival
      (reflecting energy consumed during the day's driving).
    - This produces the realistic evening charging peak at 18-20.
    """
    fleet = _precharge_to_sim_start(fleet)
    total_power   = np.zeros(24)
    prev_connected = np.array([
        _is_connected(SIM_START_H, fleet.at[i, 'arrival_h'], fleet.at[i, 'departure_h'])
        for i in range(N_EV)
    ], dtype=bool)

    for slot in range(24):
        h = (SIM_START_H + slot) % 24

        for i in range(N_EV):
            arr = fleet.at[i, 'arrival_h']
            dep = fleet.at[i, 'departure_h']
            connected = _is_connected(h, arr, dep)

            # Reconnection event: EV just returned from a trip.
            # Reset SoC to soc_arrival (post-driving, partially depleted).
            # Without this, EVs stay at SOC_MAX all day and show zero
            # charging demand in the evening (physically wrong).
            if connected and not prev_connected[i]:
                fleet.at[i, 'soc_current'] = fleet.at[i, 'soc_arrival']

            prev_connected[i] = connected

            soc = fleet.at[i, 'soc_current']
            if connected and soc < SOC_MAX:
                energy_needed  = (SOC_MAX - soc) * E_BAT
                energy_charged = min(P_CHARGE * EFFICIENCY, energy_needed)
                fleet.at[i, 'soc_current'] += energy_charged / E_BAT
                fleet.at[i, 'soc_current']  = min(fleet.at[i, 'soc_current'], SOC_MAX)
                total_power[slot] += P_CHARGE

    return total_power


def simulate_v2g(fleet: pd.DataFrame, v_pu_profile: np.ndarray,
                 pv_surplus: np.ndarray) -> np.ndarray:
    """
    Smart V2G: each EV charges/discharges based on bus voltage and PV surplus.

    v_pu_profile : array (24,) avg. voltage in p.u.
    pv_surplus   : array (24,) excess PV over base load [kW]
    Returns net fleet power [kW] (24,): positive = draw, negative = injection.
    """
    fleet = _precharge_to_sim_start(fleet)
    total_power    = np.zeros(24)
    prev_connected = np.array([
        _is_connected(SIM_START_H, fleet.at[i, 'arrival_h'], fleet.at[i, 'departure_h'])
        for i in range(N_EV)
    ], dtype=bool)

    for slot in range(24):
        h  = (SIM_START_H + slot) % 24
        v  = v_pu_profile[slot]
        ps = pv_surplus[slot]

        for i in range(N_EV):
            arr = fleet.at[i, 'arrival_h']
            dep = fleet.at[i, 'departure_h']
            connected = _is_connected(h, arr, dep)

            # Reset SoC on reconnection (same logic as G2V)
            if connected and not prev_connected[i]:
                fleet.at[i, 'soc_current'] = fleet.at[i, 'soc_arrival']

            prev_connected[i] = connected

            if not connected:
                continue

            soc = fleet.at[i, 'soc_current']

            if v > 1.03 and soc < SOC_MAX:
                p = P_CHARGE
                fleet.at[i, 'soc_current'] += (p * EFFICIENCY) / E_BAT
            elif v < 0.97 and soc > SOC_MIN + 0.1:
                p = -P_DISCHARGE
                fleet.at[i, 'soc_current'] -= P_DISCHARGE / (E_BAT * EFFICIENCY)
            elif ps > 0 and soc < SOC_MAX:
                p = min(P_CHARGE, ps / N_EV * 10)
                fleet.at[i, 'soc_current'] += (p * EFFICIENCY) / E_BAT
            else:
                p = 0

            fleet.at[i, 'soc_current'] = np.clip(
                fleet.at[i, 'soc_current'], SOC_MIN, SOC_MAX
            )
            total_power[slot] += p

    return total_power


if __name__ == "__main__":
    fleet = generate_fleet(seed=42)
    print(fleet.head(10))
    print(f"\nMean arrival SoC:  {fleet['soc_arrival'].mean():.2%}")
    print(f"Mean arrival time: {fleet['arrival_h'].mean():.1f}h")

    power_g2v = simulate_g2v(fleet)

    peak_slot = int(power_g2v.argmax())
    peak_hour = (SIM_START_H + peak_slot) % 24
    print(f"\nG2V peak load: {power_g2v.max():.1f} kW at {peak_hour:02d}:00")
    print()
    for slot in range(24):
        h   = (SIM_START_H + slot) % 24
        bar = "█" * int(power_g2v[slot] / 100)
        print(f"  {h:02d}:00  {power_g2v[slot]:6.1f} kW  {bar}")