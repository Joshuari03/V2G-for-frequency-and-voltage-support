"""Entry point for the V2G frequency and voltage control workflow."""

"""
main.py
-------
Central simulation loop for the V2G smart grid study.

Scenarios
---------
Each scenario is a combination of:
  - EV penetration  : 20 / 50 / 80 % of fleet connected
  - Control strategy: G2V (uncontrolled) vs V2G (smart bidirectional)
  - PV              : enabled / disabled  (toggle PV_ENABLED in network_setup.py)

Results are saved as CSV in results/ and passed to plots.py.
"""

import numpy as np
import pandas as pd
import pandapower as pp

from ev_model      import generate_fleet, simulate_g2v, simulate_v2g, N_EV, SIM_START_H
from network_setup import (build_network, update_network,
                           get_bus_voltages, get_line_loading,
                           distribute_ev_load, distribute_pv_output,
                           PV_ENABLED, N_LOAD_BUSES)
from pv_model      import get_pv_profile, get_pv_surplus, align_to_sim_start

import plots   # will call plots.py once results are ready

# ── Scenario definitions ──────────────────────────────────────────────────────
EV_PENETRATIONS = [0.2, 0.5, 0.8]    # fraction of full fleet active
STRATEGIES      = ['G2V', 'V2G']
SIM_DATE        = '2024-06-21'

# Base load profile: realistic 24h residential curve [MW total network]
# Values represent the sum across all 32 load buses.
# Shape loosely follows a typical Chinese urban feeder:
#   - morning ramp 06-09, midday plateau, evening peak 18-21, night trough
_BASE_LOAD_TOTAL_MW = np.array([
    0.8, 0.7, 0.65, 0.6, 0.6, 0.65,    # 00-05
    0.8, 1.0, 1.3,  1.5, 1.6, 1.65,    # 06-11
    1.7, 1.65,1.6,  1.5, 1.6, 1.8,     # 12-17
    2.0, 2.1, 2.0,  1.8, 1.5, 1.1,     # 18-23
])   # shape (24,)  — index 0 = 00:00

def _base_load_per_bus(h_abs: int) -> np.ndarray:
    """
    Return base load [MW] for each of the 32 buses at absolute hour h_abs.
    Distributed proportionally to the IEEE 33-bus nominal loads.
    """
    import pandapower.networks as pn
    net0      = pn.case33bw()
    weights   = net0.load['p_mw'].values / net0.load['p_mw'].sum()
    return weights * _BASE_LOAD_TOTAL_MW[h_abs % 24]


def run_scenario(penetration: float,
                 strategy: str,
                 fleet_full: pd.DataFrame,
                 pv_sim: np.ndarray) -> dict:
    """
    Run one 24-hour scenario and return a result dict.

    penetration : 0.2 / 0.5 / 0.8
    strategy    : 'G2V' or 'V2G'
    fleet_full  : full 500-EV DataFrame from generate_fleet()
    pv_sim      : shape (24,) PV output aligned to SIM_START_H
    """
    # Scale fleet to penetration level
    n_active = int(N_EV * penetration)
    fleet    = fleet_full.iloc[:n_active].copy().reset_index(drop=True)

    # ── Compute EV power profile (24 slots) ──────────────────────────────────
    # Base load for surplus calculation (aligned to sim start)
    base_load_sim = np.array([
        _BASE_LOAD_TOTAL_MW[(SIM_START_H + s) % 24] * 1000   # → kW
        for s in range(24)
    ])
    pv_surplus = get_pv_surplus(pv_sim, base_load_sim)

    def _run_power_flow(ev_profile_kw: np.ndarray) -> tuple:
        """Run PF across 24 slots; return voltages, loading, ev_kw arrays."""
        net = build_network(pv_enabled=PV_ENABLED)

        voltages_24h = np.zeros((24, 33))
        loading_24h  = np.zeros((24, 37))
        ev_kw_24h    = np.zeros(24)

        for slot in range(24):
            h_abs = (SIM_START_H + slot) % 24

            base_mw   = _base_load_per_bus(h_abs)              # (32,) MW
            ev_bus_kw = distribute_ev_load(ev_profile_kw[slot])
            pv_bus_kw = distribute_pv_output(pv_sim[slot])

            try:
                update_network(net, base_mw, ev_bus_kw, pv_bus_kw,
                               pv_enabled=PV_ENABLED)
                voltages_24h[slot] = get_bus_voltages(net)
                loading_24h[slot]  = get_line_loading(net)
            except pp.powerflow.LoadflowNotConverged:
                print(f"  [WARN] Power flow did not converge at slot {slot} "
                      f"({h_abs:02d}:00) — filling with NaN")
                voltages_24h[slot] = np.nan
                loading_24h[slot]  = np.nan

            ev_kw_24h[slot] = ev_profile_kw[slot]

        return voltages_24h, loading_24h, ev_kw_24h

    if strategy == 'G2V':
        ev_total_kw = simulate_g2v(fleet)
        voltages_24h, loading_24h, ev_kw_24h = _run_power_flow(ev_total_kw)
    else:
        # Pass 1: run G2V to get a realistic voltage profile
        ev_g2v_kw = simulate_g2v(fleet.copy())
        voltages_ref, _, _ = _run_power_flow(ev_g2v_kw)
        v_profile = np.nanmean(voltages_ref, axis=1)
        if np.isnan(v_profile).all():
            v_profile = np.ones(24)

        # Pass 2: run V2G with the measured voltage profile
        ev_total_kw = simulate_v2g(fleet.copy(), v_profile, pv_surplus)
        voltages_24h, loading_24h, ev_kw_24h = _run_power_flow(ev_total_kw)

    return {
        'penetration' : penetration,
        'strategy'    : strategy,
        'pv_enabled'  : PV_ENABLED,
        'ev_kw'       : ev_kw_24h,           # (24,)
        'voltages'    : voltages_24h,         # (24, 33)
        'line_loading': loading_24h,          # (24, 37)
    }


def compute_metrics(result: dict) -> dict:
    """
    Compute scalar summary metrics from a scenario result.

    Returns a flat dict suitable for a summary DataFrame row.
    """
    V   = result['voltages']                   # (24, 33)
    ev  = result['ev_kw']                      # (24,)

    # Mean absolute voltage deviation from 1.0 pu (all buses, all hours)
    volt_dev_mean = float(np.nanmean(np.abs(V - 1.0)))

    # Minimum voltage across all buses and hours
    volt_min = float(np.nanmin(V))

    # Number of (slot, bus) pairs with voltage outside [0.95, 1.05]
    violations = int(np.sum((V < 0.95) | (V > 1.05)))

    # EV peak load [kW]
    ev_peak = float(ev.max())

    # EV load factor: avg / peak (higher = flatter = better for G2V)
    ev_load_factor = float(ev.mean() / ev_peak) if ev_peak > 0 else 0.0

    return {
        'penetration'    : result['penetration'],
        'strategy'       : result['strategy'],
        'pv_enabled'     : result['pv_enabled'],
        'volt_dev_mean'  : volt_dev_mean,
        'volt_min_pu'    : volt_min,
        'volt_violations': violations,
        'ev_peak_kw'     : ev_peak,
        'ev_load_factor' : ev_load_factor,
    }


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print(f"  V2G Smart Grid Simulation")
    print(f"  Date: {SIM_DATE}  |  PV: {'ON' if PV_ENABLED else 'OFF'}")
    print("=" * 60)

    # Generate fleet once — shared across all scenarios
    fleet_full = generate_fleet(seed=42)

    # PV profile aligned to simulation time axis
    pv_raw = get_pv_profile(date=SIM_DATE)
    pv_sim = align_to_sim_start(pv_raw, sim_start_h=SIM_START_H)

    all_results = []
    all_metrics = []

    total = len(EV_PENETRATIONS) * len(STRATEGIES)
    done  = 0

    for pen in EV_PENETRATIONS:
        for strat in STRATEGIES:
            done += 1
            label = f"EV={int(pen*100)}%  strategy={strat}  PV={'ON' if PV_ENABLED else 'OFF'}"
            print(f"\n[{done}/{total}] Running: {label} ...")

            result  = run_scenario(pen, strat, fleet_full, pv_sim)
            metrics = compute_metrics(result)

            all_results.append(result)
            all_metrics.append(metrics)

            print(f"       volt_dev={metrics['volt_dev_mean']:.4f} pu  "
                  f"violations={metrics['volt_violations']}  "
                  f"ev_peak={metrics['ev_peak_kw']:.0f} kW")

    # ── Save metrics summary ──────────────────────────────────────────────────
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.to_csv("results/metrics_summary_2500kW_min40soc.csv", index=False)
    print("\n\nMetrics saved → results/metrics_summary_2500kW_min40soc.csv")
    print(df_metrics.to_string(index=False))

    # ── Save per-scenario voltage timeseries ──────────────────────────────────
    for r in all_results:
           fname = (f"results/voltages_"
               f"EV{int(r['penetration']*100)}_"
               f"{r['strategy']}_"
               f"PV{'ON' if r['pv_enabled'] else 'OFF'}_2500kW_min40soc.csv")
           df_v = pd.DataFrame(
            r['voltages'],
            columns=[f'bus_{i}' for i in range(33)]
        )
           df_v.insert(0, 'slot', range(24))
           df_v.insert(1, 'hour', [(SIM_START_H + s) % 24 for s in range(24)]) 
           df_v.to_csv(fname, index=False)

    print("\nVoltage CSVs saved → results/")

    # ── Generate plots ────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plots.plot_all(all_results, pv_sim, _BASE_LOAD_TOTAL_MW, SIM_START_H)
    print("Done. Check results/ for all output files.")