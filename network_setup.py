"""Load and configure the IEEE 33-bus network."""

"""
network_setup.py
----------------
Loads the IEEE 33-bus radial distribution network (pandapower built-in),
distributes EV load and optional PV generation across all load buses (2-33),
and exposes helper functions used by the main simulation loop.

Toggle PV on/off globally:
    PV_ENABLED = True   → PV sgens are added to the network
    PV_ENABLED = False  → pure load network (no generation except slack)
"""

import numpy as np
import pandapower as pp
import pandapower.networks as pn

# ── Global toggle ─────────────────────────────────────────────────────────────
PV_ENABLED = True       # ← change to False to disable all PV generation

# ── PV sizing ─────────────────────────────────────────────────────────────────
PV_TOTAL_KW   = 500.0   # total PV capacity installed across the network [kW]
# PV is distributed proportionally to the bus base load (bigger loads → more roof)

# ── Bus definitions ───────────────────────────────────────────────────────────
# IEEE 33-bus: bus 0 is the slack (substation), buses 1-32 are load buses.
# pandapower uses 0-indexed bus IDs internally.
SLACK_BUS  = 0
LOAD_BUSES = list(range(1, 33))   # buses 1..32  (= nodes 2..33 in literature)
N_LOAD_BUSES = len(LOAD_BUSES)    # 32 buses


def build_network(pv_enabled: bool = PV_ENABLED) -> pp.pandapowerNet:
    """
    Build and return the base IEEE 33-bus network.

    Steps:
      1. Load case33bw (standard radial feeder, 12.66 kV)
      2. Optionally add PV sgens distributed proportionally to base load
      3. Run a baseline power flow to verify convergence
      4. Return the net object (loads NOT yet modified for EV or time-step)
    """
    net = pn.case33bw()
    net.line['max_i_ka'] = 0.4   # 400 A thermal limit for all lines (IEEE 33-bus is mostly 200-300 A)
    if pv_enabled:
        _add_pv_generators(net)

    # Baseline power flow check
    pp.runpp(net, algorithm='bfsw', verbose=False)
    assert net.converged, "Baseline power flow did not converge!"

    return net


def _add_pv_generators(net: pp.pandapowerNet) -> None:
    """
    Add static generators (sgen) to every load bus, sized proportionally
    to each bus's base active load.

    Using pp.create_sgen with p_mw=0 as placeholder — actual output is
    updated each time step by update_network().
    """
    base_loads = net.load['p_mw'].values          # shape (32,)
    total_base = base_loads.sum()

    for idx, bus_idx in enumerate(LOAD_BUSES):
        # Proportional share of total PV capacity [kW → MW]
        pv_share_mw = (base_loads[idx] / total_base) * PV_TOTAL_KW / 1000.0

        pp.create_sgen(
            net,
            bus=bus_idx,
            p_mw=0.0,           # set to 0 — will be updated each time step
            q_mvar=0.0,
            name=f"PV_bus{bus_idx + 1}",
            type="PV",
            max_p_mw=pv_share_mw,
            controllable=False
        )


def update_network(net: pp.pandapowerNet,
                   base_load_mw: np.ndarray,
                   ev_load_kw: np.ndarray,
                   pv_output_kw: np.ndarray,
                   pv_enabled: bool = PV_ENABLED) -> None:
    """
    Update all load and sgen values for a single time step, then run power flow.

    Parameters
    ----------
    net           : pandapowerNet object (modified in place)
    base_load_mw  : shape (32,) base residential/commercial load [MW]
    ev_load_kw    : shape (32,) EV fleet load per bus [kW]  (+ = draw, - = inject)
    pv_output_kw  : shape (32,) PV generation per bus [kW]
    pv_enabled    : if False, PV sgens are zeroed regardless of pv_output_kw
    """
    for idx in range(N_LOAD_BUSES):
        total_load_mw = base_load_mw[idx] + ev_load_kw[idx] / 1000.0
        # EV injection (negative ev_load_kw) reduces net load
        net.load.at[idx, 'p_mw'] = max(total_load_mw, 0.0)

    if pv_enabled and len(net.sgen) > 0:
        for idx in range(N_LOAD_BUSES):
            net.sgen.at[idx, 'p_mw'] = pv_output_kw[idx] / 1000.0

    pp.runpp(net, algorithm='bfsw', verbose=False)


def get_bus_voltages(net: pp.pandapowerNet) -> np.ndarray:
    """Return array (33,) of bus voltages in p.u. after last power flow."""
    return net.res_bus['vm_pu'].values


def get_line_loading(net: pp.pandapowerNet) -> np.ndarray:
    """Return array (32,) of line loading percentages after last power flow."""
    return net.res_line['loading_percent'].values


def distribute_ev_load(total_ev_kw: float) -> np.ndarray:
    """
    Split total EV fleet power across the 32 load buses, proportional
    to each bus's base load (busier feeders host more EVs).

    Returns array (32,) in kW.
    """
    net = pn.case33bw()
    base_loads = net.load['p_mw'].values
    weights = base_loads / base_loads.sum()
    return weights * total_ev_kw


def distribute_pv_output(pv_profile_kw: float) -> np.ndarray:
    """
    Split a scalar PV output value across 32 buses proportional to
    installed capacity (which is proportional to base load).

    Returns array (32,) in kW.
    """
    net = pn.case33bw()
    base_loads = net.load['p_mw'].values
    weights = base_loads / base_loads.sum()
    return weights * pv_profile_kw


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print(f"  PV_ENABLED = {PV_ENABLED}")
    print("=" * 55)

    net = build_network()

    voltages = get_bus_voltages(net)
    loading  = get_line_loading(net)

    print(f"\nBaseline converged: {net.converged}")
    print(f"Buses            : {len(net.bus)}")
    print(f"Lines            : {len(net.line)}")
    print(f"Loads            : {len(net.load)}")
    print(f"Sgens (PV)       : {len(net.sgen)}")

    print(f"\nVoltage range    : {voltages.min():.4f} – {voltages.max():.4f} pu")
    print(f"Line loading max : {loading.max():.2f} %")

    print(f"\nWorst bus (lowest V): bus {voltages.argmin()} → {voltages.min():.4f} pu")

    print("\nVoltage profile (all 33 buses):")
    for i, v in enumerate(voltages):
        flag = " ← VIOLATION" if v < 0.95 or v > 1.05 else ""
        bar  = "█" * int((v - 0.90) / 0.002)
        print(f"  Bus {i:2d}  {v:.4f} pu  {bar}{flag}")