# V2G Simulation Logic

This file summarizes, in simple terms, how the project works and how EV charging/discharging is controlled.

## Flow overview
- main.py runs multiple scenarios combining EV penetration (20/50/80%) and strategy (G2V or V2G).
- For each scenario:
  - compute the EV power profile in kW (24 hourly slots) with ev_model.py
  - distribute EV and PV power across the 32 load buses
  - run power flow on the IEEE 33-bus network with pandapower
  - save metrics and plots

## EV model (ev_model.py)
- Fleet of 500 EVs with:
  - evening arrival time (about 15:00-23:00)
  - morning departure time (about 06:00-10:00)
  - arrival state of charge (soc_arrival)
- Minimum SOC is fixed at 40% to protect the battery.
- Two connection windows:
  - Home: evening arrival until morning departure
  - Work: 09:00-17:00
- Each EV has daily commute energy split into two legs (morning/evening). SOC is updated on each connection transition, so daytime charging affects night demand.

### G2V (uncontrolled)
- If the EV is connected and SOC is not full, it charges at maximum power P_CHARGE.
- It never discharges to the grid.
- SOC is updated every slot, respecting minimum and maximum SOC.

### V2G (smart)
The logic uses two signals:
- Average grid voltage (profile in p.u.)
- PV surplus (PV production minus base load)

Behavior:
- During work hours (09:00-17:00): charge only if there is PV surplus, otherwise stay idle.
- Outside work hours:
  - If voltage is high (>1.04) and SOC is not full: charge at full power.
  - If voltage is low (<0.96) and SOC is above the safety threshold: discharge to the grid (V2G).
  - If PV surplus exists and SOC is not full: charge proportionally (shared across connected EVs).
  - Otherwise charge slowly at 0.3 * P_CHARGE to smooth the evening peak.
- A deadband (0.005 p.u.) and a per-EV mode memory reduce rapid switching near thresholds.
- SOC is always kept within [SOC_MIN, SOC_MAX].

## PV model (pv_model.py)
- 24-hour PV profile from the clearsky model (pvlib).
- Total PV capacity set to 2500 kW.
- PV surplus is computed as:
  - surplus = max(PV - base_load, 0)
- The PV profile is aligned to the simulation start time (06:00).

## Network and power flow (network_setup.py)
- IEEE 33-bus network (pandapower).
- Base load distributed across the 32 load buses proportionally to nominal loads.
- EV and PV are distributed with the same proportional logic.
- Power flow is solved each hour; bus voltages and line loadings are saved.

## Key metrics (main.py)
- Mean voltage deviation from 1.0 p.u.
- Minimum voltage observed
- Number of violations (outside 0.95-1.05)
- EV peak power
- EV load factor (mean/peak)

## Outputs
- CSV files with metrics and voltage profiles.
- Plots for EV load, voltage profiles, heatmaps, and the average SOC profile.
- All files are saved with suffix _2500kW_min40soc.
