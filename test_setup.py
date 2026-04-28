import pandapower as pp
import pandapower.networks as pn

net = pn.case33bw() # loads a standard IEEE test case for a 33-bus distribution network studying DER (Distributed Energy Resources) integration
pp.runpp(net) # executes the power flow to calculate voltages and currents in the network (baseline without EVs and PVs)
print(net.res_bus.vm_pu)  # prints the voltages in per unit for each bus in the network