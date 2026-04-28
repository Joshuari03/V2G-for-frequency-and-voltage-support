import pandapower as pp
import pandapower.networks as pn

net = pn.case33bw()
pp.runpp(net)
print(net.res_bus.vm_pu)  # deve stampare 33 valori di tensione