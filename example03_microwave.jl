# example03_microwave.jl
# Calculate the distribution function with microwave read power
using Printf
using NLsolve,LineSearches
using LinearAlgebra
using Plots

include("module_qpnoneq.jl")

import .Qpnoneq

k = Qpnoneq.k
h = Qpnoneq.h

### Prepare system
T = 0.25		# set temperature of superconducting film
al = Qpnoneq.Aluminum() # Choose superconducting material
ne = 1000	# Discretize the distribution function in this many bins
de = 1.0	# The bins will be this wide in ueV
delta0 = Qpnoneq.round_de(al.delta0,de)	# Round the gap to the nearest ueV

# Pick a read frequency that's an integer multiple of de
hf = 400e6 * h
hf = Qpnoneq.round_de(hf,de)
freq = hf / h
ihf = Int(round(hf / de))

@show freq
@show ihf

#Prepare a thermal distribution of quasiparticles and phonons
Es = Qpnoneq.get_Es(delta0,de,ne)
Oms = Qpnoneq.get_Oms(de,ne)
fthermal = Qpnoneq.fermi_dirac.(Es,T)
nthermal = Qpnoneq.bose_einstein.(Oms,T)

Ptarget_uW = 1 	# pW
V = 1000.0		# um^3
Ptarget = Ptarget_uW / (V*Qpnoneq.pJ_per_ueV)

# Calculate power absorption for the thermal distribution at low power
Iqp0,_ = Qpnoneq.Iqp(fthermal,delta0,de,ihf)
B = Ptarget / Qpnoneq.power_integral(Iqp0,Es,al.N0)

logspace(xmin,xmax,n) = exp.(range(log(xmin),stop=log(xmax),length=n))
Bs = B*logspace(0.001,100.,20)

# Calculate dfdt with all parameters except f fixed
function dfdt(f)
	df,Jf,Jn = Qpnoneq.dfEdt(f,nthermal,delta0,de,al.tau0,al.Tc)
	Iqp,JIqpf = Qpnoneq.Iqp(f,delta0,de,ihf)
	df += B.*Iqp
	Jf .+= B.*JIqpf
	df,Jf
end

### Solve system

initial_f = Qpnoneq.fermi_dirac.(Es,T)

# Solve for the equilibrium f: df/dt (f) = 0
result = nlsolve(only_fj(dfdt),initial_f,
			method=:newton,linesearch=MoreThuente(),show_trace=true)
fsol = result.zero
result.zero = [NaN] # Wipe out the arrays in the result object for printing
result.initial_x = [NaN]
@show result

### Calculate quasiparticle number density and compare
nqp = Qpnoneq.nqp_integral(fsol,Es,al.N0)

nth_approx = 2 * al.N0 * sqrt(2*pi*k*T*delta0) * exp(-delta0/(k*T))
nth_exact = Qpnoneq.nqp_thermal(al.N0,delta0,T,0.0)
@show nqp,nth_approx,nth_exact

### Fit Fermi-Dirac distributions to the empirical distributions
opt_FDT = Qpnoneq.fit_FDT(fsol,Es).minimizer
opt_FDTmu = Qpnoneq.fit_FDTmu(fsol,Es).minimizer

fFDT = Qpnoneq.fermi_dirac.(Es,opt_FDT)
fFDTmu = Qpnoneq.fermi_dirac.(Es,opt_FDTmu[1],opt_FDTmu[2])

plot(Es,fsol,label="sim")
plot!(Es,fFDT,label=@sprintf("fit T=%.3f K,mu=0 ueV",opt_FDT))
plot!(Es,fFDTmu,label=@sprintf("fit T=%.3f K,mu=%.1f ueV",opt_FDTmu[1],opt_FDTmu[2]))
xlims!(delta0,500)
