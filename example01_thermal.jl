# example01_thermal.jl
# Check that the fermi-dirac distribution is the equilibrium solution for quasiparticles
using Printf
using NLsolve,LineSearches
using LinearAlgebra
using Plots

include("module_qpnoneq.jl")

import .Qpnoneq

k = Qpnoneq.k

### Prepare system
T = 0.3		# set temperature of superconducting film
al = Qpnoneq.Aluminum() # Choose superconducting material
ne = 1000	# Discretize the distribution function in this many bins
de = 1.0	# The bins will be this wide in ueV
delta0 = Qpnoneq.round_de(al.delta0,de)	# Round the gap to the nearest ueV

#Prepare a thermal distribution of quasiparticles and phonons
Es = Qpnoneq.get_Es(delta0,de,ne)
Oms = Qpnoneq.get_Oms(de,ne)
fthermal = Qpnoneq.fermi_dirac.(Es,T)
nthermal = Qpnoneq.bose_einstein.(Oms,T)


# Calculate dfdt with all parameters except f fixed
function dfdt(f)
	df,Jf,Jn = Qpnoneq.dfEdt(f,nthermal,delta0,de,al.tau0,al.Tc)
	df,Jf
end

### Solve system

initial_f = Qpnoneq.fermi_dirac.(Es,0.07)
# Solve for the equilibrium f: df/dt (f) = 0
result = nlsolve(only_fj(dfdt),initial_f,
			method=:newton,linesearch=MoreThuente(),show_trace=true)
fsol = result.zero
result.zero = [NaN] # Wipe out the arrays in the result object for printing
result.initial_x = [NaN]
@show result

### Calculate quasiparticle number density and compare to analytical results
nqp = Qpnoneq.nqp_integral(fsol,Es,al.N0)

nth_approx = 2 * al.N0 * sqrt(2*pi*k*T*delta0) * exp(-delta0/(k*T))
nth_exact = Qpnoneq.nqp_thermal(al.N0,delta0,T,0.0)
@show nqp,nth_approx,nth_exact

### Calculate quasiparticle time constant for solution
# Use J = d(df/dt)/df to measure the linear dynamics about the nonlinear solution
err,J = dfdt(fsol)
# Estimate the quasiparticle time constant as the slowest decaying eigenmode of J
w = eigmin(-J)
#tauqp = -1.0 / w[end]
tauqp = 1.0 / w
@printf "tau_{qp}: %.1f us" (1e6*tauqp)

### Fit Fermi-Dirac distributions to the empirical distribution
opt_FDT = Qpnoneq.fit_FDT(fsol,Es).minimizer
opt_FDTmu = Qpnoneq.fit_FDTmu(fsol,Es).minimizer

fFDT = Qpnoneq.fermi_dirac.(Es,opt_FDT)
fFDTmu = Qpnoneq.fermi_dirac.(Es,opt_FDTmu[1],opt_FDTmu[2])

plot(Es,fsol,label="sim")
plot!(Es,fFDT,label=@sprintf("fit T=%.3f K,mu=0 ueV",opt_FDT))
plot!(Es,fFDTmu,label=@sprintf("fit T=%.3f K,mu=%.1f ueV",opt_FDTmu[1],opt_FDTmu[2]))
xlims!(delta0,500)
