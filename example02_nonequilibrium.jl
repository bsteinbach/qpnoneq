# example02_nonequilibrium.jl
# Look at what happens when we have excess quasiparticles

include("module_qpnoneq.jl")
import .Qpnoneq

using Printf
using NLsolve,LineSearches
using LinearAlgebra
using Plots
using LsqFit

### Prepare system
k = Qpnoneq.k
T = 0.07		# set temperature of bath
al = Qpnoneq.Aluminum() # Choose superconducting material
ne = 1000	# Discretize the distribution function in this many bins
de = 1.0	# The bins will be this wide in ueV
delta0 = Qpnoneq.round_delta(al.delta0,de)	# Round the gap to the nearest ueV

#Prepare a thermal distribution of quasiparticles and phonons
Es = Qpnoneq.get_Es(delta0,de,ne)
Oms = Qpnoneq.get_Oms(de,ne)
fthermal = Qpnoneq.fermi_dirac.(Es,T)
nthermal = Qpnoneq.bose_einstein.(Oms,T)

dfdt_injected = ones(ne)
dfdt_power = Qpnoneq.power_integral(dfdt_injected,Es,al.N0)
V = 1000 # um^3
# dfdt_injected / pW
powercal = dfdt_power * V * Qpnoneq.pJ_per_ueV

function logspace(x0,x1,n)
	exp.(range(log(x0),stop=log(x1),length=n))
end

npower = 10
powers = logspace(0.001,100,npower)
dfdt_injecteds = powers .* powercal
powers ./= V * Qpnoneq.pJ_per_ueV

initial_f = Qpnoneq.fermi_dirac.(Es,0.2)
function eval_power(dfdt_injected)
	global initial_f
	# Calculate dfdt with all parameters except f fixed
	function dfdt(f)
		df,Jf,Jn = Qpnoneq.dfEdt(f,nthermal,delta0,de,al.tau0,al.Tc)
		# Add in a forcing term creating quasiparticles
		df .+= dfdt_injected
		df,Jf
	end

	### Solve system

	# Solve for the equilibrium f: df/dt (f) = 0
	result = nlsolve(only_fj(dfdt),initial_f,
				method=:newton,linesearch=MoreThuente(),show_trace=true)
	fsol = result.zero
	initial_f = fsol

	nqp = Qpnoneq.nqp_integral(fsol,Es,al.N0)
	### Calculate quasiparticle time constant for solution
	# Use J = d(df/dt)/df to measure the linear dynamics about the nonlinear solution
	err,J = dfdt(fsol)
	# Estimate the quasiparticle time constant as the slowest decaying eigenmode of J
	w = eigmin(-J)
	#tauqp = -1.0 / w[end]
	tauqp = 1.0 / w
	@printf "tau_{qp}: %.1f us" (1e6*tauqp)

	### Fit Fermi-Dirac distributions to the empirical distributions
	Tfit = Qpnoneq.fit_FDT(fsol,Es).minimizer
	Tfit,nqp,tauqp
end

Ts = zeros(npower)
nqps = zeros(npower)
taus = zeros(npower)

for i in 1:npower
	Ts[i],nqps[i],taus[i] = eval_power(dfdt_injecteds[i])
end


taus = [0.0004059150351426761, 0.00022207004533978072, 0.00011947535526335139, 6.368632423052234e-5, 3.3776375949925416e-5, 1.7862951303850448e-5, 9.43095687480012e-6, 4.973048737206256e-6, 2.619160124971449e-6, 1.3772379744619589e-6]
nqps = [204.1319053042584, 396.5630642372121, 769.2095750845632, 1489.0974757223207, 2876.0029528381106, 5539.652968722637, 10637.073469882536, 20351.549677905423, 38776.48042611041, 73533.89995211695]
Ts = [0.2219888970735318, 0.23694523596020353, 0.2538470287782349, 0.27306462868449677, 0.2950776490023831, 0.320512112945843, 0.3501967001847941, 0.3852493504408286, 0.4272153549171989, 0.4782984284030481]
powers = [0.0010000000000000002, 0.0035938136638046297, 0.012915496650148841, 0.046415888336127795, 0.16681005372000593, 0.5994842503189413, 2.1544346900318847, 7.742636826811275, 27.825594022071257, 100.00000000000004]

plot(powers,nqps.*taus,xscale=:log10,marker='o')
xlabel!("P (ueV/um^3/s)")
plot(powers,(nqps.*delta0 ./ taus),xscale=:log10,marker='o')

plot(Ts,powers,yscale=:log10,marker='o')
ylabel!("Power (pW)")
xlabel!("T (K)")

# tau ~ 1/nqp
# P = nqp delta / tau ~ nqp^2
# nqp ~ sqrt(T) exp(-delta/(k T))
# P ~ T exp(-2 delta/(k T))

@. ym(T) = T*exp(-2*delta0/(k*T))

c = powers[end] ./ ym(Ts[end])
plot!(Ts,c.*ym(Ts))
