# example03_microwave.jl
# Calculate the distribution function with microwave read power
using Printf
using NLsolve,LineSearches
using LinearAlgebra
using Plots
using LsqFit

include("module_qpnoneq.jl")

import .Qpnoneq

k = Qpnoneq.k
h = Qpnoneq.h

### Prepare system
T = 0.3		# set temperature of superconducting film
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

Ptarget_pW = 1 	# pW
V = 1000.0		# um^3
Ptarget = Ptarget_pW / (V*Qpnoneq.pJ_per_ueV)

# Scale B so that it gives the correct microwave absorption for low powers
# At low powers there is no effect on f from the microwave power
Iqp0,_ = Qpnoneq.Iqp(fthermal,delta0,de,ihf)
Bscale = Ptarget / Qpnoneq.power_integral(Iqp0,Es,al.N0)

logspace(xmin,xmax,n) = exp.(range(log(xmin),stop=log(xmax),length=n))
npower = 10
power_in = logspace(0.0001,100.,npower)
Bs = Bscale .* power_in

initial_f = Qpnoneq.fermi_dirac.(Es,T)
function calcf(B)
	global initial_f
	function dfdt(f)
		df,Jf,Jn = Qpnoneq.dfEdt(f,nthermal,delta0,de,al.tau0,al.Tc)
		Iqp,JIqpf = Qpnoneq.Iqp(f,delta0,de,ihf)
		df += B.*Iqp
		Jf .+= B.*JIqpf
		df,Jf
	end

	### Solve system

	# Solve for the equilibrium f: df/dt (f) = 0
	result = nlsolve(only_fj(dfdt),initial_f,
				method=:newton,linesearch=MoreThuente(),show_trace=true)
	fsol = result.zero
	initial_f = fsol
	result.zero = [NaN] # Wipe out the arrays in the result object for printing
	result.initial_x = [NaN]
	@show result
	Iqp,_ = Qpnoneq.Iqp(fsol,delta0,de,ihf)
	power = B * Qpnoneq.power_integral(Iqp,Es,al.N0) * (V*Qpnoneq.pJ_per_ueV)
end

powers = zeros(npower)
for i in 1:npower
	powers[i] = calcf(Bs[i])
end

@. function Pmodel(P,Ps)
	log10(P / sqrt(1.0 + P/Ps))
end

p0 = [0.001]
result = curve_fit(Pmodel,power_in,log10.(powers),p0)
popt = coef(result)

pfine = logspace(minimum(power_in),maximum(power_in),1000)
Pfit = Pmodel(pfine,popt)

plot(power_in,powers,xscale=:log10,yscale=:log10,marker=:o,label="sim")
plot!(pfine,10 .^ Pfit,label="fit")
xlabel!("Microwave power (pW)")
ylabel!("Absorbed power (pW)")
