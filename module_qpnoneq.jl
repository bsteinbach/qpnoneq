


module Qpnoneq

using Optim
using Roots
using Parameters
using QuadGK

"Boltzmann constant, ueV/K"
k = 8.617333262e1
"Planck constant, ueV s"
h = 4.135667696e-9
"pJ/ueV"
pJ_per_ueV = 1.60217662e-13

"Fermi-Dirac distribution"
fermi_dirac(E,T,mu=0.0) = 1.0/(exp((E-mu)/(k*T)) + 1.0)
"Bose-einstein distribution"
bose_einstein(E,T,mu=0.0) = 1.0/(exp((E-mu)/(k*T)) - 1.0)

"Trapezoid sum for a 1D array"
trapz(y) = 0.5*(y[1]+y[end]) + sum(y[2:end-1])

"Gap integral for f=0"
f0gapintegral(delta,endpoint) = acoth(endpoint/sqrt(endpoint^2 - delta^2))

"""Gap integral for discretized f
use an analytical formula for delta to delta + de assuming f is constant at f(delta)
use the trapezoid rule for delta+de to delta + (ne-1)*de
use the analytical formula assuming f=0 from (ne-1)*de to Edebye"""
function fgapintegral(f,delta,de,ne,Edebye)
	u = 1.0 .- 2.0.*f
	Es = get_Es(delta,de,ne)

	zeroterm = u[1] * f0gapintegral(delta,Es[2])	# analytical from delta to Es[2]
	trapterms = de .* u[2:end] ./ sqrt.(Es[2:end].^2 .- delta^2)	# trapezoid from Es[2] to Es[end]
	rest = f0gapintegral(delta,Edebye) - f0gapintegral(delta,Es[end])	# analytical from Es[end] to Edebye
	total = zeroterm + trapz(trapterms) + rest
end

"""integrate total quasiparticle density
Assume that f[i] is constant between energy Es[i] and Es[i+1]
"""
function nqp_integral(f,Es,N0)
	de = Es[2]-Es[1]
	delta = Es[1]
	# analytical integral of density of states for constant f
	weight = sqrt.((Es .+ de).^2 .- delta^2) .- sqrt.(Es.^2 .- delta^2)
	4*N0*sum(f.*weight)
end

""" numerically integrate quasiparticle density for a Fermi-Dirac distribution """
function nqp_thermal(N0,delta,T,mu)
	# Use a change of variables to eliminate the density of states singularity
	integrand(eps) = fermi_dirac(sqrt(eps^2 + delta^2),T,mu)
	integral,err = quadgk(integrand,0,Inf)
	4 * N0 * integral
end

""" integral of power in a df/dt term assuming df/dt E is constant over a bin """
function power_integral(dfdt,Es,N0)
	nqp_integral(dfdt .* Es,Es,N0)
end

get_Es(delta,de,ne::Int) = delta .+ (0:(ne-1)).*de
get_Oms(de,ne::Int) = (1:ne).*de

"Round delta to an integer multiple of de"
round_de(x,de) = de * round(x / de)

"Solve for the gap given a the distribution function f"
function solve_gap(f,delta0,de,ne,Edebye,N0Vbcs)
	resid(delta) = 1.0/N0Vbcs - fgapintegral(f,delta,de,ne,Edebye)
	delta = fzero(resid,delta0)
end

"Calculate N0Vbcs that gives a consistent gap with discretized zero f"
function solve_N0Vbcs(delta,de,ne,Edebye)
	f = zeros(ne)
	N0Vbcs = 1.0 / fgapintegral(f,delta,de,ne,Edebye)
end

softening = 0.001im
function rhosoft(E,delta)
	deltac = delta + softening * delta
	real(E/sqrt(E^2 - deltac^2))
end

function rhosoft2(E,delta)
	deltac = delta + softening * delta
	real(E/sqrt(deltac^2 - E^2))
end

function sigma1(f,delta,de,ihnu)
	ne = length(f)
	hf = ihnu * de
	integral = 0.0
	for i in 1:ne
		E = delta + (i-1)*de
		if i + ihnu <= ne
			term = (2/hf)*(1 + delta^2/(E*(E+hf)))
			term *= rhosoft(E,delta)*rhosoft(E+hf,delta)
			integral += term*(f[i] - f[i+ihnu])
		end
	end
	return integral
end

function sigma2(f,delta,de,ihnu)
	ne = length(f)
	hf = ihnu * de
	integral = 0.0
	for i in 1:ihnu
		E = delta + (i-ihnu-1)*de
		if i + ihnu <= ne
			term = (1/hf)*(1 + delta^2/(E*(E+hf)))
			term *=  rhosoft2(E,delta)*rhosoft2(E+hf,delta)
			integral += term*(1-2*f[i])
		end
	end
	return integral
end

"Calculate effect on quasiparticles from a single frequency electromagnetic source"
function Iqp(f::Array{Float64,1},delta::Float64,de::Float64,ihf::Int)
	ne = length(f)
	Es = get_Es(delta,de,ne)
	rhotable = rhosoft.(Es,delta)
	igap2 = Int(round(2*delta/de))
	hf = de*ihf
	K = zeros(ne)
	dKdf = zeros(ne,ne)
	for i in 1:ne
		E = Es[i]

		j = i + ihf	# qp E, photon hf <-> qp E+hf
		if j <= ne
			dosM = 2 * rhotable[j]*(1 + delta^2/(E*(E+hf)))
			K[i] += dosM*(f[j]-f[i])
			dKdf[i,i] -= dosM
			dKdf[i,j] += dosM
		end

		j = i - ihf # qp E-hf, photon hf <-> qp E
		if 1 <= j
			dosM = -2*rhotable[j]*(1 + delta^2/(E*(E-hf)))
			K[i] += dosM*(f[i] - f[j])
			dKdf[i,i] += dosM
			dKdf[i,j] -= dosM
		end

		# photon hf <-> qp E, qp hf-E
		j = ihf - i - igap2
		if 1 <= j
			dosM = 2*rhotable[j]*(1 - delta^2/(E*(hf-E)))
			K[i] += dosM*(1-f[i]-f[j])
			dKdf[i,i] -= dosM
			dKdf[i,j] -= dosM
		end
	end
	K,dKdf
end

"Calculate the time evolution of the quasiparticle distribution function df/dt
f: quasiparticle distribution function
n: phonon distribution function
delta should be an integer multiple of de"
function dfEdt(f::Array{Float64,1},n::Array{Float64,1},delta::Float64,de::Float64,tau0::Float64,Tc::Float64)
	ne = length(f)
	Es = get_Es(delta,de,ne)
	Oms = get_Oms(de,ne)
	rhotable = rhosoft.(Es,delta)
	igap2 = Int(round(2*delta/de))
	dfdt = zeros(ne)
	Jf = zeros(ne,ne)
	Jn = zeros(ne,ne)
	for i in 1:ne
		E = Es[i]
		for j in 1:ne
			Om = Oms[j]
			k = i+j	# qp E, phonon Om <-> qp E+Om
			if 1 <= k && k <= ne
				dos = Om^2*rhotable[k]*(1-delta^2/(E*(E+Om)))
				occ = f[i]*(1-f[k])*n[j] - (1-f[i])*f[k]*(n[j]+1)
				dfdt[i] += dos*occ
				Jf[i,i] += dos*(n[j] + f[k])
				Jf[i,k] += dos*(-1 + f[i] - n[j])
				Jn[i,j] += dos*(f[i] - f[k])
			end
			k = i-j		# qp E <-> qp E-Om, phonon Om
			if 1 <= k && k <= ne
				dos = Om^2*rhotable[k]*(1-delta^2/(E*(E-Om)))
				occ = f[i]*(1-f[k])*(n[j]+1) - (1-f[i])*f[k]*n[j]
				dfdt[i] += dos*occ
				Jf[i,i] += dos*(1-f[k] + n[j])
				Jf[i,k] += dos*(-f[i] - n[j])
				Jn[i,j] += dos*(f[i] - f[k])
			end
			k = j - i - igap2 + 2	# qp E/i, qp Om-E/k <-> phonon Om/j
			if 1 <= k && k <= ne
				dos = Om^2*rhotable[k]*(1+delta*delta/(E*(Om-E)))
				occ = f[i]*f[k]*(n[j]+1) - (1-f[i])*(1-f[k])*n[j]
				dfdt[i] += dos*occ
				Jf[i,i] += dos*(f[k] + n[j])
				Jf[i,k] += dos*(f[i] + n[j])
				Jn[i,j] += dos*(-1 + f[i] + f[k])
			end
		end
	end
	scale = -de/(tau0*(k*Tc)^3)
	dfdt .*= scale
	Jf .*= scale
	return dfdt,Jf,Jn
end


"Calculate effect of quasiparticles on phonons
f: quasiparticle distribution function
n: phonon distribution function
delta should be an integer multiple of de"
function dNOmdt(n::Array{Float64,1},f::Array{Float64,1},delta::Float64,de::Float64,tau0ph::Float64)
	ne = length(f)
	Es = get_Es(delta,de,ne)
	Oms = get_Oms(de,ne)
	rhotable = rhosoft.(Es,delta)
	igap2 = Int(round(2*delta/de))
	dndt = zeros(ne)
	Jf = zeros(ne,ne)
	Jn = zeros(ne,ne)
	for j in 1:ne
		Om = Oms[j]
		for i in 1:ne
			E = Es[i]
			k = i+j	# qp E, phonon Om <-> qp E+Om
			if 1 <= k && k <= ne
				@assert isapprox(Es[k],E+Om)
				dos = 2*rhotable[i]*rhotable[k]*(1-delta^2/(E*(E+Om)))
				occ = f[i]*(1-f[k])*n[j] - (1-f[i])*f[k]*(n[j]+1)
				dndt[i] += dos*occ
				Jf[j,i] += dos*(f[k] + n[j])
				Jf[j,k] += dos*(-1 + f[i] - n[j])
				Jn[j,j] += dos*(f[i] - f[k])
			end
			k = j - i - igap2 + 2	# qp E/i, qp Om-E/k <-> phonon Om/j
			if 1 <= k && k <= ne
				@assert isapprox(Es[k],Om-E)
				dos = rhotable[i]*rhotable[k]*(1+delta*delta/(E*(Om-E)))
				occ = (1-f[i])*(1-f[k])*n[j] - f[i]*f[k]*(n[j]+1)
				dndt[i] += dos*occ
				Jf[j,i] += dos*(-f[k] - n[j])
				Jf[j,k] += dos*(-f[i] - n[j])
				Jn[j,j] += dos*(1 - f[i] - f[k])
			end
		end
	end
	scale = -de/(pi*tau0ph*delta)
	dndt .*= scale
	Jf .*= scale
	Jn .*= scale
	return dndt,Jn,Jf
end

"Fit a Fermi-Dirac distribution with free T (K) and fixed mu=0"
function fit_FDT(f,Es,Tmin=0.01,Tmax=0.7)
	function resid(T)
		df = f .- fermi_dirac.(Es,T)
		1e6*sum(df.^2)
	end
	result = optimize(resid,Tmin,Tmax)
end

"Fit a Fermi-Dirac distribution with free T (K) and free mu (ueV)"
function fit_FDTmu(f,Es,T0=0.3,mu0=0.0)
	function resid(p)
		T,mu = p[1],p[2]
		df = f .- fermi_dirac.(Es,T,mu)
		1e6*sum(df.^2)
	end
	x0 = [T0,mu0]
	result = optimize(resid,x0)
end

@with_kw struct Superconductor
	delta0::Float64		# ueV
	N0::Float64			# 1/um^3/ueV
	Edebye::Float64		# ueV
	tau0::Float64		# s
	tau0ph::Float64		# s
	Tc::Float64			# K

end

function Aluminum()
	delta0 = 180.0	# ueV
	Tc = delta0 / (1.763*k) # K
	al = Superconductor(delta0 = delta0,
		N0 = 1.74e4,		# 1/um^3/ueV
		Edebye = 428 * k,	# ueV
		tau0 = 438e-9,		# s
		tau0ph = 0.26e-9,	# s
		Tc = Tc)
end

end
