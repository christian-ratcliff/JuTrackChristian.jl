using JuTrack
using Enzyme
using Integrals
using Serialization
using Random
using Interpolations, LinearAlgebra
using SpecialFunctions
ESR_crab = deserialize("src/demo/ESR/esr_main_linearquad.jls")

const C_LIGHT = 299792458

function compute_haissinski(RING::Vector{AbstractElement}, particles::Beam, wake,m::Int64, kmax::Float64, current::Float64, numIters::Int64, ϵ_conv::Float64 )
    #m is the number of points in the distribution
    #kmax is the min and max of range of distro, units of σ_z0
    #current is the bunch current
    #numIters is the number of iterations to run
    #ϵ_conv is the convergence criterion
    circum = sum(RING[i].len for i in 1:length(RING))
    energy = particles.energy   
end

# sum(ESR_crab[i].len for i in 1:length(ESR_crab))
#r = (z, δ, s)
r = zeros(Float64, 10000, 3)
rand!(r[1:2])
size(r,1)

mutable struct BeamPipe
    radius::Float64
    σ_c::Float64 #conductivity
end
function BeamPipe(radius::Float64, σ_c::Float64)
    return BeamPipe(radius, σ_c)
end


struct Haissinski
    # Parameters extracted from the ring and wake object
    circumference::Float64
    energy::Float64
    f_s::Float64
    ν_s::Float64
    σ_e::Float64
    σ_z::Float64
    η::Float64
    γ::Float64
    β::Float64
    numIters::Int
    ϵ_conv::Float64
    ds::Float64
    s::Vector{Float64}
    wtot_fun::Interpolation{Float64, 1, Gridded{Float64}, GriddedInterpolation{Float64, 1, Gridded{Float64}, Float64}}
    m::Int
    npoints::Int
    kmax::Float64
    dq::Float64
    q_array::Vector{Float64}
    weights::Vector{Float64}
    dtFlag::Bool
    Ic::Float64
    ϕ::Vector{Float64}
    ϕ_0::Vector{Float64}
    Smat::Matrix{Float64}
    allFi::Vector{Float64}
    pseudo_inv::Vector{Float64}
    res::Vector{Float64}
    conv::Float64

    function Haissinski(RING::Vector{AbstractElement},beam::Beam,  m::Int64, kmax::Float64, current::Float64, numIters::Int64, ϵ_conv::Float64)
        circumference = sum(RING[i].len for i in 1:length(RING))
        energy = beam.energy
        f_s = #IDK where to get this     
        ν_s = f_s / (C_LIGHT / circumference)
        σ_e = σ_e * energy  #energy spread
        σ_z = σ_z #bunch length, I am not sure where I get this, or if this is something that I just set

        η = #IDK where to get this
        γ = beam.gamma
        β = beam.beta
        beampipe = BeamPipe(.01, 5.8e17)
        char_length = (C_LIGHT * beampipe.radius^2 / (2 * beampipe.σ_c * π))^(1/3)
        s = range(0, circumference, circumference/(size(beam.r), 1) ) / σ_z
        ds = s[end] - s[end-1]
        wake = zeros(size(s))
        wake = [(beam.charge / (beampipe.radius^2)) * (char_length / σ_z)^(3/2) * (modified_bessel_composite(s[i] / σ_z))]

        
        wtot_fun = linear_interpolation(s, wake)

        if m % 2 != 0
            throw(ArgumentError("m must be even and int"))
        end

        npoints = 2 * m + 1
        dq = kmax / m
        q_array = -kmax .+ dq .* collect(0:(npoints-1))

        obj = new(circumference, energy, f_s, ν_s, σ_e, σ_z, η, γ, β, numIters, ϵ_conv, ds, s, wtot_fun, m, npoints, kmax, dq, q_array)
        obj.set_weights()
        obj.dtFlag = false
        obj.set_I(current)
        obj.initial_ϕ()
        obj.precompute_S()
        obj.compute_Smat()

        return obj
    end
end

function set_weights(obj::Haissinski)
    npoints = obj.npoints
    weights = ones(npoints)
    weights[2:2:end] .= 4.0
    weights[1] = 1.0
    weights[end] = 1.0
    weights .= obj.dq / 3
    obj.weights = weights
end

function modified_bessel_composite(x::Float64)
    return abs(x)^(3/2) * exp(-x^2 / 4) *(besseli(1/4, x^2/4) - besseli(-3/4, x^2/4) + besseli(-1/4, x^2/4) - besseli(3/4, x^2/4))
end

function precompute_S(obj::Haissinski)
    println("Computing integrated wake potential")
    sr = collect(range(2*minimum(obj.q_array), stop=abs(2*minimum(obj.q_array))+obj.ds, step=obj.ds))
    res = zeros(length(sr))

    topend = trapz(obj.wtot_fun.(collect(range(maximum(sr), stop=maximum(obj.s), step=obj.ds))))
    for p in 1:length(sr)
        if p % 10000 == 0
            println("$p out of $(length(sr))")
        end
        rr = collect(range(sr[p], stop=maximum(sr), step=obj.ds))
        val = trapz(obj.wtot_fun.(rr)) + topend
        res[p] = val
    end
    obj.Sfun_range = sr
    obj.Sfun = linear_interpolation(sr, res)
end

function compute_Smat(obj::Haissinski)
    obj.Smat = zeros(obj.npoints, obj.npoints)
    for iq1 in 1:obj.npoints
        for iq2 in 1:obj.npoints
            obj.Smat[iq1, iq2] = obj.Sfun(obj.q_array[iq1] - obj.q_array[iq2])
        end
    end
end

function set_I(obj::Haissinski, current::Float64)
    N = current * obj.circumference / (C_LIGHT * obj.β * qe)
    obj.Ic = sign(obj.η) * qe * N / (2 * π * obj.ν_s * obj.σ_e)
    println("Normalised current: $(obj.Ic * 1e12) pC/V")
end

function initial_ϕ(obj::Haissinski)
    obj.ϕ = exp.(-obj.q_array.^2 / 2) .* obj.Ic ./ sqrt(2 * π)
    obj.ϕ_0 = copy(obj.ϕ)
end

function Fi(obj::Haissinski)
    obj.allFi = zeros(obj.npoints)
    for i in 1:obj.npoints
        sum1 = 0.0
        for j in 1:obj.npoints
            sum1b = 0.0
            for k in 1:obj.npoints
                sum1b += obj.weights[k] * obj.Smat[j, k] * obj.ϕ[k]
            end
            sum1 += obj.weights[j] * exp(-obj.q_array[j]^2 / 2 + sum1b)
        end
        sum2 = 0.0
        for j in 1:obj.npoints
            sum2 += obj.weights[j] * obj.Smat[i, j] * obj.ϕ[j]
        end
        obj.allFi[i] = obj.ϕ[i] * sum1 - obj.Ic * exp(-obj.q_array[i]^2 / 2 + sum2)
    end
end

function dFi_ij(obj::Haissinski, i::Int, j::Int)
    sum1 = 0.0
    for k in 1:obj.npoints
        sum1b = 0.0
        for li in 1:obj.npoints
            sum1b += obj.weights[li] * obj.Smat[k, li] * obj.ϕ[li]
        end
        kron = i == j ? 1 : 0
        sum1 += obj.weights[k] * (kron + obj.ϕ[i] * obj.weights[j] * obj.Smat[k, j]) * exp(-obj.q_array[k]^2 / 2 + sum1b)
    end

    sum2 = 0.0
    for k in 1:obj.npoints
        sum2 += obj.weights[k] * obj.Smat[i, k] * obj.ϕ[k]
    end

    val = sum1 - obj.Ic * obj.weights[j] * obj.Smat[i, j] * exp(-obj.q_array[i]^2 / 2 + sum2)
    return val
end

function dFi_dϕj(obj::Haissinski)
    if !obj.dtFlag
        t0 = time()
        dFi_ij(obj, 1, 1)
        dt = time() - t0
        println("Computing dFi/dϕj, will take approximately $(round(dt * obj.npoints^2 / 60, digits=3)) minutes per iteration")
        obj.dtFlag = true
    end
    allDat = [dFi_ij(obj, i, j) for i in 1:obj.npoints for j in 1:obj.npoints]
    obj.alldFi_dϕj = reshape(allDat, obj.npoints, obj.npoints)
end
ϕ
function compute_new_ϕ(obj::Haissinski)
    ainv = inv(obj.alldFi_dϕj)
    obj.pseudo_inv = ainv * (-obj.allFi)
    obj.ϕ_1 = obj.pseudo_inv + obj.ϕ
end

function update(obj::Haissinski)
    obj.ϕ = copy(obj.ϕ_1)
end

function convergence(obj::Haissinski)
    obj.conv = sum(abs.(obj.ϕ_1 - obj.ϕ)) / sum(abs.(obj.ϕ))
end

function set_output(obj::Haissinski)
    obj.res = reverse(copy(obj.ϕ_1))
end

function solve(obj::Haissinski)
    for it in 1:obj.numIters
        Fi(obj)
        dFi_dϕj(obj)
        compute_new_ϕ(obj)
        convergence(obj)
        set_output(obj)
        println("Iteration: $it, Delta: $(obj.conv)")
        if obj.conv < obj.ϵ_conv
            println("Converged")
            break
        end
        if it != obj.numIters - 1
            update(obj)
        end
    end
    if obj.conv > obj.ϵ_conv
        println("Did not converge")
    end
end




