include("src/JuTrack.jl")
using .JuTrack
using Plots
using Random
using Optim
using BenchmarkTools

struct weightsini <: AbstractVector{Float64}
    tracex::Float64
    tracey::Float64
    sumk::Float64
end

struct weights <: AbstractVector{Float64}
    tracex::Float64
    tracey::Float64
    tunex::Float64
    tuney::Float64
    dispx::Float64
    dispy::Float64
    betamaxx::Float64
    betamaxy::Float64
    betavarx::Float64
    betavary::Float64
    sumk::Float64
end

function ring_gen(k::Vector{Float64})
    D1 = DRIFT(name="D1", len=0.7)
    D2 = DRIFT(name="D2", len=0.2)
    QF1 = KQUAD(name="QF1", len=0.6, k1=k[1])
    QF2 = KQUAD(name="QF2", len=0.6, k1=k[2])
    QF3 = KQUAD(name="QF3", len=0.6, k1=k[3])
    QF4 = KQUAD(name="QF4", len=0.6, k1=k[4])
    QF5 = KQUAD(name="QF5", len=0.6, k1=k[5])
    QF6 = KQUAD(name="QF6", len=0.6, k1=k[6])
    QF7 = KQUAD(name="QF7", len=0.6, k1=k[7])
    QF8 = KQUAD(name="QF8", len=0.6, k1=k[8])
    QD1 = KQUAD(name="QD1", len=0.6, k1=k[9])
    QD2 = KQUAD(name="QD2", len=0.6, k1=k[10])
    QD3 = KQUAD(name="QD3", len=0.6, k1=k[11])
    QD4 = KQUAD(name="QD4", len=0.6, k1=k[12])
    BendingAngle = pi/2.
    BD1 = SBEND(name="BD1", len=2.1, angle=BendingAngle )
    ringout = [QF1, D1, QD1, D1, QF2, D2, BD1, D2,
            QF3, D1, QD2, D1, QF4, D2, BD1, D2,
            QF5, D1, QD3, D1, QF6, D2, BD1, D2,
            QF7, D1, QD4, D1, QF8, D2, BD1, D2]
    return ringout
end

# function ring_gen(k::Vector{Float64})
#     D1 = DRIFT(name="D1", len=0.2)
#     QD1 = KQUAD(name="QD1", len=0.3, k1=k[1])
#     QD2 = KQUAD(name="QD2", len=0.3, k1=k[2])
#     QD3 = KQUAD(name="QD3", len=0.3, k1=k[3])
#     QD4 = KQUAD(name="QD4", len=0.3, k1=k[4])
#     BendingAngle = pi/2.
#     BD1 = SBEND(name="BD1", len=1.7, angle=BendingAngle )
#     # ringout = [QF1, D1, QD1, D1, QF2, BD1,
#     #         QF3, D1, QD2, D1, QF4, BD1,
#     #         QF5, D1, QD3, D1, QF6, BD1,
#     #         QF7, D1, QD4, D1, QF8, BD1]
#     ringout = [D1, QD1, D1, BD1,
#                 D1, QD2, D1, BD1,
#                 D1, QD3, D1, BD1,
#                 D1, QD4, D1, BD1]
#     return ringout
# end

function get_changed_idx(RING)
    changed_idx = Int64[]
    for i in eachindex(RING)
        if RING[i].name[1] == 'Q'
            push!(changed_idx, i)
        end
    end
    return changed_idx
end

function get_changed_elements(RING,k)
    changed_idx = get_changed_idx(RING)
    changed_ele = KQUAD[]
    num_quads = length(changed_idx)
    idx = 1
    for i in changed_idx
        if RING[i].name[1:2] == "QF"
            push!(changed_ele, KQUAD(len=RING[i].len, k1=k[idx]))
        elseif RING[i].name[1:2] == "QD"
            push!(changed_ele, KQUAD(len=RING[i].len, k1=-k[idx]))
        end
        idx += 1
    end
    return changed_ele
end

function kopt(θ::Vector{Float64}, k::Vector{Float64}, b::Vector{Float64})
    return @. θ*k + b
end

function get_tune(k::Vector{Float64}, RING::Vector{AbstractElement}, dimension::String, changed_idx::Vector{Int64})
    changed_ele = get_changed_elements(RING, k)
    refpts = [i for i in 1:length(RING)]
    twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
    if dimension == "x"
        phase = twi[end].mux - twi[1].mux
    elseif dimension == "y"
        phase = twi[end].muy - twi[1].muy
    else
        error("Invalid dimension")
    end
    tune = phase/(2*π)
    return tune
end

function get_trace(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String)
    changed_ele = get_changed_elements(RING, k)
    m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
    if dimension == "x"
        trace = m66[1, 1] + m66[2, 2]
    elseif dimension == "y"
        trace = m66[3, 3] + m66[4, 4]
    else
        error("Invalid dimension")
    end
    return trace
end

ideal_trace(tune::Float64) = cos(tune * 2 * π)*2

function trace_loss(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String, ideal_trace::Float64)
    trace = get_trace(k, RING, changed_idx, dimension)
    return (abs(ideal_trace) - abs(trace))^2
end



function total_loss_ini(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, ideal_tracex::Float64, ideal_tracey::Float64, w::weightsini)
    lossx = w.tracex*trace_loss(k, RING, changed_idx, "x", ideal_tracex)
    lossy = w.tracey*trace_loss(k, RING, changed_idx, "y", ideal_tracey)
    return lossx + lossy + w.sumk*sum(k.^2)
end

function tune_loss(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String, ideal_tune::Float64)
    tune = get_tune(k, RING, dimension, changed_idx)
    return (abs(ideal_tune) - abs(tune))^2
end

function get_beta_var(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String)
    changed_ele = get_changed_elements(RING, k)
    refpts = [i for i in 1:length(RING)]
    twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
    if dimension == "x"
        beta = [twi[i].betax for i in eachindex(twi)]
    elseif dimension == "y"
        beta = [twi[i].betay for i in eachindex(twi)]
    else
        error("Invalid dimension")
    end
    max_beta = maximum(beta)
    min_beta = minimum(beta)
    return (max_beta - min_beta)/(max_beta + min_beta)
end

function get_disp(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String)
    changed_ele = get_changed_elements(RING, k)
    # m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
    # if dimension == "x"
    #     disp = m66[1, 6]
    # elseif dimension == "y"
    #     disp = m66[3, 6]
    # else
    #     error("Invalid dimension")
    # end
    refpts = [i for i in 1:length(RING)]
    twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
    eta = Float64[]
    for i in eachindex(twi)
        if twi[i].name[1] == 'D'
            if dimension == "x"
                push!(eta, twi[i].eta[1])
            elseif dimension == "y"
                push!(eta, twi[i].eta[3])
            else
                error("Invalid dimension")
            end
        end
    end
    return eta
end

function disp_loss(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String)
    eta = get_disp(k, RING, changed_idx, dimension)
    return sum(eta.^2)
end

ideal_beta_max(ideal_tune::Float64, L::Float64) = L * (1+sin(π*ideal_tune))/sin(π*ideal_tune)

function get_beta_max(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, dimension::String)
    changed_ele = get_changed_elements(RING, k)
    refpts = [i for i in 1:length(RING)]
    twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
    if dimension == "x"
        beta = [twi[i].betax for i in eachindex(twi)]
    elseif dimension == "y"
        beta = [twi[i].betay for i in eachindex(twi)]
    else
        error("Invalid dimension")
    end
    return maximum(beta)
end

function beta_max_loss(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, ideal_tune::Float64, L::Float64, dimension::String)
    beta_max = get_beta_max(k, RING, changed_idx, dimension)
    ideal_beta = ideal_beta_max(ideal_tune, L)
    return (ideal_beta - beta_max)^2
end

function total_loss(k::Vector{Float64}, RING::Vector{AbstractElement}, changed_idx::Vector{Int64}, ideal_tracex::Float64, ideal_tracey::Float64, ideal_tunex::Float64, ideal_tuney::Float64, L::Float64, w::weights)
    lossx = w.tracex*trace_loss(k, RING, changed_idx, "x", ideal_tracex) + w.tunex*tune_loss(k, RING, changed_idx, "x", ideal_tunex) + w.dispx*disp_loss(k, RING, changed_idx, "x") - w.betamaxx*beta_max_loss(k, RING, changed_idx, ideal_tunex, L, "x") + w.betavarx*get_beta_var(k, RING, changed_idx, "x")
    lossy = w.tracey*trace_loss(k, RING, changed_idx, "y", ideal_tracey) + w.tuney*tune_loss(k, RING, changed_idx, "y", ideal_tuney) + w.dispy*disp_loss(k, RING, changed_idx, "y") - w.betamaxy*beta_max_loss(k, RING, changed_idx, ideal_tuney, L, "y") + w.betavary*get_beta_var(k, RING, changed_idx, "y")

    return lossx + lossy + w.sumk*sum(k.^2)
end




weightsini(;tracex::Float64=1.0, tracey::Float64=1.0, sumk::Float64=1.0) = weightsini(tracex, tracey, sumk)
winitial = weightsini()

weights(;tracex = 1.0, tracey = 1.0, tunex = 1.0, tuney = 1.0, dispx = 1.0, dispy = 1.0, betamaxx = 1.0, betamaxy = 1.0, betavarx = 1.0, betavary = 1.0, sumk = 1.0) = weights(tracex, tracey, tunex, tuney, dispx, dispy, betamaxx, betamaxy, betavarx, betavary, sumk)
wopt = weights()

L_fodo = 0.6*3 + 0.7*2
kf = [1.0 for i in 1:8]
kd = [-1.0 for i in 1:4]
krand = kopt(1.0 .+ rand(12), [kf; kd], rand(12))

# krand = kopt(1.0 .+ rand(4), kd, -rand(4))


RINGini = ring_gen(krand .+ 0.2)
changed_idx = get_changed_idx(RINGini)
# @btime get_changed_idx(RING)
changed_ele = get_changed_elements(RINGini, krand)
# @btime get_changed_elements(RING, krand)
trace_x = get_trace(krand, RINGini, changed_idx, "x")
# @btime get_trace(krand, RING, changed_idx, "x")
ideal_tunex = 0.58
ideal_tuney = 0.53

ideal_tracex = ideal_trace(ideal_tunex)
ideal_tracey = ideal_trace(ideal_tuney)
# println("ideal tracex: ", ideal_tracex, " ideal tracey: ", ideal_tracey)

res1 = optimize(k -> total_loss_ini(k, RINGini, changed_idx, ideal_tracex, ideal_tracey, winitial), krand, NelderMead(), Optim.Options(g_tot = 1e-5, iterations = 5_000, show_trace = true, show_every = 250))
println(res1.minimizer)
println(res1)

k = res1.minimizer
RING = ring_gen(k)
res2 = optimize(k -> total_loss(k, RING, changed_idx, ideal_tracex, ideal_tracey, ideal_tunex, ideal_tuney, L_fodo, wopt), k, NelderMead(), Optim.Options(g_tot = 1e-5, iterations = 5_000, show_trace = true, show_every = 250))
println(res2)
koptimal = res2.minimizer
println(koptimal)





