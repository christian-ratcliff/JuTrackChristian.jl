include("src/JuTrack.jl")
using .JuTrack
using Enzyme
using StochasticAD
using Distributions
using DistributionsAD
using Random
using Statistics
using StatsBase
using LinearAlgebra
using Zygote
using ForwardDiff
using GaussianDistributions
using GaussianDistributions: correct, ⊕
using Measurements
using UnPack
using Plots
using LaTeXStrings
using BenchmarkTools



function fodo_ring()
        D1 = DRIFT(name="D1", len=0.34)
        D2 = DRIFT(name="D2", len=0.59)
        QF1 = KQUAD(name="QF1", len=0.32, k1=1.06734, rad=1 )
        QD1 = KQUAD(name="QD1", len=0.32, k1=-1.192160, rad=1  )
        BendingAngle = pi/2
        BD1 = SBEND(name="BD1", len=0.72, angle=BendingAngle, e1=BendingAngle, e2=0.0 , rad=1 )
        BD2 = SBEND(name="BD2", len=0.72, angle=BendingAngle, e1=0.0, e2=BendingAngle, rad=1 )

        FODO = (QF1, D1, QD1, D1, QF1)

        CELL = Vector{AbstractElement}(undef, 8)
        for i in eachindex(FODO)
                CELL[i] = FODO[i]
        end
        CELL[length(FODO)+1] = BD1
        CELL[length(FODO)+2] = D2
        CELL[length(FODO)+3] = BD2
                # CELL[i+172] = M5[i]
        
        ELIST = Vector{AbstractElement}(undef, 2*length(CELL))
        for i in eachindex(CELL)
                ELIST[i] = CELL[i]
                ELIST[i+length(CELL)] = CELL[i]
                #     ELIST[i+2*length(CELL)] = CELL[i]
                #     ELIST[i+3*length(CELL)] = CELL[i]
        end
        return ELIST
end

ring_length = 0.32 + 0.34 + 0.32 + 0.34 + 0.32 + 0.72 + 0.59 + 0.72
N = 1000 # Number of steps
s_length = ring_length/N
n = 100000  # Number of particles

