include("../src/JuTrack.jl")
include("make_threading_tests.jl")
using .JuTrack
using Enzyme
using Test
using BenchmarkTools
using .threading_tests


singleparticle_drift_tests()
# multiparticle_drift_tests()


# DRIFT SINGLE
#   887.795 ns (31 allocations: 3.86 KiB)
#   110.647 μs (352 allocations: 40.42 KiB)
#   3.002 μs (68 allocations: 7.88 KiB)
#   124.885 μs (389 allocations: 49.44 KiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# DRIFT MULTI
#   3.088 ms (100030 allocations: 11.90 MiB)
#   1.341 ms (100351 allocations: 11.94 MiB)
#   19.537 ms (270058 allocations: 25.87 MiB)
#   2.677 ms (270379 allocations: 25.91 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true


# const particles_initial::Array{Float64, 2} = fill(0.1, (10,6))

# const l::Float64 =1.23

# function create_drift(l)
#     local dr::DRIFT = DRIFT(len=l)
#     return dr
# end

# function drift_track(l)
#     local particles::Array{Float64, 2}  = fill(0.1, (10,6))
#     beam = Beam(r=particles)
#     line = [create_drift(l)]
#     linepass!(line, beam)
#     return beam.r
# end

# function drift_track_mthread(l)
#     local particles::Array{Float64, 2}  = fill(0.1, (10,6))
#     beam = Beam(r=particles)
#     line = [create_drift(l)]
#     plinepass!(line, beam)
#     return beam.r
# end

# @btime drift_track(l)
# @btime drift_track_mthread(l)
# grad1 = autodiff(Forward, drift_track, DuplicatedNoNeed, Duplicated(l, 1.0))
# grad2 = autodiff(Forward, drift_track_mthread, DuplicatedNoNeed, Duplicated(l, 1.0))
# @btime autodiff(Forward, drift_track, DuplicatedNoNeed, Duplicated(l, 1.0))
# @btime autodiff(Forward, drift_track_mthread, DuplicatedNoNeed, Duplicated(l, 1.0))
# println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)

# println(particles_intital)
# particles = @views particles_initial[:,:]
# particles = @view particles_initial[:,:,:]
# println(particles)
# 