include("../src/JuTrack.jl")
include("make_threading_tests.jl")
using .JuTrack
using Enzyme
using Test
using BenchmarkTools
using .threading_tests


const bend_angle_in = pi/2
const hkick_in = 0.02
const vkick_in = 0.03
const l_in = 1.23
const k1_in = 1.0627727
const k2_in = 1.0627727
const k3_in = 1.0627727
const f_in = 60.
const ks_in = 1.0627727


singleparticle_drift_tests(l_in)
multiparticle_drift_tests(l_in)


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



# function create_drift(l)
#     dr = DRIFT(len=l)
#     return dr
# end

# function drift_track(l)
#     particles = zeros(Float64, 10000, 6)
#     # particles = zeros(SMatrix{10000, 6})
#     particles[:,1] .= .001
#     particles[:,2] .= .0001 
#     beam = Beam(particles)
#     line = [create_drift(l)]
#     linepass!(line, beam)
#     return beam.r
# end

# function drift_track_mthread(l)
#     particles = zeros(Float64, 10000, 6)
#     # particles = zeros(SMatrix{10000, 6})
#     particles[:,1] .= .001
#     particles[:,2] .= .0001 
#     beam = Beam(particles)
#     line = [create_drift(l)]
#     plinepass!(line, beam)
#     return beam.r
# end

# @btime drift_track(l_in)
# @btime drift_track_mthread(l_in)
# grad1 = autodiff(Forward, drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# grad2 = autodiff(Forward, drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# @btime autodiff(Forward, drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# @btime autodiff(Forward, drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)


#In drift.jl, instead of the if conditions being " ... != zeros(6) or zeros(6,6), I was able to achieve 
# DRIFT SINGLE
#   881.857 ns (29 allocations: 3.39 KiB) (~6ns reduction, 2 fewer allocations, .01 KiB increase)
#   125.526 μs (350 allocations: 40.95 KiB) (~15 μs increase, 2 fewer allocations, ~0.5 KiB increase )
#   2.978 μs (64 allocations: 6.94 KiB) (~negligibile decrease, 4 fewer allocations, ~0.9 Kib decrease)
#   130.695 μs (385 allocations: 50.50 KiB) (6 μs increase, 4 fewer allocations, 1 KiB increase)
# Single Thread AutoDiff = Multithread Autodiff?  true
# DRIFT MULTI
#   1.902 ms (60032 allocations: 2.75 MiB) (1 ms decrease, ~40000 fewer allocations, 9.15 MiB decrease )
#   669.636 μs (60353 allocations: 2.79 MiB) (672 μs decrease, ~40000 fewer allocations. ~9 MiB decrease)
#   16.804 ms (190062 allocations: 7.56 MiB) (~3 ms decrease, ~80000 fewer allocations, ~18.5 MiB decrease)
#   1.739 ms (190383 allocations: 7.60 MiB) (~1ms decrease, ~80000 fewer allocations, ~18.5 MiB decrease)
# Single Thread AutoDiff = Multithread Autodiff?  true