include("../src/JuTrack.jl")
include("make_threading_tests.jl")
using .JuTrack
using Enzyme
# Enzyme.API.runtimeActivity!(true)
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



# singleparticle_drift_tests(l_in)
multiparticle_drift_tests(l_in)

function singleparticle_create_drift(l)
    dr = DRIFT(len=l)
    return dr
end

function singleparticle_drift_track(l)
    particles = [.001 .0001 .0005 .0002 0.0 0.0]  
    beam = Beam(particles)
    line = [singleparticle_create_drift(l)]
    linepass!(line, beam)
    return beam.r
end
function singleparticle_drift_track_mthread(l)
    particles = [.001 .0001 .0005 .0002 0.0 0.0] 
    beam = Beam(particles)
    line = [singleparticle_create_drift(l)]
    plinepass!(line, beam)
    return beam.r
end

function multiparticle_create_drift(l)
    dr = DRIFT(len=l)
    return dr
end

function multiparticle_drift_track(l)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [multiparticle_create_drift(l)]
    linepass!(line, beam)
    return beam.r
end
function multiparticle_drift_track_mthread(l)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [multiparticle_create_drift(l)]
    plinepass!(line, beam)
    return beam.r
end
particles = zeros(Float64, 10000, 6) 
particles[:,1] .= .001
particles[:,2] .= .0001



# println("DRIFT SINGLE")
# @btime singleparticle_drift_track(l_in)
# GC.gc()
# @btime singleparticle_drift_track_mthread(l_in)
# println("   ENZYME SEGFAULT'S HERE")
# println("   ENZYME SEGFAULT'S HERE")
# GC.gc()
# grad1 = autodiff(Forward, singleparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# grad2 = autodiff(Forward, singleparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# GC.gc()
# @btime autodiff(Forward, singleparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# @btime autodiff(Forward, singleparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)

# println("DRIFT MULTI")
# @btime multiparticle_drift_track(l_in)
# GC.gc()
# @btime multiparticle_drift_track_mthread(l_in)
# GC.gc()
# grad1 = autodiff(Forward, multiparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# grad2 = autodiff(Forward, multiparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# GC.gc()
# @btime autodiff(Forward, multiparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# @btime autodiff(Forward, multiparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)

# @btime abcde = zeros(Float64, 10000, 6) #Making the intial preallocation should only take up 2 allocations, 468.80 KiB
# @btime abcdef = zeros(Float64, 1, 6) # 24.544 ns (1 allocation: 112 bytes)


#################   ORIGINAL 
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

#Changes made:
#   In drift.jl simplified array_to_matrix() and matrix_to_array, this had minor effect
#   In drift.jl, drift6!() and fastdrift!() replaced ==1 with isone() this had minor effect
#   In drift.jl, DriftPass(_P)!() replaces != zeros(6) and zeros(6,6) with !iszero(), this had major effect
#   In drift.jl, changed the multiple or statements in DriftPass(_P)!() to have absolute values, this had minor effect
#   In JuTrack.jl, changed global variables to be constant global variables, this had major effect
#   In track_mthread.jl, remove unneedded variables rout and ele, minor effect on time

#################   CURRENT 
# DRIFT SINGLE
#   625.582 ns (23 allocations: 2.78 KiB)
#   121.318 μs (344 allocations: 39.34 KiB)
#    ENZYME SEGFAULT'S HERE
#    ENZYME SEGFAULT'S HERE
# DRIFT MULTI
#   572.615 μs (30 allocations: 940.12 KiB)
#   176.391 μs (351 allocations: 976.69 KiB)
#   545.513 μs (56 allocations: 1.76 MiB)
#   368.831 μs (377 allocations: 1.80 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true