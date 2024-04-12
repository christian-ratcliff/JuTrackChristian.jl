include("../src/JuTrack.jl")
include("make_threading_tests.jl")
using .JuTrack
using Enzyme
Enzyme.API.runtimeActivity!(true)
using Test
using BenchmarkTools
# using .threading_tests



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
# multiparticle_drift_tests(l_in)

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



println("DRIFT SINGLE")
# singleparticle_drift_track(l_in)
# println(singleparticle_drift_track_mthread(l_in))
# @btime singleparticle_drift_track(l_in)
# GC.gc()
# @btime singleparticle_drift_track_mthread(l_in)
# GC.gc()
# grad1 = autodiff(Forward, singleparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# grad2 = autodiff(Forward, singleparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# GC.gc()
@btime autodiff(Forward, singleparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
@btime autodiff(Forward, singleparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)

println("DRIFT MULTI")
# multiparticle_drift_track(l_in)
# multiparticle_drift_track_mthread(l_in)
@btime multiparticle_drift_track(l_in)
GC.gc()
@btime multiparticle_drift_track_mthread(l_in)
GC.gc()
grad1 = autodiff(Forward, multiparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
grad2 = autodiff(Forward, multiparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
GC.gc()
@btime autodiff(Forward, multiparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
@btime autodiff(Forward, multiparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)

# @btime abcde = zeros(Float64, 10000, 6)
#Making the intial preallocation should only take up 2 allocations

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


#In drift.jl, instead of the if conditions being " ... != zeros(6) or zeros(6,6), and instead assigning a variable to zeros() I was able to achieve 
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
# Now the allocations are directly related to the increse in the number of particles

#Able to further reduce memory by changing the array_to_matrix and matrix_to_array to avoid loops, and changing my previous method to use !iszero. Noticeable improvement in mthread time 
# DRIFT SINGLE
#   794.720 ns (29 allocations: 2.88 KiB) 
#   118.483 μs (350 allocations: 39.44 KiB)
#   2.923 μs (62 allocations: 5.86 KiB)
#   119.695 μs (383 allocations: 47.42 KiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# DRIFT MULTI
#   1.604 ms (60030 allocations: 1.83 MiB) 650.67x more memory with 10000 particles vs 1
#   355.256 μs (60351 allocations: 1.87 MiB) 48.55x more memory with 10000 particles vs 1
#   16.820 ms (190056 allocations: 5.73 MiB) 1001x more memory with 10000 particles vs 1
#   1.523 ms (190377 allocations: 5.77 MiB) 124.6x more memory with 10000 particles vs 1
# Single Thread AutoDiff = Multithread Autodiff?  true



#By changing the variables defined in the global scope in JuTrack.jl to constants, significant reduction in discrepancy between 