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


#Results are reported in the following format:

# (Element Type) (Single or Multiple Particle)
# (tracking with one thread)
# (tracking with 64 threads)
# (autodiff with one thread)
# (autodiff with 64 threads)
# (are they the same)

#Many of the single particle tests lead to a SegFault for a reason that I don't know, which is why only the multi are reported


# multiparticle_drift_tests(l_in)
# singleparticle_drift_tests(l_in)

# multiparticle_rbend_tests(bend_angle_in)
# singleparticle_rbend_tests(bend_angle_in)

# multiparticle_sbend_tests(bend_angle_in)
# singleparticle_sbend_tests(bend_angle_in)

# multiparticle_hcorrector_tests(hkick_in)
# singleparticle_hcorrector_tests(hkick_in)

# multiparticle_vcorrector_tests(vkick_in)
# singleparticle_vcorrector_tests(vkick_in)

# multiparticle_quad_tests(k1_in)
# singleparticle_quad_tests(k1_in)

# multiparticle_sext_tests(k2_in)
# singleparticle_sext_tests(k2_in)

# multiparticle_oct_tests(k3_in)
# singleparticle_oct_tests(k3_in)

# multiparticle_RFCA_tests(f_in)
# singleparticle_RFCA_tests(f_in)

# multiparticle_sol_tests(ks_in)
# singleparticle_sol_tests(ks_in)

# multiparticle_thinMulti_tests(k1_in)
# singleparticle_thinMulti_tests(k1_in)


# DRIFT MULTI
#   477.466 μs (34 allocations: 1020.08 KiB)
#   227.447 μs (355 allocations: 1.03 MiB)
#   558.798 μs (62 allocations: 1.84 MiB)
#   406.863 μs (383 allocations: 1.88 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# RBEND MULTI
#   9.950 ms (39 allocations: 1020.67 KiB)
#   663.935 μs (362 allocations: 1.04 MiB)
#   14.320 ms (72 allocations: 1.84 MiB)
#   770.015 μs (397 allocations: 1.90 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# SBEND MULTI
#   10.088 ms (39 allocations: 1020.67 KiB)
#   700.254 μs (362 allocations: 1.04 MiB)
#   14.293 ms (72 allocations: 1.84 MiB)
#   871.976 μs (397 allocations: 1.90 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# HCORRECTOR MULTI
#   445.526 μs (32 allocations: 1019.86 KiB)
#   320.982 μs (353 allocations: 1.03 MiB)
#   508.133 μs (58 allocations: 1.84 MiB)
#   411.322 μs (379 allocations: 1.88 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  false
# VCORRECTOR MULTI
#   449.213 μs (32 allocations: 1019.86 KiB)
#   305.133 μs (353 allocations: 1.03 MiB)
#   553.097 μs (58 allocations: 1.84 MiB)
#   424.416 μs (379 allocations: 1.88 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  false
# QUAD MULTI
#   4.613 ms (40 allocations: 1020.72 KiB)
#   491.822 μs (363 allocations: 1.04 MiB)
#   7.392 ms (74 allocations: 1.84 MiB)
#   625.743 μs (399 allocations: 1.90 MiB)
# SEXT MULTI
#   5.091 ms (40 allocations: 1020.72 KiB)
#   588.334 μs (363 allocations: 1.04 MiB)
#   8.251 ms (74 allocations: 1.84 MiB)
#   733.185 μs (399 allocations: 1.90 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# OCT MULTI
#   5.690 ms (40 allocations: 1020.72 KiB)
#   547.387 μs (363 allocations: 1.04 MiB)
#   9.586 ms (74 allocations: 1.84 MiB)
#   742.603 μs (399 allocations: 1.90 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# RFCA MULTI
#   259.727 μs (28 allocations: 1018.92 KiB)
#   212.078 μs (349 allocations: 1.03 MiB)
#   398.698 μs (50 allocations: 1.84 MiB)
#   392.225 μs (371 allocations: 1.88 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# SOL MULTI
#   640.140 μs (32 allocations: 1019.86 KiB)
#   343.264 μs (353 allocations: 1.03 MiB)
#   882.736 μs (58 allocations: 1.84 MiB)
#   450.364 μs (379 allocations: 1.89 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true
# THINMULTI MULTI
#   491.031 μs (39 allocations: 1020.61 KiB)
#   342.022 μs (360 allocations: 1.03 MiB)
#   603.221 μs (72 allocations: 1.84 MiB)
#   450.645 μs (393 allocations: 1.89 MiB)
# Single Thread AutoDiff = Multithread Autodiff?  true





#################### You can ignore everything below##########################

# function singleparticle_create_drift(l)
#     dr = DRIFT(len=l)
#     return dr
# end

# function singleparticle_drift_track(l)
#     particles = [.001 .0001 .0005 .0002 0.0 0.0]  
#     beam = Beam(particles)
#     line = [singleparticle_create_drift(l)]
#     linepass!(line, beam)
#     return beam.r
# end
# function singleparticle_drift_track_mthread(l)
#     particles = [.001 .0001 .0005 .0002 0.0 0.0] 
#     beam = Beam(particles)
#     line = [singleparticle_create_drift(l)]
#     plinepass!(line, beam)
#     return beam.r
# end

# function multiparticle_create_drift(l)
#     dr = DRIFT(len=l)
#     return dr
# end

# function multiparticle_drift_track(l)
#     particles = zeros(Float64, 10000, 6)
#     particles[:,1] .= .001
#     particles[:,2] .= .0001 
#     beam = Beam(particles)
#     line = [multiparticle_create_drift(l)]
#     linepass!(line, beam)
#     return beam.r
# end
# function multiparticle_drift_track_mthread(l)
#     particles = zeros(Float64, 10000, 6)
#     particles[:,1] .= .001
#     particles[:,2] .= .0001 
#     beam = Beam(particles)
#     line = [multiparticle_create_drift(l)]
#     plinepass!(line, beam)
#     return beam.r
# end




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
# # @btime autodiff(Forward, singleparticle_drift_track, DuplicatedNoNeed, Duplicated(l_in, 1.0))
# # @btime autodiff(Forward, singleparticle_drift_track_mthread, DuplicatedNoNeed, Duplicated(l_in, 1.0))
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

# print(Threads.nthreads())

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