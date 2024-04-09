module threading_tests

include("../src/JuTrack.jl")
include("mthread_multi_test_funcs.jl")
include("mthread_single_test_funcs.jl")
using .JuTrack
using .mthread_multi_test_funcs 
using .mthread_single_test_funcs 
using Enzyme
using Test
using BenchmarkTools
const multiparticle = mthread_multi_test_funcs
const singleparticle = mthread_single_test_funcs






function multiparticle_rbend_tests(bend_angle)
    println("RBEND MULTI")
    @btime multiparticle.rbend_track(bend_angle)
    @btime multiparticle.rbend_track_mthread(bend_angle)
    grad1 = autodiff(Forward, multiparticle.rbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    grad2 = autodiff(Forward, multiparticle.rbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, multiparticle.rbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, multiparticle.rbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_rbend_tests(bend_angle)
    println("RBEND SINGLE")
    @btime singleparticle.rbend_track(bend_angle)
    @btime singleparticle.rbend_track_mthread(bend_angle)
    grad1 = autodiff(Forward, singleparticle.rbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    grad2 = autodiff(Forward, singleparticle.rbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, singleparticle.rbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, singleparticle.rbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_sbend_tests(bend_angle)
    println("SBEND MULTI")
    @btime multiparticle.sbend_track(bend_angle)
    @btime multiparticle.sbend_track_mthread(bend_angle)
    grad1 = autodiff(Forward, multiparticle.sbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    grad2 = autodiff(Forward, multiparticle.sbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, multiparticle.sbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, multiparticle.sbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_sbend_tests(bend_angle)
    println("SBEND SINGLE")
    @btime singleparticle.sbend_track(bend_angle)
    @btime singleparticle.sbend_track_mthread(bend_angle)
    grad1 = autodiff(Forward, singleparticle.sbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    grad2 = autodiff(Forward, singleparticle.sbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, singleparticle.sbend_track, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    @btime autodiff(Forward, singleparticle.sbend_track_mthread, DuplicatedNoNeed, Duplicated(bend_angle, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_hcorrector_tests(hkick)
    println("HCORRECTOR MULTI")
    @btime multiparticle.hcorrector_track(hkick)
    @btime multiparticle.hcorrector_track_mthread(hkick)
    grad1 = autodiff(Forward, multiparticle.hcorrector_track, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    grad2 = autodiff(Forward, multiparticle.hcorrector_track_mthread, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    @btime autodiff(Forward, multiparticle.hcorrector_track, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    @btime autodiff(Forward, multiparticle.hcorrector_track_mthread, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_hcorrector_tests(hkick)
    println("HCORRECTOR SINGLE")
    @btime singleparticle.hcorrector_track(hkick)
    @btime singleparticle.hcorrector_track_mthread(hkick)
    grad1 = autodiff(Forward, singleparticle.hcorrector_track, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    grad2 = autodiff(Forward, singleparticle.hcorrector_track_mthread, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    @btime autodiff(Forward, singleparticle.hcorrector_track, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    @btime autodiff(Forward, singleparticle.hcorrector_track_mthread, DuplicatedNoNeed, Duplicated(hkick, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_vcorrector_tests(vkick)
    println("VCORRECTOR MULTI")
    @btime multiparticle.vcorrector_track(vkick)
    @btime multiparticle.vcorrector_track_mthread(vkick)
    grad1 = autodiff(Forward, multiparticle.vcorrector_track, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    grad2 = autodiff(Forward, multiparticle.vcorrector_track_mthread, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    @btime autodiff(Forward, multiparticle.vcorrector_track, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    @btime autodiff(Forward, multiparticle.vcorrector_track_mthread, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_vcorrector_tests(vkick)
    println("VCORRECTOR SINGLE")
    @btime singleparticle.vcorrector_track(vkick)
    @btime singleparticle.vcorrector_track_mthread(vkick)
    grad1 = autodiff(Forward, singleparticle.vcorrector_track, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    grad2 = autodiff(Forward, singleparticle.vcorrector_track_mthread, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    @btime autodiff(Forward, singleparticle.vcorrector_track, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    @btime autodiff(Forward, singleparticle.vcorrector_track_mthread, DuplicatedNoNeed, Duplicated(vkick, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_drift_tests(l)
    println("DRIFT MULTI")
    @btime multiparticle.drift_track($l)
    @btime multiparticle.drift_track_mthread($l)
    grad1 = autodiff(Forward, multiparticle.drift_track, DuplicatedNoNeed, Duplicated(l, 1.0))
    grad2 = autodiff(Forward, multiparticle.drift_track_mthread, DuplicatedNoNeed, Duplicated(l, 1.0))
    @btime autodiff(Forward, multiparticle.drift_track, DuplicatedNoNeed, Duplicated($l, 1.0))
    @btime autodiff(Forward, multiparticle.drift_track_mthread, DuplicatedNoNeed, Duplicated($l, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_drift_tests(l)
    println("DRIFT SINGLE")
    @btime singleparticle.drift_track($l)
    @btime singleparticle.drift_track_mthread($l)
    grad1 = autodiff(Forward, singleparticle.drift_track, DuplicatedNoNeed, Duplicated(l, 1.0))
    grad2 = autodiff(Forward, singleparticle.drift_track_mthread, DuplicatedNoNeed, Duplicated(l, 1.0))
    @btime autodiff(Forward, singleparticle.drift_track, DuplicatedNoNeed, Duplicated($l, 1.0))
    @btime autodiff(Forward, singleparticle.drift_track_mthread, DuplicatedNoNeed, Duplicated($l, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_quad_tests(k1)
    println("QUAD MULTI")
    @btime multiparticle.quad_track(k1)
    @btime multiparticle.quad_track_mthread(k1)
    grad1 = autodiff(Forward, multiparticle.quad_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    grad2 = autodiff(Forward, multiparticle.quad_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, multiparticle.quad_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, multiparticle.quad_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    return nothing
end

function singleparticle_quad_tests(k1)
    println("QUAD SINGLE")
    @btime singleparticle.quad_track(k1)
    @btime singleparticle.quad_track_mthread(k1)
    grad1 = autodiff(Forward, singleparticle.quad_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    grad2 = autodiff(Forward, singleparticle.quad_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, singleparticle.quad_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, singleparticle.quad_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    return nothing
end

function multiparticle_sext_tests(k1)
    println("SEXT MULTI")
    @btime multiparticle.sext_track(k2)
    @btime multiparticle.sext_track_mthread(k2)
    grad1 = autodiff(Forward, multiparticle.sext_track, DuplicatedNoNeed, Duplicated(k2, 1.0))
    grad2 = autodiff(Forward, multiparticle.sext_track_mthread, DuplicatedNoNeed, Duplicated(k2, 1.0))
    @btime autodiff(Forward, multiparticle.sext_track, DuplicatedNoNeed, Duplicated(k2, 1.0))
    @btime autodiff(Forward, multiparticle.sext_track_mthread, DuplicatedNoNeed, Duplicated(k2, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_sext_tests(k2)
    println("SEXT SINGLE")
    @btime singleparticle.sext_track(k2)
    @btime singleparticle.sext_track_mthread(k2)
    grad1 = autodiff(Forward, singleparticle.sext_track, DuplicatedNoNeed, Duplicated(k2, 1.0))
    grad2 = autodiff(Forward, singleparticle.sext_track_mthread, DuplicatedNoNeed, Duplicated(k2, 1.0))
    @btime autodiff(Forward, singleparticle.sext_track, DuplicatedNoNeed, Duplicated(k2, 1.0))
    @btime autodiff(Forward, singleparticle.sext_track_mthread, DuplicatedNoNeed, Duplicated(k2, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_oct_tests(k3)
    println("OCT MULTI")
    @btime multiparticle.oct_track(k3)
    @btime multiparticle.oct_track_mthread(k3)
    grad1 = autodiff(Forward, multiparticle.oct_track, DuplicatedNoNeed, Duplicated(k3, 1.0))
    grad2 = autodiff(Forward, multiparticle.oct_track_mthread, DuplicatedNoNeed, Duplicated(k3, 1.0))
    @btime autodiff(Forward, multiparticle.oct_track, DuplicatedNoNeed, Duplicated(k3, 1.0))
    @btime autodiff(Forward, multiparticle.oct_track_mthread, DuplicatedNoNeed, Duplicated(k3, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_oct_tests(k3)
    println("OCT SINGLE")
    @btime singleparticle.oct_track(k3)
    @btime singleparticle.oct_track_mthread(k3)
    grad1 = autodiff(Forward, singleparticle.oct_track, DuplicatedNoNeed, Duplicated(k3, 1.0))
    grad2 = autodiff(Forward, singleparticle.oct_track_mthread, DuplicatedNoNeed, Duplicated(k3, 1.0))
    @btime autodiff(Forward, singleparticle.oct_track, DuplicatedNoNeed, Duplicated(k3, 1.0))
    @btime autodiff(Forward, singleparticle.oct_track_mthread, DuplicatedNoNeed, Duplicated(k3, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_RFCA_tests(f)
    println("RFCA MULTI")
    @btime multiparticle.RFCA_track(f)
    @btime multiparticle.RFCA_track_mthread(f)
    grad1 = autodiff(Forward, multiparticle.RFCA_track, DuplicatedNoNeed, Duplicated(f, 1.0))
    grad2 = autodiff(Forward, multiparticle.RFCA_track_mthread, DuplicatedNoNeed, Duplicated(f, 1.0))
    @btime autodiff(Forward, multiparticle.RFCA_track, DuplicatedNoNeed, Duplicated(f, 1.0))
    @btime autodiff(Forward, multiparticle.RFCA_track_mthread, DuplicatedNoNeed, Duplicated(f, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_RFCA_tests(f)
    println("RFCA SINGLE")
    @btime singleparticle.RFCA_track(f)
    @btime singleparticle.RFCA_track_mthread(f)
    grad1 = autodiff(Forward, singleparticle.RFCA_track, DuplicatedNoNeed, Duplicated(f, 1.0))
    grad2 = autodiff(Forward, singleparticle.RFCA_track_mthread, DuplicatedNoNeed, Duplicated(f, 1.0))
    @btime autodiff(Forward, singleparticle.RFCA_track, DuplicatedNoNeed, Duplicated(f, 1.0))
    @btime autodiff(Forward, singleparticle.RFCA_track_mthread, DuplicatedNoNeed, Duplicated(f, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_sol_tests(ks)
    println("SOL MULTI")
    @btime multiparticle.sol_track(ks)
    @btime multiparticle.sol_track_mthread(ks)
    grad1 = autodiff(Forward, multiparticle.sol_track, DuplicatedNoNeed, Duplicated(ks, 1.0))
    grad2 = autodiff(Forward, multiparticle.sol_track_mthread, DuplicatedNoNeed, Duplicated(ks, 1.0))
    @btime autodiff(Forward, multiparticle.sol_track, DuplicatedNoNeed, Duplicated(ks, 1.0))
    @btime autodiff(Forward, multiparticle.sol_track_mthread, DuplicatedNoNeed, Duplicated(ks, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_sol_tests(ks)
    println("SOL SINGLE")
    @btime singleparticle.sol_track(ks)
    @btime singleparticle.sol_track_mthread(ks)
    grad1 = autodiff(Forward, singleparticle.sol_track, DuplicatedNoNeed, Duplicated(ks, 1.0))
    grad2 = autodiff(Forward, singleparticle.sol_track_mthread, DuplicatedNoNeed, Duplicated(ks, 1.0))
    @btime autodiff(Forward, singleparticle.sol_track, DuplicatedNoNeed, Duplicated(ks, 1.0))
    @btime autodiff(Forward, singleparticle.sol_track_mthread, DuplicatedNoNeed, Duplicated(ks, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function multiparticle_thinMulti_tests(k1)
    println("THINMULTI MULTI")
    @btime multiparticle.thinMulti_track(k1)
    @btime multiparticle.thinMulti_track_mthread(k1)  
    grad1 = autodiff(Forward, multiparticle.thinMulti_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    grad2 = autodiff(Forward, multiparticle.thinMulti_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, multiparticle.thinMulti_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, multiparticle.thinMulti_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

function singleparticle_thinMulti_tests(k1)
    println("THINMULTI SINGLE")
    @btime singleparticle.thinMulti_track(k1)
    @btime singleparticle.thinMulti_track_mthread(k1)  
    grad1 = autodiff(Forward, singleparticle.thinMulti_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    grad2 = autodiff(Forward, singleparticle.thinMulti_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, singleparticle.thinMulti_track, DuplicatedNoNeed, Duplicated(k1, 1.0))
    @btime autodiff(Forward, singleparticle.thinMulti_track_mthread, DuplicatedNoNeed, Duplicated(k1, 1.0))
    println("Single Thread AutoDiff = Multithread Autodiff?  ", grad1 == grad2)
    return nothing
end

export multiparticle_rbend_tests, singleparticle_rbend_tests
export multiparticle_sbend_tests, singleparticle_sbend_tests
export multiparticle_hcorrector_tests, singleparticle_hcorrector_tests
export multiparticle_vcorrector_tests, singleparticle_vcorrector_tests
export multiparticle_drift_tests, singleparticle_drift_tests
export multiparticle_quad_tests, singleparticle_quad_tests
export multiparticle_sext_tests, singleparticle_sext_tests
export multiparticle_oct_tests, singleparticle_oct_tests
export multiparticle_RFCA_tests, singleparticle_RFCA_tests
export multiparticle_sol_tests, singleparticle_sol_tests
export multiparticle_thinMulti_tests, singleparticle_thinMulti_tests



end