module mthread_multi_test_funcs

include("../src/JuTrack.jl")
using .JuTrack
using Enzyme
using Test
using BenchmarkTools




function create_sbend(BendingAngle)
    SBD = SBEND(name="BD", len=0.72, angle=BendingAngle/2, e1=BendingAngle/2, e2=0.0 , rad=1)
    return [SBD]
end        

function sbend_track_mthread(BendingAngle)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001
    beam = Beam(particles)
    line = create_sbend(BendingAngle)
    plinepass!(line, beam)
    return beam.r
end

function sbend_track(BendingAngle)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001
    beam = Beam(particles)
    line = create_sbend(BendingAngle)
    linepass!(line, beam)
    return beam.r
end

function create_rbend(BendingAngle)
    RBD = RBEND(name="BD", len=0.72, angle=BendingAngle/2, rad=1)
    return [RBD]
end        

function rbend_track_mthread(BendingAngle)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = create_rbend(BendingAngle)
    plinepass!(line, beam)
    return beam.r
end

function rbend_track(BendingAngle)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = create_rbend(BendingAngle)
    linepass!(line, beam)
    return beam.r
end

function hcorrector_track(hkick)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [HKICKER(name="HKICK", len=1.5, xkick=hkick)]
    linepass!(line, beam)
    return beam.r
end

function hcorrector_track_mthread(hkick)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [HKICKER(name="HKICK", len=1.5, xkick=hkick)]
    plinepass!(line, beam)
    return beam.r
end

function vcorrector_track(vkick)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [VKICKER(name="VKICK", len=1.5, ykick=vkick)]
    linepass!(line, beam)
    return beam.r
end

function vcorrector_track_mthread(vkick)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [VKICKER(name="VKICK", len=1.5, ykick=vkick)]
    plinepass!(line, beam)
    return beam.r
end

function create_drift(l)
    dr = DRIFT(len=l)
    return dr
end


function drift_track(l)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_drift(l)]
    linepass!(line, beam)
    return beam.r
end

function drift_track_mthread(l)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_drift(l)]
    plinepass!(line, beam)
    return beam.r
end

function create_quad(k)
    return KQUAD(len=0.5, k1=k)
end

function quad_track(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_quad(k)]
    linepass!(line, beam)
    return beam.r
end

function quad_track_mthread(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_quad(k)]
    plinepass!(line, beam)
    return beam.r
end

function create_sext(k)
    return KSEXT(len=0.5, k2=k)
end

function sext_track(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_sext(k)]
    linepass!(line, beam)
    return beam.r
end

function sext_track_mthread(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_sext(k)]
    plinepass!(line, beam)
    return beam.r
end

function create_oct(k)
    return KOCT(len=0.5, k3=k)
end

function oct_track(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_oct(k)]
    linepass!(line, beam)
    return beam.r
end

function oct_track_mthread(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_oct(k)]
    plinepass!(line, beam)
    return beam.r
end

function create_RFCA(f)
    return RFCA(len=1.034, volt=2.2, freq=f, energy = 30e4)
end

function RFCA_track(f)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_RFCA(f)]
    linepass!(line, beam)
    return beam.r
end

function RFCA_track_mthread(f)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_RFCA(f)]
    plinepass!(line, beam)
    return beam.r
end


function create_sol(k)
    return SOLENOID(len=0.5, ks=k)
end

function sol_track(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_sol(k)]
    linepass!(line, beam)
    return beam.r
end

function sol_track_mthread(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_sol(k)]
    plinepass!(line, beam)
    return beam.r
end

function create_thinMulti(k)
    return thinMULTIPOLE(len=0.5, PolynomB = [0.0, k, 1.43, -1.32])
end

function thinMulti_track(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_thinMulti(k)]
    linepass!(line, beam)
    return beam.r
end

function thinMulti_track_mthread(k)
    particles = zeros(Float64, 10000, 6)
    particles[:,1] .= .001
    particles[:,2] .= .0001 
    beam = Beam(particles)
    line = [create_thinMulti(k)]
    plinepass!(line, beam)
    return beam.r
end


export create_sbend, sbend_track, sbend_track_mthread
export create_rbend, rbend_track, rbend_track_mthread
export hcorrector_track, hcorrector_track_mthread
export vcorrector_track, vcorrector_track_mthread
export create_drift, drift_track, drift_track_mthread
export create_quad, quad_track, quad_track_mthread
export create_sext, sext_track, sext_track_mthread
export create_oct, oct_track, oct_track_mthread
export create_RFCA, RFCA_track, RFCA_track_mthread
export create_sol, sol_track, sol_track_mthread
export create_thinMulti, thinMulti_track, thinMulti_track_mthread

end