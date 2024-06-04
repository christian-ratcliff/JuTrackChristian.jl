include("../src/JuTrack.jl")
using .JuTrack
using LaTeXStrings
# using Enzyme
# Enzyme.API.runtimeActivity!(true)



function f(x1) # treat beam as a constant will lead to 0 gradient
    # particles = randn(50000, 6) / 1e3
    # ebeam = Beam(particles, znbin=2)
    RLCwake = LongitudinalRLCWake(180e9, 5.5e3, x1)
    D1 = DRIFT(len=1.0)
    # RLCwake = KSEXT(len=1.0, k2= x1)
    # vbase=3.42*8.5e6
    # ϕs=10.0
    # vact=vbase/cos(ϕs*π/180.0)
    # mainRFe=AccelCavity(591e6, vact, 7560.0, π-ϕs*π/180.0)
    # tunex, tuney=50.08, 44.14
    # alphac=3.42/tunex/tunex
    # lmap=LongitudinalRFMap(alphac, mainRFe)

    particles = zeros(50000, 6) 
    particles[:, 1] .= randn(50000) ./ 1e3
    particles[:, 2] .= randn(50000) ./ 1e5
    particles[:, 3] .= randn(50000) ./ 1e3
    particles[:, 4] .= randn(50000) ./ 1e4
    particles[:, 5] .= randn(50000) ./ 1e6
    particles[:, 6] .= randn(50000) ./ 1e4
    ebeam = Beam(particles)
    histogram1DinZ!(ebeam)

    linepass!([D1, RLCwake], ebeam)
    get_emittance!(ebeam)
    # scatter(ebeam.r[:, 1], ebeam.r[:, 2], markersize=0.1, label="x-y")
    return ebeam.emittance
end

grad = autodiff(Forward, f, Duplicated,  Duplicated(180e9, 1.0)) # 11.713 s (1750155 allocations: 74.56 GiB)
println(grad)




