include("src/JuTrack.jl")
using .JuTrack
using BenchmarkTools

function ring_gen_ini(k::Float64)
    D1 = DRIFT(name="D1", len=0.7)
    QF1 = KQUAD(name="QF1", len=0.6, k1=k)
    QD1 = KQUAD(name="QD1", len=0.6, k1=-k)
    BendingAngle = pi/2.
    BD1 = SBEND(name="BD1", len=2.1, angle=BendingAngle )

    ringout = [QF1, D1, QD1, D1, QF1, BD1,
            QF1, D1, QD1, D1, QF1, BD1,
            QF1, D1, QD1, D1, QF1, BD1,
            QF1, D1, QD1, D1, QF1, BD1]

    return ringout
end

kp(θ, k, b) = @. θ*k + b

kstart = 2.2
θ = 1.0+0.1*rand()
b = 0.1*rand()
RING = ring_gen_ini(kstart)

order = 3
dp = 0.0
x = CTPS(0.0, 1, 6, order)
px = CTPS(0.0, 2, 6, order)
y = CTPS(0.0, 3, 6, order)
py = CTPS(0.0, 4, 6, order)
z = CTPS(dp, 5, 6, order)
delta = CTPS(0.0, 6, 6, order)
rin = [x, px, y, py, z, delta]

k = kp(θ,kstart,b)

#Both of these located in EdwardsTengTwiss.jl
# ADlinepass_TPSA! is a function inside of ADfindm66
@btime ADlinepass_TPSA!(RING, rin, [], [])
#65.857 ms (2128910 allocations: 205.31 MiB)
@btime ADfindm66(RING, 0.0, 3, [], [])
#59.226 ms (1881728 allocations: 183.41 MiB)


