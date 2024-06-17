include("src/JuTrack.jl")
using .JuTrack
using Plots
using Random
using Optim

function ring_gen(k)
        D1 = DRIFT(name="D1", len=0.7)
        QF1 = KQUAD(name="QF1", len=0.6, k1=k[1])
        QF2 = KQUAD(name="QF2", len=0.6, k1=k[2])
        QF3 = KQUAD(name="QF3", len=0.6, k1=k[3])
        QF4 = KQUAD(name="QF4", len=0.6, k1=k[4])
        QF5 = KQUAD(name="QF5", len=0.6, k1=k[5])
        QF6 = KQUAD(name="QF6", len=0.6, k1=k[6])
        QF7 = KQUAD(name="QF7", len=0.6, k1=k[7])
        QF8 = KQUAD(name="QF8", len=0.6, k1=k[8])
        QD1 = KQUAD(name="QD1", len=0.6, k1=k[9])
        QD2 = KQUAD(name="QD2", len=0.6, k1=k[10])
        QD3 = KQUAD(name="QD3", len=0.6, k1=k[11])
        QD4 = KQUAD(name="QD4", len=0.6, k1=k[12])
        BendingAngle = pi/2.
        BD1 = SBEND(name="BD1", len=2.1, angle=BendingAngle )

        ringout = [QF1, D1, QD1, D1, QF2, BD1,
                QF3, D1, QD2, D1, QF4, BD1,
                QF5, D1, QD3, D1, QF6, BD1,
                QF7, D1, QD4, D1, QF8, BD1]

        return ringout
end


function ηx(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[6, 1]
end

function ηx_loss(θ, k, b, RING)
        return (ηx(k, RING) - ηx(kp(θ, k, b), RING))^2
end

function ηy(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[6, 3]
end

function ηy_loss(θ, k, b, RING)
        return (ηy(k, RING) - ηy(kp(θ, k, b), RING))^2
end

function tracex(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[1,1] + m66[2,2]
end

function tracex_loss(θ, k, b, RING)
        return (tracex(k, RING) - tracex(kp(θ, k, b), RING))^2
end

function tracey(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[3,3] + m66[4,4]
end

function tracey_loss(θ, k, b, RING)
        return (tracey(k, RING) - tracey(kp(θ, k, b), RING))^2
end

function βxmax(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        return max(twi.betax)
end

function βxmax_loss(θ, k, b, RING)
        return (βxmax(k, RING) - βxmax(kp(θ, k, b), RING))^2
end

function βymax(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        return max(twi.betay)
end

function βymax_loss(θ, k, b, RING)
        return (βymax(k, RING) - βymax(kp(θ, k, b), RING))^2
end

function βxvar(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        return (max(twi.betax) - minimum(twi.betax))/(max(twi.betax) + minimum(twi.betax))
end

function βxvar_loss(θ, k, b, RING)
        return (βxvar(k, RING) - βxvar(kp(θ, k, b), RING))^2
end

function βyvar(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        return (max(twi.betay) - minimum(twi.betay))/(max(twi.betay) + minimum(twi.betay))
end

function βyvar_loss(θ, k, b, RING)
        return (βyvar(k, RING) - βyvar(kp(θ, k, b), RING))^2
end

function tunex(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return acos((m66[1,1] + m66[2,2])/2)/(2*π)
end

function tunex_loss(θ, k, b, RING)
        # return (tunex(k, RING) - tunex(kp(θ, k, b), RING))^2
        return (0.58 - tunex(kp(θ, k, b), RING))^2
end

function tuney(k, RING)
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[2].len, k1=k[2]),KQUAD(len=RING[3].len, k1=k[3]),KQUAD(len=RING[4].len, k1=k[4]),
                KQUAD(len=RING[5].len, k1=k[5]), KQUAD(len=RING[6].len, k1=k[6]), KQUAD(len=RING[7].len, k1=k[7]), KQUAD(len=RING[8].len, k1=k[8]),
                KQUAD(len=RING[9].len, k1=k[9]), KQUAD(len=RING[10].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[11]), KQUAD(len=RING[12].len, k1=k[12])]
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return acos((m66[3,3] + m66[4,4])/2)/(2*π)
end

function tuney_loss(θ, k, b, RING)
        # return (tuney(k, RING) - tuney(kp(θ, k, b), RING))^2
        return (0.58 - tuney(kp(θ, k, b), RING))^2
end

kp(θ, k, b) = @. θ*k + b

function total_loss(θ, k, b, RING)
        w = fill(1.0, 10)
        return w[1]*ηx_loss(θ, k, b, RING) + w[2]*ηy_loss(θ, k, b, RING) + w[3]*tracex_loss(θ, k, b, RING) + w[4]*tracey_loss(θ, k, b, RING) - w[5]*βxmax_loss(θ, k, b, RING) - w[6]*βymax_loss(θ, k, b, RING) + w[7]*βxvar_loss(θ, k, b, RING) + w[8]*βyvar_loss(θ, k, b, RING) + w[9]*tunex_loss(θ, k, b, RING) + w[10]*tuney_loss(θ, k, b, RING)
end

k=zeros(12)
θ = rand(Float64, 12)
# b = rand(Float64, 12)
b = zeros(12)
k[1:8] .= 2.4
k[9:12] .= -2.4
ring = ring_gen(k)
grads= zeros(12)

counter = 0
# w = fill(1.0, 10)
# function optimize_(θ, k, b, ring)
        while total_loss(θ, k, b, ring) > 1e-2 || counter < 100
                println("work")
                diffmatrix = zeros(12, 12)
                for i in 1:12
                        diffmatrix[i,i] = 1.0
                        
                        grads[i] = try 
                                autodiff(Forward, total_loss, Duplicated, Duplicated(θ, diffmatrix[:,i]), Const(k), Const(b), Const(ring))
                                
                        catch e
                                if isa(e, ErrorException)
                                        println("Trying again caught")
                                else
                                        println("idk")
                                end
                                break
                                # counter += 1
                                # println("Trying again")
                                # return optimize_(θ, k, b, ring)
                                # break
                        end
                end
                # grads[1] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[2] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[3] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[4] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[5] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[6] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[7] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[8] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[9] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[10] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[11] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
                # grads[12] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]), Const(k), Const(b), Const(ring_gen(k)))
                θ .-= 0.1 .* grads
                counter += 1
        end

        println("new k: ", kp(θ, k, b))

# end

optimize_(θ, k, b, ring)

# abc = zeros(12, 12)
# for i in 1:12
#         abc[i,i] = 1.0
#         println(abc[:,i])
# end
# print(abc[:, 12])

# print([1 2 3] == [1, 2, 3])