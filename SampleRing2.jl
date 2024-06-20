include("src/JuTrack.jl")
using .JuTrack
using Plots
using Random
using Enzyme
using Optim
using BenchmarkTools

function ring_gen(k::Vector{Float64})
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


function ηx(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[6, 1]
end

function ηx_loss(θ, k, b, RING)
        return (ηx(k, RING) - ηx(kp(θ, k, b), RING))^2
end

function ηy(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[6, 3]
end

function ηy_loss(θ, k, b, RING)
        return (ηy(k, RING) - ηy(kp(θ, k, b), RING))^2
end

function tracex(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[1,1] + m66[2,2]
end

function tracex_loss(θ, k, b, RING)
        return (tracex(k, RING) - tracex(kp(θ, k, b), RING))^2
end

function tracey(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return m66[3,3] + m66[4,4]
end

function tracey_loss(θ, k, b, RING)
        return (tracey(k, RING) - tracey(kp(θ, k, b), RING))^2
end

function βxmax(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        betaxstore = zeros(12)
        for i in 1:12
                betaxstore[i] = twi[i].betax
        end     
        return maximum(betaxstore)
end

function βxmax_loss(θ, k, b, RING)
        return (βxmax(k, RING) - βxmax(kp(θ, k, b), RING))^2
end

function βymax(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        betaystore = zeros(12)
        for i in 1:12
                betaystore[i] = twi[i].betay
        end     
        return maximum(betaystore)
end

function βymax_loss(θ, k, b, RING)
        return (βymax(k, RING) - βymax(kp(θ, k, b), RING))^2
end

function βxvar(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        betaxstore = zeros(12)
        for i in 1:12
                betaxstore[i] = twi[i].betax
        end     
        return (maximum(betaxstore) - minimum(betaxstore))/(maximum(betaxstore) + minimum(betaxstore))
end

function βxvar_loss(θ, k, b, RING)
        return (βxvar(k, RING) - βxvar(kp(θ, k, b), RING))^2
end

function βyvar(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        betaystore = zeros(12)
        for i in 1:12
                betaystore[i] = twi[i].betay
        end     
        return (maximum(betaystore) - minimum(betaystore))/(maximum(betaystore) + minimum(betaystore))
end

function βyvar_loss(θ, k, b, RING)
        return (βyvar(k, RING) - βyvar(kp(θ, k, b), RING))^2
end

function tunex(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        return acos((m66[1,1] + m66[2,2])/2)/(2*π)
end

function tunex_loss(θ, k, b, RING)
        # return (tunex(k, RING) - tunex(kp(θ, k, b), RING))^2
        return (0.58 - tunex(kp(θ, k, b), RING))^2
end

function tuney(k::Vector{Float64}, RING::Vector{AbstractElement})
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k[1]), KQUAD(len=RING[3].len, k1=k[9]), KQUAD(len=RING[5].len, k1=k[2]), KQUAD(len=RING[7].len, k1=k[3]),
                KQUAD(len=RING[9].len, k1=k[10]), KQUAD(len=RING[11].len, k1=k[4]), KQUAD(len=RING[13].len, k1=k[5]),KQUAD(len=RING[15].len, k1=k[11]),
                KQUAD(len=RING[17].len, k1=k[6]), KQUAD(len=RING[19].len, k1=k[7]), KQUAD(len=RING[21].len, k1=k[12]),KQUAD(len=RING[23].len, k1=k[8])] 
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

function trace_updated_focus_quad_strengthx(k::Float64, RING::Vector{AbstractElement})
        # changed_idx = [1, 5, 8, 12, 15, 19, 22, 26]
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k), KQUAD(len=RING[3].len, k1=-k), KQUAD(len=RING[5].len, k1=k), KQUAD(len=RING[7].len, k1=k),
                KQUAD(len=RING[9].len, k1=-k), KQUAD(len=RING[11].len, k1=k), KQUAD(len=RING[13].len, k1=k),KQUAD(len=RING[15].len, k1=-k),
                KQUAD(len=RING[17].len, k1=k), KQUAD(len=RING[19].len, k1=k), KQUAD(len=RING[21].len, k1=-k),KQUAD(len=RING[23].len, k1=k)] 
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        trace = m66[1, 1] + m66[2, 2]
        return trace
end

function trace_updated_focus_quad_strengthy(k::Float64, RING::Vector{AbstractElement})
        # changed_idx = [1, 5, 8, 12, 15, 19, 22, 26]
        changed_idx = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
        changed_ele = [KQUAD(len=RING[1].len, k1=k), KQUAD(len=RING[3].len, k1=-k), KQUAD(len=RING[5].len, k1=k), KQUAD(len=RING[7].len, k1=k),
                KQUAD(len=RING[9].len, k1=-k), KQUAD(len=RING[11].len, k1=k), KQUAD(len=RING[13].len, k1=k),KQUAD(len=RING[15].len, k1=-k),
                KQUAD(len=RING[17].len, k1=k), KQUAD(len=RING[19].len, k1=k), KQUAD(len=RING[21].len, k1=-k),KQUAD(len=RING[23].len, k1=k)] 
        # changed_ele = fill(KQUAD(len=0.6, k1=k),length(changed_idx))
        m66 = ADfindm66(RING, 0.0, 3, changed_idx, changed_ele)
        trace = m66[3, 3] + m66[4, 4]
        return trace
end

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

function relu_rev(x::Float64)
        if x > 0. 
                return 0.
        else
                return -x
        end
end

function relu(x::Float64, y::Float64)
        if x > y
                return x
        else
                return 0.
        end
end

function relu_exact(x::Float64, y::Float64)

end

function ini_opt_loss(θ::Float64, k::Float64, b::Float64, RING::Vector{AbstractElement})
        # return (2.0-abs(trace_updated_focus_quad_strengthx(kp(θ, k, b), RING)))^2 + (2.0-abs(trace_updated_focus_quad_strengthy(kp(θ, k, b), RING)))^2 + 10*relu_rev(kp(θ, k, b))^2
        return  relu(abs(trace_updated_focus_quad_strengthx(kp(θ, k, b), RING)),2.0) + relu(abs(trace_updated_focus_quad_strengthy(kp(θ, k, b), RING)),2.0) + exp(50*relu_rev(kp(θ, k, b))) - 1.0

        # return (0.0-abs(trace_updated_focus_quad_strengthx(kp(θ, k, b), RING)))^2 + (0.0-abs(trace_updated_focus_quad_strengthy(kp(θ, k, b), RING)))^2 + relu(abs(trace_updated_focus_quad_strengthx(kp(θ, k, b), RING)),2.0) + relu(abs(trace_updated_focus_quad_strengthy(kp(θ, k, b), RING)),2.0) + exp(relu_rev(kp(θ, k, b))) - 1.0
        # return exp(2.0*(-2.0+abs(trace_updated_focus_quad_strengthx(kp(θ, k, b), RING)))) + exp(2.0*(-2.0+abs(trace_updated_focus_quad_strengthy(kp(θ, k, b), RING))))
        # return ((trace_updated_focus_quad_strengthx(k, RING)-trace_updated_focus_quad_strengthx(kp(θ, k, b), RING))^2 + (trace_updated_focus_quad_strengthy(k, RING)-trace_updated_focus_quad_strengthy(kp(θ, k, b), RING))^2)
end

function order_magntiude(x::Float64)
        return floor(log10(abs(x)))
end

function opt_ini(k::Float64, RING::Vector{AbstractElement}, iter::Int64, step::Float64)
        # new_k_vals = zeros(iter)
        # tracex_vals = zeros(iter)
        # tracey_vals = zeros(iter)
        counter = 1
        m66 = ADfindm66(RING, 0.0, 3, [], [])
        tracex = m66[1, 1] + m66[2, 2]
        tracey = m66[3, 3] + m66[4, 4]
        # θnew = θ
        # bnew = b
        # tracex_new = 3.
        # tracey_new = 3.
        # push!(new_k_vals, kp(θ,k,b))
        # push!(tracex_vals,tracex)
        # push!(tracey_vals, tracey)
        # new_k_vals[1] = kp(θ,k,b)
        # tracex_vals[1] = tracex
        # tracey_vals[1] = tracey 
        mom_factor = step/1.618
        other_fac = step - mom_factor
        mom1 = 0.
        mom2 = 0.
        θ = 1.0+0.01*rand()
        b = 0.01*rand()
        while abs(tracex) > 2. || abs(tracey) > 2.
        # while ini_opt_loss(θ, k, b, RING) > 1e-2
                grad1 = autodiff(Forward, ini_opt_loss, Duplicated, Duplicated(θ,1.0), Const(k), Const(b), Const(RING))
                grad2 = autodiff(Forward, ini_opt_loss, Duplicated, Const(θ), Const(k), Duplicated(b, 1.0), Const(RING))
                mom1 = mom_factor*mom1 + other_fac*grad1[1]
                mom2 = mom_factor*mom2 + other_fac*grad2[1]
                θ -= step * mom1
                b -= step * mom2
                # θ -= step*grad1[1]
                # b -= b - step*grad2[1]
                tracex = trace_updated_focus_quad_strengthx(kp(θ,k,b), RING)
                tracey = trace_updated_focus_quad_strengthy(kp(θ,k,b), RING) 
                if isnan(tracex) || isnan(tracey) 
                        println("Nan detected in traces")
                        # θ = rand()
                        # b = rand()
                        # tracex = 300.
                        # tracey = 300.
                        break
                end
                # new_k_vals[counter] = kp(θ,k,b)
                # tracex_vals[counter] = tracex
                # tracey_vals[counter] = tracey
                
                println("Tuning at step: ", counter, " at k: ", kp(θ,k,b), " with trace x: ", tracex, " and trace y: ", tracey)
                if counter > iter 
                        break
                end
                # loss_val = ini_opt_loss(θ, k, b, RING)
                # println("Loss: ",loss_val)
                
                counter +=1
        end  
        println("Tuning finished at step: ", counter-1, " at k: ", kp(θ,k,b), " with trace x: ", tracex, " and trace y: ", tracey)  
        return kp(θ,k,b), tracex, tracey
end




kstart = 2.2
niter = 300
step = 1e-3
θini = 1.0+0.1*rand()
bini = 0.1*rand()

order = 3
dp = 0.0
x = CTPS(0.0, 1, 6, order)
px = CTPS(0.0, 2, 6, order)
y = CTPS(0.0, 3, 6, order)
py = CTPS(0.0, 4, 6, order)
z = CTPS(dp, 5, 6, order)
delta = CTPS(0.0, 6, 6, order)
rin = [x, px, y, py, z, delta]

RING = ring_gen_ini(kstart)
k = kp(θini,kstart,bini)
@btime ADfindm66(RING, 0.0, 3, [], [])
@btime trace_updated_focus_quad_strengthx(k, RING)



# kini_vals_ini, tracexini_vals, traceyini_vals = opt_ini(θini, kstart, bini, ring_gen_ini(kp(θini, kstart, bini)), niter, step)
# kini, tracexini, traceyini = opt_ini(θini, kstart, bini, ring_gen_ini(kstart), niter, step)
# @btime opt_ini(kstart,ring_gen_ini(kstart), niter, step)
# opt_ini(kstart,ring_gen_ini(kstart), niter, step)

# println("Initial tuning finished at k1: ", kini, " with trace x: ", tracexini, " and trace y: ", traceyini[end])
# kstart = kini_vals_ini[end]
# k=zeros(12)
# θ = rand(Float64, 12)
# # b = rand(Float64, 12)
# b = zeros(12)
# k[1:8] .= kstart
# k[9:12] .= -kstart
# ring = ring_gen(k)
# grads= zeros(12)

# counter = 0

# # total_loss(θ, k, b, ring)
# w = fill(1.0, 10)
# function optimize_(θ, k, b, ring)
#         counter = 0
#         while total_loss(θ, k, b, ring) > 1e-2 || counter < 100
#                 diffmatrix = zeros(12, 12)
#                 for i in 1:12
#                         diffmatrix[i,i] = 1.0
                        
#                         grads[i] = try 
#                                 autodiff(Forward, total_loss, Duplicated, Duplicated(θ[i], diffmatrix[:,i]), Const(k), Const(b), Const(ring))
                                
#                         catch e
#                                 if isa(e, ErrorException)
#                                         println("Trying again caught")
#                                 else
#                                         println("idk")
#                                 end
#                                 break
#                                 # counter += 1
#                                 # println("Trying again")
#                                 # return optimize_(θ, k, b, ring)
#                                 # break
#                         end
#                 end
#                 # grads[1] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[2] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[3] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[4] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[5] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[6] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[7] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[8] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[9] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[10] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[11] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 # grads[12] = autodiff(Forward, total_loss, Duplicated, Duplicated(θ, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]), Const(k), Const(b), Const(ring_gen(k)))
#                 θ .-= 0.1 .* grads
#                 counter += 1
#                 println(counter)
#         end

#         println("new k: ", kp(θ, k, b))

# end

# optimize_(θ, k, b, ring)

# βxmax(k, ring)
# total_loss(θ, k, b, ring)
