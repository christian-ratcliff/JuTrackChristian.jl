include("src/JuTrack.jl")
using .JuTrack
using Enzyme
using Plots
using Random

function ring_gen(kf, kd)
        D1 = DRIFT(name="D1", len=0.7)
        QF1 = KQUAD(name="QF1", len=0.6, k1=kf)
        QD1 = KQUAD(name="QD1", len=0.6, k1=kd)
        BendingAngle = pi/2.
        BD1 = SBEND(name="BD1", len=2.1, angle=BendingAngle )

        ringout = [QF1, D1, QD1, D1, QF1, BD1,
                QF1, D1, QD1, D1, QF1, BD1,
                QF1, D1, QD1, D1, QF1, BD1,
                QF1, D1, QD1, D1, QF1, BD1]

        return ringout
end

function trace_updated_focus_quad_strength(k, ring)
        # changed_idx = [1, 5, 8, 12, 15, 19, 22, 26]
        changed_idx = [1, 5, 7, 11, 13, 17, 19, 23]
        changed_ele = [KQUAD(len=0.6, k1=k), KQUAD(len=0.6, k1=k), 
                        KQUAD(len=0.6, k1=k), KQUAD(len=0.6, k1=k), 
                        KQUAD(len=0.6, k1=k), KQUAD(len=0.6, k1=k),
                        KQUAD(len=0.6, k1=k), KQUAD(len=0.6, k1=k)]
        # changed_ele = fill(KQUAD(len=0.6, k1=k),length(changed_idx))
        m66 = ADfindm66(ring, 0.0, 3, changed_idx, changed_ele)
        trace = m66[1, 1] + m66[2, 2]
        return trace
end

function get_tune(k_vals,RING)
        changed_idx = [1, 5, 7, 11, 13, 17, 19, 23]
        new_QF1 = KQUAD(len=RING[1].len, k1=k_vals[1])
        new_QF2 = KQUAD(len=RING[5].len, k1=k_vals[2])
        new_QF3 = KQUAD(len=RING[7].len, k1=k_vals[3])
        new_QF4 = KQUAD(len=RING[11].len, k1=k_vals[4])
        new_QF5 = KQUAD(len=RING[13].len, k1=k_vals[5])
        new_QF6 = KQUAD(len=RING[17].len, k1=k_vals[6])
        new_QF7 = KQUAD(len=RING[19].len, k1=k_vals[7])
        new_QF8 = KQUAD(len=RING[23].len, k1=k_vals[8])
        changed_ele = [new_QF1, new_QF2, new_QF3, new_QF4, new_QF5, new_QF6, new_QF7, new_QF8]
        refpts = [i for i in 1:length(RING)]
        twi = ADtwissring(RING, 0.0, 1, refpts, changed_idx, changed_ele)
        phase = twi[end].dmux - twi[1].dmux
        tune = phase/(2*π)
        return tune
end

function optimize_tune(k, iter, stepsize, RING)
        target = 0.53
        k_vals = zeros(8, iter)
        goal_vals = []
        grad_vals = zeros(8, iter)
        changed_idx = [1, 5, 7, 11, 13, 17, 19, 23]
        g0 = get_tune(k, RING)

        for i in 1:iter
                grad1 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(RING))
                grad5 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]), Const(RING))
                grad7 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]), Const(RING))
                grad11 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]), Const(RING))
                grad13 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0]), Const(RING))
                grad17 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0]), Const(RING))
                grad19 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0]), Const(RING))
                grad23 = autodiff(Forward, get_tune, Duplicated, Duplicated(k, [0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]), Const(RING))
        
                k[1] .-= step .* grad1[2]
                k[2] .-= step .* grad5[2]
                k[3] .-= step .* grad7[2]
                k[4] .-= step .* grad11[2]
                k[5] .-= step .* grad13[2]
                k[6] .-= step .* grad17[2]
                k[7] .-= step .* grad19[2]
                k[8] .-= step .* grad23[2]

                new_tune = get_tune(k, RING)
                println("init: ", g0, " now: ", new_tune, "at step ", i)
                k_vals[:, i] = k
                push!(goal_vals, new_tune)
                grad_vals[:, i] = [grad1[2], grad5[2], grad7[2], grad11[2], grad13[2], grad17[2], grad19[2], grad23[2]]
                if abs(new_tune) < target 
                        println("Tune tuning finished at step ", i)
                        break
                end
                if i==iter && abs(new_tune) > target
                        println("Tuning tune did not converge in number of iterations")
                end
        end
        return k_vals, goal_vals, grad_vals
end



function optimize_k(k, iter, stepsize)
        k_vals = Float64[]
        trace_vals = Float64[]
        grad_vals = Float64[]
        for i in 1:iter
                ring = ring_gen(k, -k)
                trace0 = trace_updated_focus_quad_strength(k, ring)
                grad = autodiff(Forward, trace_updated_focus_quad_strength, DuplicatedNoNeed, Duplicated(k, 1.0),  Const(ring))
                if trace0 > 2
                        k -= stepsize * grad[1]
                else
                        k += stepsize * grad[1]
                end
                ring = ring_gen(k, -k)
                trace1 = trace_updated_focus_quad_strength(k, ring)
                # println("Trace1: ", trace1, " grad: ", grad, "at step ", i, " k1 = ", k)
                push!(k_vals, k)
                push!(trace_vals, trace1)
                push!(grad_vals, grad[1])
                if abs(trace1) < 2.0
                        println("Principal tuning finished at step ", i," at k1: ",k, " with trace1: ", trace1, " and target: ", 2.0)
                        break
                end
                if i==iter && abs(trace1) > 2.0
                        println("Principal tuning did not converge in number of iterations")
                end
        end
        return k_vals, trace_vals, grad_vals
end



function Q_perturb(RING)
        for i in eachindex(RING)
                if RING[i] isa KQUAD
                        k1 = RING[i].k1
                        k1 = k1 * (1 + 0.001 * randn())
                        new_KQ = KQUAD(name=RING[i].name, len=RING[i].len, k1=k1)
                        RING[i] = new_KQ
                end
        end
        return RING
end


kstart = 1.2
# ring = ring_gen(kstart, -2*kstart)

niter = 100
step = 0.01

kini_vals, traceini_vals, grad_vals = optimize_k(kstart, niter, step)
ring = ring_gen(kini_vals[end], -kini_vals[end])
m66 = ADfindm66(ring, 0.0, 3, [], [])
print(m66)
println("k1: ", kini_vals[end], " trace: ", traceini_vals[end])
# ring = ring_gen(kini_vals[end], -kini_vals[end])

# # m66 = ADfindm66(ring, 0.0, 3, [], [])
# # trace = m66[1, 1] + m66[2, 2]
# # println(trace)
# # println("Focus k1: ", x0[1], " Defoucs k1: ", x0[2], " Trace: ", trace_vals[end])
# # idxs = [1, 5, 7, 11, 13, 17, 19, 23]
# ring_perturb = Q_perturb(ring)
# kinit = [ring_perturb[1].k1; ring_perturb[5].k1; ring_perturb[7].k1; ring_perturb[11].k1; ring_perturb[13].k1; ring_perturb[17].k1; ring_perturb[19].k1; ring_perturb[23].k1]

# # println(kinit)
# k_vals, goal_vals, grad_vals = optimize_tune(kinit, niter, step, ring)

# plot_steps = 5
# p1 = plot(1:plot_steps, k_vals[1, 1:plot_steps], title = L"Evolution\ of\ k", xlabel = L"Iterations", ylabel = L"Strength (m^{-1})", label=L"k_1", line=:dash, marker=:circle)
# for i in 2:7
#         label_str = "k_{$i}"  
#         full_label = latexstring(label_str)
#         plot!(1:plot_steps, k_vals[i, 1:plot_steps], label=full_label, line=:dash, marker=:circle)
# end
# p2 = plot(1:plot_steps, goal_vals[1:plot_steps], title = L"Evolution\ of\ \Delta \phi", xlabel = L"Iterations", ylabel = L"phase\ advance(rad)", legend = false, line=:dash, marker=:circle)
# p3 = plot(1:plot_steps, grad_vals[1, 1:plot_steps], title = L"Evolution\ of\ gradient", xlabel = L"Iterations", ylabel = L"\partial \frac{\Delta \phi}{\partial k}", label=L"k_1", line=:dash, marker=:circle)
# for i in 2:7
#         label_str = "k_{$i}"  
#         full_label = latexstring(label_str)
#         plot!(1:plot_steps, grad_vals[i, 1:plot_steps], label=full_label, line=:dash, marker=:circle)
# end
# plot(p1, p2, p3, layout = (3, 1), size=(800, 650))







