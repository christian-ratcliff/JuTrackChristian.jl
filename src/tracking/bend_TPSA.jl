# include("drift_TPSA.jl")
include("drift_TPSA.jl")
include("fringe_TPSA.jl")

function bndthinkick!(r::Vector{CTPS{T, TPS_Dim, Max_TPS_Degree}}, A, B, L, irho, max_order) where {T, TPS_Dim, Max_TPS_Degree}
    # Calculate multipole kick in a curved elemrnt (bending magnet)
    # The reference coordinate system  has the curvature given by the inverse 
    # (design) radius irho.
    # IMPORTANT !!!
    # The magnetic field Bo that provides this curvature MUST NOT be included in the dipole term
    # PolynomB[1](MATLAB notation)(C: B[0] in this function) of the By field expansion
    
    # The kick is given by
    
    #            e L      L delta      L x
    # theta  = - --- B  + -------  -  -----  , 
    #      x     p    y     rho           2
    #             0                    rho
    
    #          e L
    # theta  = --- B
    #      y    p   x
    #            0
        ### this section is type unstable
        # ReSum = B[max_order + 1]
        # ImSum = A[max_order + 1]
        # ReSumTemp = 0.0
    
        # i = 1
        ReSumTemp = B[max_order + 1] * r[1] - A[max_order + 1] * r[3] + B[1]
        ImSum = A[max_order + 1] * r[1] + B[max_order + 1] * r[3] + A[1]
        ReSum = CTPS(ReSumTemp)
        for i in reverse(2:max_order)
            ReSumTemp = ReSum * r[1] - ImSum * r[3] + B[i]
            ImSum = ImSum * r[1] + ReSum * r[3] + A[i]
            ReSum = CTPS(ReSumTemp)
        end
        # ReSumTemp = tadd(tminus(tmult(B[max_order + 1], r[1]), tmult(A[max_order + 1], r[3])), B[1])
        # ImSum = tadd(tadd(tmult(A[max_order + 1], r[1]), tmult(B[max_order + 1], r[3])), A[1])
        # ReSum = CTPS(ReSumTemp)
        # for i in reverse(2:max_order)
        #     ReSumTemp = tadd(tminus(tmult(ReSum, r[1]), tmult(ImSum, r[3])), B[i])
        #     ImSum = tadd(tadd(tmult(ImSum, r[1]), tmult(ReSum, r[3])), A[i])
        #     ReSum = CTPS(ReSumTemp)
        # end
    
        # r[2] = tminus(r[2], tmult(L, tminus(ReSum, tmult(tminus(r[5],tmult(r[1], irho)), irho))))
        # r[4] = tadd(r[4], tmult(L, ImSum))
        # r[6] = tadd(r[6], tmult(L, tmult(r[1], irho)))
        r[2] -= L * (ReSum - (r[5] - r[1] * irho) * irho)
        r[4] += L * ImSum
        r[6] += L * irho * r[1]  # Path length
        return nothing
    end
    
    function BndMPoleSymplectic4Pass!(r::Vector{CTPS{T, TPS_Dim, Max_TPS_Degree}}, le, irho, A, B, max_order, num_int_steps,
        entrance_angle, exit_angle,
        FringeBendEntrance, FringeBendExit,
        fint1, fint2, gap,
        FringeQuadEntrance, FringeQuadExit,
        fringeIntM0, fringeIntP0,
        T1, T2, R1, R2, RApertures, EApertures,
        KickAngle, num_particles) where {T, TPS_Dim, Max_TPS_Degree}
        
        DRIFT1 = 0.6756035959798286638
        DRIFT2 = -0.1756035959798286639
        KICK1 = 1.351207191959657328
        KICK2 = -1.702414383919314656
        SL = le / num_int_steps
        L1 = SL * DRIFT1
        L2 = SL * DRIFT2
        K1 = SL * KICK1
        K2 = SL * KICK2
    
        if FringeQuadEntrance==2 && !isnothing(fringeIntM0) && !isnothing(fringeIntP0)
            useLinFrEleEntrance = 1
        else
            useLinFrEleEntrance = 0
        end
        if FringeQuadExit==2 && !isnothing(fringeIntM0) && !isnothing(fringeIntP0)
            useLinFrEleExit = 1
        else
            useLinFrEleExit = 0
        end
    
        B[1] -= sin(KickAngle[1]) / le
        A[1] += sin(KickAngle[2]) / le
    
    
        # Threads.@threads for c in 1:num_particles
        for c in 1:num_particles
        #     r6 = @view r[(c-1)*6+1:c*6]
            # if !isnan(r6[1])
                p_norm = 1.0 / (1.0 + r[5])
                NormL1 = L1 * p_norm
                NormL2 = L2 * p_norm
    
                # Misalignment at entrance
                if isnothing(T1)
                    ATaddvv!(r, T1)
                end
                if isnothing(R1)
                    ATmultmv!(r, R1)
                end
    
                # Edge focus at entrance
                edge_fringe_entrance!(r, irho, entrance_angle, fint1, gap, FringeBendEntrance)
    
                # Quadrupole gradient fringe entrance
                # if FringeQuadEntrance != 0 && B[2] != 0
                #     if useLinFrEleEntrance == 1
                #         linearQuadFringeElegantEntrance!(r6, B[2], fringeIntM0, fringeIntP0)
                #     else
                #         QuadFringePassP!(r6, B[2])
                #     end
                # end
    
                # Integrator
                for m in 1:num_int_steps
                    fastdrift!(r, NormL1)
                    bndthinkick!(r, A, B, K1, irho, max_order)
                    fastdrift!(r, NormL2)
                    bndthinkick!(r, A, B, K2, irho, max_order)
                    fastdrift!(r, NormL2)
                    bndthinkick!(r, A, B, K1, irho, max_order)
                    fastdrift!(r, NormL1)
                end
    
                # Quadrupole gradient fringe exit
                # if FringeQuadExit != 0 && B[2] != 0
                #     if useLinFrEleExit == 1
                #         linearQuadFringeElegantExit!(r6, B[2], fringeIntM0, fringeIntP0)
                #     else
                #         QuadFringePassN!(r6, B[2])
                #     end
                # end
    
                # Edge focus at exit
                edge_fringe_exit!(r, irho, exit_angle, fint2, gap, FringeBendExit)
    
    
                # Misalignment at exit
                if R2 !== nothing
                    ATmultmv!(r, R2)
                end
                if T2 !== nothing
                    ATaddvv!(r, T2)
                end
            # end
        end
    
        
        B[1] += sin(KickAngle[1]) / le
        A[1] -= sin(KickAngle[2]) / le
        return nothing
    end
    
    
    function pass_TPSA!(ele::SBEND, r_in::Vector{CTPS{T, TPS_Dim, Max_TPS_Degree}}, num_particles::Int64) where {T, TPS_Dim, Max_TPS_Degree}
        # ele: SBEND
        # r_in: 6-by-num_particles array
        # num_particles: number of particles
    
        irho = ele.angle / ele.len
        BndMPoleSymplectic4Pass!(r_in, ele.len, irho, ele.PolynomA, ele.PolynomB, ele.MaxOrder, ele.NumIntSteps,
            ele.e1, ele.e2,
            ele.FringeBendEntrance, ele.FringeBendExit,
            ele.fint1, ele.fint2, ele.gap,
            ele.FringeQuadEntrance, ele.FringeQuadExit,
            ele.FringeIntM0, ele.FringeIntP0,
            ele.T1, ele.T2, ele.R1, ele.R2, ele.RApertures, ele.EApertures,
            ele.KickAngle, num_particles)
        return nothing
    end
# using BenchmarkTools
# using Enzyme
# include("../lattice/canonical_elements_AT.jl")
# # include("../TPSA_Enzyme/arrayTPSA_fixedmap.jl")
# include("../TPSA_Enzyme/TPSA_fixedmap.jl")
# # # # q = KQUAD(PolynomialB=[0.0, 1.0, 0.0, 0.0])
# function f(xx)
#     q = SBEND(angle=xx[1], len=1.0)
#     x = CTPS(0.0, 1, 6, 3)
#     xp = CTPS(0.0, 2, 6, 3)
#     y = CTPS(0.0, 3, 6, 3)
#     yp = CTPS(0.0, 4, 6, 3)
#     z = CTPS(0.0, 5, 6, 3)
#     delta = CTPS(0.0, 6, 6, 3)
#     rin = [x, xp, y, yp, z, delta]
#     pass_TPSA!(q, rin, 1)
# return rin[1].map[2]
# end
# x = [0.1573]
# println(f(x))
# @btime grad = gradient(Forward, f, x)
# println(grad)