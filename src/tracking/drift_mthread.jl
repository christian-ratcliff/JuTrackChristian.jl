function ATmultmv!(r::AbstractVector{Float64}, A::Matrix{Float64})
    # multiplies 6-component column vector r by 6x6 matrix R: as in A*r
    temp = zeros(6)
    for i in 1:6
        for j in 1:6
            temp[i] += A[i, j] * r[j]
        end
    end
    for i in 1:6
        r[i] = temp[i]
    end
    return nothing
end

function ATaddvv!(r::AbstractVector{Float64}, dr::Array{Float64,1})
    for i in 1:6
        r[i] += dr[i]
    end
    return nothing
end

function fastdrift!(r::AbstractVector{Float64}, NormL::Float64, le::Float64)
    # NormL is computed externally to speed up calculations
    # in the loop if momentum deviation (delta) does not change
    # such as in 4-th order symplectic integrator w/o radiation
    # AT uses small angle approximation pz = 1 + delta. 
    # Here we use pz = sqrt((1 + delta)^2 - px^2 - py^2) for precise calculation
    r[1] += NormL * r[2]
    r[3] += NormL * r[4]
    # r[6] += NormL * (r[2]^2 + r[4]^2) / (2*(1+r[5]))
    r[5] += NormL * (1.0 + r[6]) - le
    return nothing
end

function drift6!(r::AbstractVector{Float64}, le::Float64)
    # AT uses small angle approximation pz = 1 + delta. 
    # Here we use pz = sqrt((1 + delta)^2 - px^2 - py^2) for precise calculation
    NormL = le / sqrt(((1.0 + r[6])^2 - r[2]^2 - r[4]^2))
    r[1] += NormL * r[2]
    r[3] += NormL * r[4]
    # r[6] += NormL * (r[2]^2 + r[4]^2) / (2*(1+r[5])) # for linearized approximation
    r[5] += NormL * (1.0 + r[6]) - le
    return nothing
end 
function DriftPass_P!(r_in::Array{Float64,1}, le::Float64, T1::Array{Float64,1}, T2::Array{Float64,1}, 
    R1::Array{Float64,2}, R2::Array{Float64, 2}, RApertures::Array{Float64,1}, EApertures::Array{Float64,1}, 
    num_particles::Int, lost_flags::Array{Int64,1}, noTarray::Array{Float64,1}, noRmatrix::Array{Float64,2})
    Threads.@threads for c in 1:num_particles
    # for c in 1:num_particles
        if lost_flags[c] == 1
            continue
        end
        r6 = @view r_in[(c-1)*6+1:c*6]
        if !isnan(r6[1])
            # Misalignment at entrance
            if T1 != noTarray
                ATaddvv!(r6, T1)
            end
            if R1 != noRmatrix
                ATmultmv!(r6, R1)
            end
            # Check physical apertures at the entrance of the magnet
            # if RApertures !== nothing
            #     checkiflostRectangularAp!(r6, RApertures)
            # end
            # if EApertures !== nothing
            #     checkiflostEllipticalAp!(r6, EApertures)
            # end
            drift6!(r6, le)
            # Check physical apertures at the exit of the magnet
            # if RApertures !== nothing
            #     checkiflostRectangularAp!(r6, RApertures)
            # end
            # if EApertures !== nothing
            #     checkiflostEllipticalAp!(r6, EApertures)
            # end
            # Misalignment at exit
            if R2 != noRmatrix
                ATmultmv!(r6, R2)
            end
            if T2 != noTarray
                ATaddvv!(r6, T2)
            end
            if r6[1] > CoordLimit || r6[2] > AngleLimit || r6[1] < -CoordLimit || r6[2] < -AngleLimit
                lost_flags[c] = 1
            end
        end
    end
    return nothing
end

function pass_P!(ele::DRIFT, r_in::Array{Float64,1}, num_particles::Int64, particles::Beam, noTarray::Array{Float64,1}, noRmatrix::Array{Float64,2})
    # ele: EDRIFT
    # r_in: 6-by-num_particles array
    # num_particles: number of particles
    lost_flags = particles.lost_flag
    DriftPass_P!(r_in, ele.len, ele.T1, ele.T2, ele.R1, ele.R2, ele.RApertures, ele.EApertures, num_particles, lost_flags, noTarray, noRmatrix)
    return nothing
end

# function matrix_to_array(matrix::Matrix{Float64})
#     particles = zeros(Float64, size(matrix, 1)*size(matrix, 2))
#     for i in 1:size(matrix, 1)
#         for j in 1:size(matrix, 2)
#             particles[(i-1)*size(matrix, 2)+j] = matrix[i, j]
#         end
#     end
#     return particles
# end
# r = zeros(Float64, 1000000, 6)
# r[:, 2] .= 0.1
# r_arr = matrix_to_array(r)
# beam = Beam(r)
# D = DRIFT(len=3.0)
# # pass_P!(D, r_arr, beam.nmacro, beam, zeros(Float64, 6), zeros(Float64, 6, 6))
# # println(r_arr[1:6])
# # println(Threads.nthreads())
# # @btime begin    
# #     pass_P!(D, r_arr, beam.nmacro, beam, zeros(Float64, 6), zeros(Float64, 6, 6))
# # end

# function f_multithreading(x1, x2)
#     r = zeros(Float64, 1000000, 6)
#     r[:, 2] .= 0.1
#     r_arr = matrix_to_array(r)
#     beam = Beam(r)
#     D1 = DRIFT(len=x1)
#     D2 = DRIFT(len=x2)
#     pass_P!(D1, r_arr, beam.nmacro, beam, zeros(Float64, 6), zeros(Float64, 6, 6))
#     pass_P!(D2, r_arr, beam.nmacro, beam, zeros(Float64, 6), zeros(Float64, 6, 6))
#     return r_arr[1]
# end
# println(f_multithreading(3.0, 1.0))
# # grad = gradient(Forward, f_multithreading, [3.0, 1.0])
# grad = autodiff(Forward, f_multithreading,  Duplicated, Duplicated(3.0, 1.0), Duplicated(1.0, 1.0))
# @time grad = autodiff(Forward, f_multithreading, Duplicated, Duplicated(3.0, 1.0), Duplicated(1.0, 1.0))
# println(grad)
