function matrix_to_array(matrix::Matrix{Float64})
    # particles = zeros(Float64, size(matrix, 1)*size(matrix, 2))
    # for i in 1:size(matrix, 1)
    #     for j in 1:size(matrix, 2)
    #         particles[(i-1)*size(matrix, 2)+j] = matrix[i, j]
    #     end
    # end
    # particles = vec(matrix)

    return vec(matrix)
end

function array_to_matrix(array::Vector{Float64}, n::Int)
    # particles = zeros(Float64, n, 6)
    # for i in 1:n
    #     for j in 1:6
    #         particles[i, j] = array[(i-1)*6+j]
    #     end
    # end
    # particles = reshape(array, n, 6)
    return reshape(array, n, 6)
end

function plinepass!(line, particles::Beam)
    # Note!!! A lost particle's coordinate will not be marked as NaN or Inf like other softwares 
    # Check if the particle is lost by checking the lost_flag
    np = particles.nmacro
    particles6 = matrix_to_array(particles.r)
    if length(particles6) != np*6
        error("The number of particles does not match the length of the particle array")
    end
    for i in eachindex(line)
        # ele = line[i]
        pass_P!(line[i], particles6, np, particles)   
        if isnan(particles6[1]) || isinf(particles6[1])
            println("The particle is lost at element ", i, "element name is ", line[i].name)
            # rout = array_to_matrix(particles6, np)
            # particles.r = rout
            particles.r = array_to_matrix(particles6, np)
            return nothing
        end     
    end
    # rout = array_to_matrix(particles6, np)
    # particles.r = rout
    particles.r = array_to_matrix(particles6, np)
    return nothing
end

# function plinepass!(line, particles::Beam) #this is copy of plinepass above, without the np variable created. No effect now, may have effect with multielement
#     # Note!!! A lost particle's coordinate will not be marked as NaN or Inf like other softwares 
#     # Check if the particle is lost by checking the lost_flag
#     # np = particles.nmacro
#     particles6 = matrix_to_array(particles.r)
#     if length(particles6) != particles.nmacro*6
#         error("The number of particles does not match the length of the particle array")
#     end
#     for i in eachindex(line)
#         # ele = line[i]
#         pass_P!(line[i],particles6, particles.nmacro, particles)
#         if isnan(particles6[1]) || isinf(particles6[1])
#             println("The particle is lost at element ", i, "element name is ", line[i].name)
#             particles.r = array_to_matrix(particles6,  particles.nmacro)
#             return nothing
#         end        
#     end
#     particles.r = array_to_matrix(particles6,  particles.nmacro)
#     return nothing
# end

function pADlinepass!(line, particles::Beam, changed_idx::Vector{Int}, changed_ele)
    # Note!!! A lost particle's coordinate will not be marked as NaN or Inf like other softwares 
    # Check if the particle is lost by checking the lost_flag
    np = particles.nmacro
    particles6 = matrix_to_array(particles.r)
    if length(particles6) != np*6
        error("The number of particles does not match the length of the particle array")
    end
    count = 1
    for i in eachindex(line)
        # ele = line[i]
        if i in changed_idx
            pass_P!(changed_ele[count], particles6, np, particles)
            count += 1
        else
            pass_P!(line[i], particles6, np, particles)        
        end
        if isnan(particles6[1]) || isinf(particles6[1])
            println("The particle is lost at element ", i, "element name is ", line[i].name)
            rout = array_to_matrix(particles6, np)
            particles.r = rout
            return nothing
        end
    end
    rout = array_to_matrix(particles6, np)
    particles.r = rout
    return nothing
end

function pringpass!(line::Vector{AbstractElement}, particles::Beam, nturn::Int)
    # Note!!! A lost particle's coordinate will not be marked as NaN or Inf like other softwares 
    # Check if the particle is lost by checking the lost_flag
    for i in 1:nturn
        plinepass!(line, particles)    
    end
    return nothing
end


