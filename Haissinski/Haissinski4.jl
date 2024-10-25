using Distributions
using Random
using Plots
using DSP
using LaTeXStrings
using Interpolations
using Roots
using BenchmarkTools
Random.seed!(6111)

const SPEED_LIGHT = 299792458
const ELECTRON_CHARGE = 1.602176634e-19


function centered(arr, input_arr_shape)
    # new_arr_shape is the size of the input array during convolution with DSP.conv(input_arr1, input_arr2)
    # Get the current shape of the array i.e. the result of DSP.conv()
    currshape = size(arr)
    
    # Calculate start and end indices for slicing
    startind = (currshape .- input_arr_shape) .÷ 2 .+ 1
    endind = startind .+ input_arr_shape .- 1
    
    # Create the slice objects
    slices = Tuple(startind[i]:endind[i] for i in 1:length(endind))
    
    # Return the centered portion of the array
    return arr[slices...]
end

function fangle(ϕu, ;ϕs_var = ϕs)
    return (-cos(ϕu) - cos(ϕs_var) + sin(ϕs_var) * (π - ϕu - ϕs_var)) 
end

function apply_mask_to_innermost(obj, mask)
    return [ inner_array[mask] for inner_array in obj ]
end


function longitudinal_evolve_2(n_turns::Int64, particles::Vector{Array}, sin_ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64, harmonic::Int64,acc_radius::Float64, freq_rf::Float64, pipe_radius::Float64, E0::Float64, σ_E::Float64,update_η::Bool = false, update_E0::Bool = false, SR_damping::Bool = false, use_excitation::Bool = false, use_wakefield::Bool = false)

    z = map(x->x[1], particles)
    ΔE = map(x->x[2], particles)
    γ0 = E0 / mass
    β0 = sqrt(1 - 1 /γ0 ^2)
    eV = voltage
    ϕ = similar(ΔE)
    ϕ .= 0
    WF = similar(ΔE)
    WF .= 0
    a = pipe_radius
    potential = Array[]

    master_storage = Array{Vector{Array}}(undef, n_turns)
    particles_end = Array[]

    ϕ .= .- (z * (freq_rf * 2 * π) / (β0 * SPEED_LIGHT) .- asin(sin_ϕs))
    push!(particles_end, [z, ϕ, ΔE])
    for i in 1:n_turns
        WF = similar(ΔE)
        WF .= 0
        ΔE .+= eV * (sin.(ϕ) .- sin_ϕs)

        if SR_damping == true
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
            ΔE .-= ∂U_∂E * ΔE 
        end

        if use_excitation == true
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
            excitation = sqrt(1-∂U_∂E^2)
        else
            excitation = 0
        end

        ΔE .+= excitation * rand(Normal(0, 1), length(z)) * σ_E
        if use_wakefield == true
            z .= (.-ϕ .+ asin(sin_ϕs)) * β0 * SPEED_LIGHT / (freq_rf*2 *π)
            kp = 3e1
            Z0 = 120 * π
            cτ = 4e-3
            zmask = z .< 0
            WF .= 0
            WF[zmask] .= (Z0 * SPEED_LIGHT / (π * a^2)) * exp.(z[zmask] / cτ) .* cos.(sqrt(2*kp/a) * .-z[zmask])
            potential = conv(WF, z) * (length(particles) * ELECTRON_CHARGE) / (E0 * 2 * π * radius) * 1e7 #* normcurr#* (length(particles) * ELECTRON_CHARGE^2) / (E0 * 2 * π * radius)
            potential_centered = centered(potential, size(z))
            potential .= 0
            potential = potential_centered
            combined = collect(zip(z, potential_centered))
            sorted_combined = sort(combined, by = x -> x[1])
            sorted_array1 = [x[1] for x in sorted_combined]
            sorted_array2 = [x[2] for x in sorted_combined]

            itp = LinearInterpolation(sorted_array1, sorted_array2,extrapolation_bc=Flat())
            
            ΔE .-= float(itp.(sorted_array1))
        end
        
        
        if update_E0 == true
            E0 += eV * sin_ϕs
            # E0 = mean(ΔE)
        end

        γ0 = E0/mass 
        β0 = sqrt(1 - 1/γ0^2)

        Δγ = similar(ΔE)
        Δγ .= 0
        η = similar(ΔE)
        η .= 0
        coeff = similar(ΔE)
        if update_η == true
            Δγ .= ΔE ./mass
        else
            Δγ .= 0
        end
        η .= α_c .- 1 ./(γ0 .+ Δγ).^2

        coeff .= 2 * π * harmonic * η / (β0 * β0 * E0)
        ϕ .+= coeff .* ΔE 
        z .= (.-ϕ .+ asin(sin_ϕs)) * β0 * SPEED_LIGHT / (freq_rf*2 *π)

        ϕ_mask = Array[]
        ϕ_upper = find_zero(fangle, (-π, 2*π))
        ϕ_lower = π - ϕs

        # ϕ_upper = 2*π
        # ϕ_lower = -π


        ϕ_mask = (ϕ_lower .< ϕ .< ϕ_upper)
        

        ϕ_new = Array[]
        ΔE_new = Array[]  
        z_new = Array[]

        z_new = z[ϕ_mask]
        ϕ_new = ϕ[ϕ_mask]
        ΔE_new = ΔE[ϕ_mask]


        # for j in 1:length(ϕ_mask)
        push!(particles_end, [z_new, ϕ_new, ΔE_new])
        # end

        master_storage[i] = particles_end
        
        z = z_new
        ϕ = ϕ_new
        ΔE = ΔE_new
        
    end
    return master_storage
end

function plot_phase_space_animated(n_turns::Int64, particles::Vector{Array}, ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64, harmonic::Int64,acc_radius::Float64, freq_rf::Float64, pipe_radius::Float64, E0::Float64, σ_E::Float64, update_η::Bool = false, update_E0::Bool = false, SR_damping::Bool = false, use_excitation::Bool = false, use_wakefield::Bool = false)
    gr()
    sin_ϕs = sin(ϕs)

    particles_out = longitudinal_evolve_2(turns, particles, sin_ϕs, α_c, mass, voltage, harmonic, radius, freq_rf, pipe_radius,energy,σ_E, true, true, true,true, true)

    γ = energy/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    ϕtest = collect(range(find_zero(fangle, (-2*π, 2*π)), π-ϕs, length = 100000))

    sep = sqrt.( abs.(voltage * energy * β^2 / (harmonic * π * η) * ( cos.(ϕtest) .+ cos(ϕs) .- sin(ϕs) .* (π .- ϕtest .- ϕs))))
    z_test = (.-ϕtest .+ ϕs ) * β * SPEED_LIGHT / (freq_rf*2 *π)
    sep_total = vcat(reverse(sep), -sep)
    ϕ_test_total = vcat(reverse(ϕtest), ϕtest)

    
    
    anim = @animate for i in 1:n_turns
        
        num_particles_surv = length(particles_out[1][i][1])

        plot(xlabel = L"\phi", ylabel = L"\frac{E}{\sigma _E}", title = "Turn $i; Surviving Particles = $num_particles_surv")
        plot!(xticks = (0:π/2:3*π/2, ["0", "π/2", "π", "3π/2"]), xlims = (0, 3*π/2), ylims = (-400, 400) )
        scatter!(particles_out[1][i][2],particles_out[1][i][3]/σ_E, legend = false , color = "red")
        plot!(ϕ_test_total, sep_total/σ_E, color = "black")
    end
    mp4(anim, "Haissinski/phase_space_evolution.mp4", fps=30)
end

ϕs = 5π/6
sin_ϕs = 0.5
turns = 4000
energy = 4e9
mass = .511e6
voltage = 5e6
harmonic = 360
radius = 250.
pipe_radius = .00025

α_c = 3.68e-4
γ = energy/mass
β = sqrt(1 - 1/γ^2)
η = α_c - 1/γ^2

μ_z = 0.
μ_E = 0.
σ_E = 1e6
ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT))
σ_z = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*energy/harmonic/voltage/abs(cos(ϕs))) * σ_E / energy 
println("σ_z = ", σ_z)
μ = [μ_z, μ_E]

dist_z_ini = Normal(μ_z, σ_z)
dist_E_ini = Normal(μ_E, σ_E)

z1 = rand(dist_z_ini, 10000)
E1 = rand(dist_E_ini, 10000)

Σ = [cov(z1, z1) cov(z1, E1) ; cov(E1, z1) cov(E1, E1)]
dist_total = MvNormal(μ, Σ)


num_particles = 10000
particles = Array[]
for i in 1:num_particles
    push!(particles, rand(dist_total, num_particles)[:,i])
end

ϕ0 = π/2 - ϕs
# ϕ0 = 0.
rf_cav_length = 1.
freq_rf = 280e7
σ_ϕ = σ_E * 2 * π * freq_rf * SPEED_LIGHT

plot_phase_space_animated(turns, particles, ϕs, α_c, mass, voltage, harmonic, radius, freq_rf, pipe_radius,energy,σ_E, true, true, true,true, true)

