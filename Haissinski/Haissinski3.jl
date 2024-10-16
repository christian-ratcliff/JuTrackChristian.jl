using Plots, Measures, Base.Threads

colors_set = [RGB(0.1, 0.2, 0.5), RGB(0.9, 0.2, 0.3), RGB(0.2, 0.8, 0.2), RGB(0.8, 0.8, 0.1), RGB(0.5, 0.2, 0.8), RGB(0.2, 0.9, 0.8), RGB(0.9, 0.5, 0.2), RGB(0.1, 0.6, 0.3), RGB(0.5, 0.1, 0.6), RGB(0.8, 0.3, 0.9), RGB(0.3, 0.7, 0.4), RGB(0.7, 0.4, 0.1), RGB(0.4, 0.5, 0.9), RGB(0.6, 0.2, 0.3), RGB(0.1, 0.4, 0.7), RGB(0.9, 0.6, 0.3), RGB(0.1, 0.1, 0.8), RGB(0.6, 0.8, 0.1), RGB(0.3, 0.1, 0.5), RGB(0.4, 0.4, 0.1), RGB(0.5, 0.9, 0.5), RGB(0.7, 0.5, 0.8), RGB(0.1, 0.5, 0.9), RGB(0.3, 0.6, 0.7), RGB(0.8, 0.5, 0.1), RGB(0.2, 0.5, 0.4), RGB(0.9, 0.9, 0.1), RGB(0.6, 0.1, 0.4), RGB(0.4, 0.2, 0.3), RGB(0.8, 0.7, 0.3), RGB(0.1, 0.9, 0.1), RGB(0.7, 0.1, 0.9), RGB(0.5, 0.7, 0.1), RGB(0.9, 0.1, 0.9), RGB(0.2, 0.4, 0.6)]


function longitudinal_evolve(n_turns::Int64, ϕ0_ini::Float64,ΔE_ini::Float64 , E0_ini::Float64, sin_ϕs::Float64, α_c::Float64, mass::Float64, e_volt::Float64, harmonic::Int64,radius::Float64,U0::Float64,pipe_rad::Float64, update_η::Bool = false, update_E0::Bool = false, SR_damping::Bool = false, use_wakefield::Bool = false )
    E0 = E0_ini
    ϕ = ϕ0_ini
    ΔE = ΔE_ini
    γ0 = E0/mass
    β0 = sqrt(1 - 1/γ0^2)
    eV = e_volt

    SPEED_LIGHT = 299792458
    ħ = 6.582119569e-16

    ϕ_data = Array{Float64}(undef, 0)
    ΔE_data = Array{Float64}(undef, 0)

    push!(ϕ_data, ϕ0)
    push!(ΔE_data, ΔE)
    for i in 1:n_turns
        ΔE += eV * (sin(ϕ) - sin_ϕs)

        if SR_damping == true
            # C = 2 * π * radius #is this right?
            # D =  α_c * C / (2 * π * radius)#I4/I2 we dont have the dispersion, so I am using the simplified version
            # ΔE -= U0 / E0 * (2 + D)

            # U = C_γ * E^4 / (2 * π) * I2 
            # ∂U_∂E = 2* C_γ * E^3 / (π) * I2 = 4 * C_γ * E^3 / radius
            # I2 = CIRCUM / (radius^2) = 2 * π / radius
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / radius
            ΔE -= ∂U_∂E * ΔE

            # α_fine = 1/137
            # n_photons = 5  * π * α_fine * γ0 / sqrt(3)
            # # ω_c = 3 * γ0^3 * C / (2 * radius)
            # # ω_c = 3 * γ0^3 * E0 / ħ
            # u_c = ħ * ω_c
            # avg_u_per_photon = (8 / (sqrt(3) * 15)) * u_c
            # avg_u = n_photons * avg_u_per_photon 
            # ΔE -= avg_u

            # ΔE *= (1 - 1/1000)
        end
        
        if use_wakefield == true
            # W(s) = Z0 * c / (π * a^2) Exp[-s/(c*τ)] * cos(sqrt(2*kp/a) * s)
            if ϕ-asin(sin_ϕs) > 0
                
            else

            end
        end

        if update_E0 == true
            E0 += eV * sin_ϕs
            p0 = sqrt(E0^2 - mass^2)
        end
        γ0 = E0/mass 
        β0 = sqrt(1 - 1/γ0^2)

        if update_η == true
            Δγ = ΔE/mass
        else
            Δγ = 0
        end
        η = α_c - 1/(γ0+Δγ)^2

        coeff = 2 * π * harmonic * η / (β0 * β0 * E0)
        ϕ += coeff * ΔE 

        push!(ϕ_data, ϕ)
        push!(ΔE_data, ΔE)
    end
    return ϕ_data, ΔE_data
end

turns = 750
energy = 4e9
mass = .511e6
voltage = 5e6
harmonic = 360
radius = 25.
pipe_radius = .00025

α_c = 3.68e-4
γ = energy/mass
β = sqrt(1 - 1/γ^2)
η = α_c - 1/γ^2
C_γ = 8.85e-5
U_SR = C_γ * (energy/1e9)^4 / radius * 1e9
sin_ϕs = U_SR / voltage
ϕs = asin(sin_ϕs)
ϕ0 = π/2 - ϕs
rf_cav_length = 1.


function plot_phase_space_compare(;with_effects::Bool = true, without_effects::Bool = true, lower_range::Float64 = 0., upper_range::Float64 = 0.125*energy, n_points::Int64 = 9)
    gr()
    colors = [colors_set[i % length(colors_set) + 1] for i in 1:n_points]
    idx = 1
    if with_effects == false && without_effects == false
        println("Please select at least one effect to plot")
        return
    elseif with_effects == true && without_effects == false
        plot( xlabel="ϕ", ylabel="ΔE [GeV]", margin = 5mm)
        for i in range(lower_range, upper_range, length = n_points)
            label = round(i, sigdigits = 3)
            
            ϕ_data_effects, ΔE_data_effects = longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, true, true, true, true)
            plot!(ϕ_data_effects,ΔE_data_effects/1e9, label = "ΔE = $label GeV", title = "With SR Damping and Wakefield Effects", legend = :outertopright, linecolor = colors[idx])
            idx +=1
        end
    elseif with_effects == false && without_effects == true
        plot( xlabel="ϕ", ylabel="ΔE [GeV]", margin = 5mm)
        for i in range(lower_range, upper_range, length = n_points)
            label = round(i, sigdigits = 3)

            ϕ_data, ΔE_data = longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, false, false, false, false)
            plot!(ϕ_data,ΔE_data/1e9, label = "ΔE = $label GeV", title = "With no effects", legend = :outertopright, xlabel="ϕ", ylabel="ΔE [GeV]", linecolor = colors[idx])
            idx +=1
        end
    else
        plot(xlabel="ϕ", ylabel="ΔE [GeV]", layout=(1,2), size = (1200, 600), margin = 5mm, title="Longitudinal Phase Space Evolution")
        for i in range(lower_range, upper_range, length = n_points)
            label = round(i, sigdigits = 3)
    
            ϕ_data_effects, ΔE_data_effects = longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, true, true, true, true)
            plot!(ϕ_data_effects,ΔE_data_effects/1e9, label = "ΔE = $label GeV", title = "With SR Damping and Wakefield Effects", legend = false, linecolor = colors[idx])
    
            ϕ_data, ΔE_data = longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, false, false, false, false)
            
            plot!(ϕ_data,ΔE_data/1e9, label = "ΔE = $label GeV", subplot = 2, title = "With no effects",legend = (1.15, 0.75), rightmargin = 50mm, linecolor = colors[idx])
            idx +=1
        end
    end
    plot!(xticks = (0:π/2:2*π, ["0", "π/2", "π", "3π/2", "2π"]), xlims = (0, 2*π), ylims = (-0.5, 0.5), )
end

function plot_phase_space_compare_animated(;with_effects::Bool = true, without_effects::Bool = true, lower_range::Float64 = 0., upper_range::Float64 = 0.125*energy, n_points::Int64 = 9)
    if with_effects == false && without_effects == false
        println("Please select at least one effect to plot")
        return
    end
    gr()
    energies = range(lower_range, upper_range, length = n_points)

    ϕ_data_effects = Array[]
    ΔE_data_effects = Array[]
    ϕ_data_no_effects = Array[]
    ΔE_data_no_effects = Array[]
    for i in energies
        push!(ϕ_data_effects,longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, true, true, true, true)[1])
        push!(ΔE_data_effects,longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius, true, true, true, true)[2])
        push!(ϕ_data_no_effects,longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius)[1])
        push!(ΔE_data_no_effects,longitudinal_evolve(turns, ϕ0, i, energy, sin_ϕs, α_c, mass, voltage, harmonic, radius, U_SR, pipe_radius)[2])
    end
    label = ["ΔE = $i GeV" for i in energies]
    if with_effects && without_effects
        plot(xlabel="ϕ", ylabel="ΔE [GeV]", size = (1200, 600),layout=(1,2), margin = 2mm, title="Longitudinal Phase Space Evolution", labels = label)
        filename = "Haissinski/phase_space_comparison_side_by_side.gif"
    elseif with_effects
        plot(xlabel="ϕ", ylabel="ΔE [GeV]", size = (600, 600), margin = 5mm, title="Longitudinal Phase Space Evolution, with Damping and Wakefield Effects", labels = label)
        filename = "Haissinski/phase_space_comparison_with_effects.gif"
    elseif without_effects
        plot(xlabel="ϕ", ylabel="ΔE [GeV]", size = (600, 600), margin = 5mm, title="Longitudinal Phase Space Evolution, without any effects", labels = label)
        filename = "Haissinski/phase_space_comparison_without_effects.gif"
    end
    colors = [colors_set[i % length(colors_set) + 1] for i in 1:n_points]
    
    anim = @animate for i in 1:turns
        # Initialize the plot based on the chosen effects
        if with_effects && without_effects
            
            scatter!(map(x -> x[i], ϕ_data_effects) ,map(x -> x[i], ΔE_data_effects)/1e9 , markercolor = colors, title = "With SR Damping and Wakefield Effects", subplot = 1, markersize = 3, legend = false, )
            scatter!(map(x -> x[i], ϕ_data_no_effects) ,map(x -> x[i], ΔE_data_no_effects)/1e9 , markercolor = colors, title = "With no effects", legend = false, subplot = 2, markersize = 3)
        elseif with_effects
            scatter!(map(x -> x[i], ϕ_data_effects) ,map(x -> x[i], ΔE_data_effects)/1e9 , markercolor = colors, title = "With SR Damping and Wakefield Effects", subplot = 1, markersize = 3, markerstrokewidth = 0., legend = false)
        elseif without_effects
            scatter!(map(x -> x[i], ϕ_data_no_effects) ,map(x -> x[i], ΔE_data_no_effects)/1e9 , markercolor = colors, title = "With no effects", legend = false, subplot = 2, markersize = 3, markerstrokewidth = 0.)
        end
        plot!( xlabel="ϕ", ylabel="ΔE [GeV]", margin = 5mm)
        plot!(xticks=(0:π/2:2*π, ["0", "π/2", "π", "3π/2", "2π"]), xlims=(0, 2*π), ylims=(-0.5, 0.5))
    end

    mp4(anim, filename, fps=30)
end


plot_phase_space_compare(;n_points = 7,)

plot_phase_space_compare_animated( n_points = 7)
### HOW TO CENTER?

#IF I CHANGE THE ΔE_INI to anything > .1*E0, it reaches a stable point far away
# xlims!(950, 1050)
# W(s) = Z0 * c / (π * a^2) Exp[-s/(c*τ)] * cos(sqrt(2*kp/a) * s)