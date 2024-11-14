"""
    Module: Beam Evolution Simulation
    
This code implements a high-performance beam evolution simulation for particle accelerators,
including functionality for particle generation, evolution tracking, and data visualization.
"""

using Distributions ;
using Random ;
Random.seed!(12345) ;
using LaTeXStrings ;
using Interpolations ;
using Roots ;
using BenchmarkTools ;
using Base.Threads ;
using StaticArrays ;
using SIMD ;
using StructArrays ;
using CircularArrays ;
using PreallocationTools ;
using LinearAlgebra ;
using LoopVectorization ;
using CairoMakie ;
using KernelDensity ;
using Statistics ;
using StatsBase ;
using FFTW ;
using Colors ;
using VideoIO ;
using Dierckx ;
using Dates ;
using HDF5 ;
using Distributed ;
using OffsetArrays ;
using ColorSchemes
using ThreadsX ;
using FLoops ; 
using FHist ; 
using ProgressMeter ; 
using Strided ;
using ProfileSVG ;
using Profile ;
using FileIO ; 


# Physical constants

const SPEED_LIGHT = 299792458 ;
const ELECTRON_CHARGE = 1.602176634e-19 ;
const MASS_ELECTRON = 0.51099895069e6 ;
const INV_SQRT_2π = 1 / sqrt(2 * π) ;

#=
Core Data Structures
=#

"""
    ParticleState{T<:AbstractFloat}

Immutable structure representing the state of a single particle in the beam.

# Fields
- `z::T`: Longitudinal position relative to the reference particle
- `ΔE::T`: Energy deviation from the reference energy
- `ϕ::T`: Phase relative to the RF wave

# Example
```julia
particle = ParticleState(0.1, 1e-3, 0.5)  # position, energy deviation, phase
```
"""
struct ParticleState{T<:AbstractFloat}
    z::T
    ΔE::T
    ϕ::T
end ;

"""
    BeamTurn{T,N}

Container for particle states across multiple turns, optimized for efficient memory access.

# Fields
- `states::SVector{N,StructArray{ParticleState{T}}}`: Static vector of particle states for each turn

# Type Parameters
- `T`: Floating-point precision type
- `N`: Number of turns plus one (includes initial state)
"""

struct BeamTurn{T,N}
    states::Array{StructArray{ParticleState{T}}, 1}
end

# Constructor
# function BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T
#     states = SVector{n_turns+1}(
#         StructArray{ParticleState{T}}((
#             Vector{T}(undef, n_particles),
#             Vector{T}(undef, n_particles),
#             Vector{T}(undef, n_particles)
#         )) for _ in 1:n_turns+1
#     )
#     return BeamTurn{T,n_turns+1}(states)
# end

function BeamTurn{T}(n_turns::Integer, n_particles::Integer) where T
    states = SVector{n_turns+1}( 
        StructArray{ParticleState{T}}((
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles),
            Vector{T}(undef, n_particles)
        )) for _ in 1:n_turns+1
    )
    return BeamTurn{T, n_turns+1}(states)
end

# copyto! for a single turn's worth of particle states
function Base.copyto!(dest::BeamTurn, turn_idx::Integer, src::StructArray{ParticleState{T}}) where T
    copyto!(dest.states[turn_idx].z, src.z)
    copyto!(dest.states[turn_idx].ΔE, src.ΔE)
    copyto!(dest.states[turn_idx].ϕ, src.ϕ)
    return dest
end

# Helper to copy from arrays of coordinates
function Base.copyto!(dest::BeamTurn, turn_idx::Integer, x::AbstractVector, px::AbstractVector, z::AbstractVector)
    copyto!(dest.states[turn_idx].x, x)
    copyto!(dest.states[turn_idx].px, px)
    copyto!(dest.states[turn_idx].z, z)
    return dest
end

# Helper methods
Base.getindex(pt::BeamTurn, i::Integer) = pt.states[i]
Base.iterate(pt::BeamTurn, state=1) = state > length(pt.states) ? nothing : (pt.states[state], state + 1)
Base.length(pt::BeamTurn{T,N}) where {T,N} = N

"""
    SimulationBuffers{T<:AbstractFloat}

Pre-allocated buffers for efficient computation during simulation.

# Fields
- `WF::Vector{T}`: Buffer for wakefield calculations
- `potential::Vector{T}`: Buffer for potential energy calculations
- `Δγ::Vector{T}`: Buffer for gamma factor deviations
- `η::Vector{T}`: Buffer for slip factor calculations
- `coeff::Vector{T}`: Buffer for temporary coefficients
- `temp_z::Vector{T}`: General temporary storage for z coordinates
- `temp_ΔE::Vector{T}`: General temporary storage for energy deviations
- `temp_ϕ::Vector{T}`: General temporary storage for phases

Used internally to minimize memory allocations during simulation steps.
"""
struct SimulationBuffers{T<:AbstractFloat}
    WF::Vector{T}
    potential::Vector{T}
    Δγ::Vector{T}
    η::Vector{T}
    coeff::Vector{T}
    temp_z::Vector{T}
    temp_ΔE::Vector{T}
    temp_ϕ::Vector{T}
    WF_temp::Vector{T}
    λ::Vector{T}
    convol::Vector{Complex{T}}
end ;


#=
High-Level Simulation Functions
=#


"""
    longitudinal_evolve(n_turns, particle_states, ϕs, α_c, mass, voltage, harmonic, 
                       acc_radius, freq_rf, pipe_radius, E0, σ_E; kwargs...) 
                       -> Union{BeamTurn{T,N}, Tuple{BeamTurn{T,N}, Vector{Any}}}

Simulate longitudinal beam evolution over multiple turns.

# Arguments
- `n_turns::Int`: Number of turns to simulate
- `particle_states::StructArray{ParticleState{T}}`: Initial particle states
- `ϕs::T`: Synchronous phase
- `α_c::T`: Momentum compaction factor
- `mass::T`: Particle mass
- `voltage::T`: RF voltage
- `harmonic::Int`: RF harmonic number
- `acc_radius::T`: Accelerator radius
- `freq_rf::T`: RF frequency
- `pipe_radius::T`: Beam pipe radius
- `E0::T`: Reference energy
- `σ_E::T`: Energy spread

# Keywords
- `update_η::Bool=false`: Enable slip factor updates
- `update_E0::Bool=false`: Enable reference energy updates
- `SR_damping::Bool=false`: Enable synchrotron radiation damping
- `use_excitation::Bool=false`: Enable quantum excitation
- `use_wakefield::Bool=false`: Enable wakefield effects
- `plot_potential::Bool=false`: Enable potential plotting
- `write_to_file::Bool=false`: Enable data writing to file
- `output_file::String="particles_output.hdf5"`: Output file path
- `additional_metadata::Dict{String,Any}=Dict{String,Any}()`: Additional simulation metadata

# Returns
- Without potential plots: `BeamTurn{T,N}` containing particle states for each turn
- With potential plots: Tuple of `BeamTurn{T,N}` and vector of potential plots

# Example
```julia
# Basic usage
states = longitudinal_evolve(1000, initial_states, 0.0, 1e-3, mass_electron,
                           1e6, 400, 100.0, 500e6, 0.02, 1e9, 1e-3)

# With additional features
states, plots = longitudinal_evolve(1000, initial_states, 0.0, 1e-3, mass_electron,
                                  1e6, 400, 100.0, 500e6, 0.02, 1e9, 1e-3;
                                  SR_damping=true, plot_potential=true)
```
"""

function threaded_fieldwise_copy!(destination, source)
    # Ensure you have the same number of particles
    @assert length(destination.z) == length(source.z)

    # Parallelize the copy operation for each particle
    @turbo for i in 1:length(source.z)
        destination.z[i] = source.z[i]
        destination.ΔE[i] = source.ΔE[i]
        destination.ϕ[i] = source.ϕ[i]
    end
end

function assign_to_turn!(particle_trajectory, particle_states, turn)
    # Use the threaded fieldwise copy to assign to the specified turn
    threaded_fieldwise_copy!(particle_trajectory.states[turn], particle_states)
end

function longitudinal_evolve(
    n_turns::Int,
    particle_states::StructArray{ParticleState{T}},
    ϕs::T,
    α_c::T,
    mass::T,
    voltage::T,
    harmonic::Int,
    acc_radius::T,
    freq_rf::T,
    pipe_radius::T,
    E0::T,
    σ_E::T;
    update_η::Bool=false,
    update_E0::Bool=false,
    SR_damping::Bool=false,
    use_excitation::Bool=false,
    use_wakefield::Bool=false, 
    plot_potential::Bool=false,
    plot_WF::Bool = false,
    write_to_file::Bool=false,
    output_file::String="particles_output.hdf5",
    additional_metadata::Dict{String, Any}=Dict{String, Any}()
) where T<:AbstractFloat

    # Pre-compute constants
    γ0 = E0 / mass
    β0 = sqrt(1 - 1/γ0^2)
    η0 = α_c - 1/(γ0^2)
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Initialize sizes and buffers
    n_particles = length(particle_states.z)
    buffers = create_simulation_buffers(n_particles,Int(n_particles/10), T)
    
    particle_trajectory = BeamTurn{T}(n_turns, n_particles)
    @views particle_trajectory.states[1] .= particle_states

# Initialize file and write metadata if writing is enabled
    if write_to_file
        timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
        folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(n_particles)"
        folder_storage = joinpath(folder_storage, timestamp)
        mkpath(folder_storage)
        
        output_file = joinpath(folder_storage, output_file)

        h5open(output_file, "w") do file
            # Create datasets with chunking for efficient writing
            chunk_size = min(n_particles, 10000)
            z_dset = create_dataset(file, "z", T, ((n_particles, n_turns + 1)), 
                                chunk=(chunk_size, 1))
            phi_dset = create_dataset(file, "phi", T, ((n_particles, n_turns + 1)), 
                                    chunk=(chunk_size, 1))
            dE_dset = create_dataset(file, "dE", T, ((n_particles, n_turns + 1)), 
                                chunk=(chunk_size, 1))
            
            # Write initial state
            z_dset[:, 1] = particle_states.z
            phi_dset[:, 1] = particle_states.ϕ
            dE_dset[:, 1] = particle_states.ΔE
            
            # Create metadata group
            meta_group = create_group(file, "metadata")
            
            # Write simulation parameters
            simulation_metadata = Dict{String, Any}(
                "n_turns" => n_turns,
                "n_particles" => n_particles,
                "sync_phase" => ϕs,
                "alpha_c" => α_c,
                "mass" => mass,
                "voltage" => voltage,
                "harmonic" => harmonic,
                "acc_radius" => acc_radius,
                "freq_rf" => freq_rf,
                "pipe_radius" => pipe_radius,
                "E0" => E0,
                "sigma_E" => σ_E,
                "update_eta" => update_η,
                "update_E0" => update_E0,
                "SR_damping" => SR_damping,
                "use_excitation" => use_excitation,
                "use_wakefield" => use_wakefield,
                "gamma0" => γ0,
                "beta0" => β0,
                "eta0" => η0,
                "timestamp" => string(Dates.now())
            )
            
            # Merge with additional metadata if provided
            merge!(simulation_metadata, additional_metadata)
            
            # Write all metadata
            for (key, value) in simulation_metadata
                meta_group[key] = value
            end
        end
    end
    if use_wakefield
        kp = T(3e1)
        Z0 = T(120π)
        cτ = T(4e-3)
        wake_factor = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt = sqrt(2*kp/pipe_radius)
    end

    potential_plots = plot_potential ? Vector{Any}(undef, n_turns) : nothing
    WF_plots = plot_WF ? Vector{Any}(undef, n_turns) : nothing
    p = Progress(n_turns, desc="Simulating Turns: ")
    # Main evolution loop with minimal allocations
    @inbounds for turn in 1:n_turns
        # Energy kick from RF using SIMD
        @turbo for i in 1:n_particles
            particle_states.ΔE[i] += voltage * (sin(particle_states.ϕ[i]) - sin_ϕs)
        end
        
        # Apply SR damping if enabled
        if SR_damping
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] *= (1 - ∂U_∂E)
            end
        end
        
        # Apply quantum excitation if enabled
        if use_excitation
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            excitation = sqrt(1-∂U_∂E^2) * σ_E
            randn!(buffers.potential)
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] += excitation * buffers.potential[i]
            end
        end
        
        # Apply wakefield effects if enabled
        if use_wakefield
            ν_s = sqrt(voltage * harmonic * α_c / (2π * E0))
            curr = (abs(η) / η) * ELECTRON_CHARGE / (2 * π * ν_s * σ_E) * (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
            if plot_potential && plot_WF

                
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles, curr, σ_z
                )
                # Store potential plot for this turn
                fig = Figure(size=(800, 500))

                # Create axes that will be reused
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                scatter!(ax,
                    particle_states.z / σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig

                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                # xlims!(ax2, -.02, 0)
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2
                
            elseif plot_WF && !plot_potential
 
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles, curr, σ_z
                )
                fig2 = Figure(size=(800, 500))
                ax2 = Axis(fig2[1, 1], xlabel=L"z", ylabel=L"\mathrm{WF}")
                xlims!(ax2, -.02, 0)
                scatter!(ax2,
                    particle_states.z,
                    buffers.WF,
                    markersize = 3
                )
                WF_plots[turn] = fig2

            elseif plot_potential && !plot_WF
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles, curr, σ_z
                )
                # Store potential plot for this turn
                fig = Figure(size=(800, 500))

                # Create axes that will be reused
                ax = Axis(fig[1, 1], xlabel=L"z / \sigma_z", ylabel=L"\mathrm{Potential}")
                scatter!(ax,
                    particle_states.z /σ_z,
                    buffers.potential,
                    markersize = 3
                )
                potential_plots[turn] = fig
            else
                # @btime apply_wakefield_inplace!(
                #     $particle_states, $buffers, $wake_factor, $wake_sqrt, $cτ,
                #     $E0, $acc_radius, $n_particles, curr
                # )
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles, curr, σ_z
                )
            end
        end
        
        # Update reference energy if needed
        if update_E0
            E0 += voltage * sin_ϕs
            γ0 = E0/mass 
            β0 = sqrt(1 - 1/γ0^2)
        end
        
        # Update phase advance using pre-allocated buffers
        if update_η
            @turbo for i in 1:n_particles
                buffers.Δγ[i] = particle_states.ΔE[i] / mass
                buffers.η[i] = α_c - 1/(γ0 + buffers.Δγ[i])^2
                buffers.coeff[i] = 2π * harmonic * buffers.η[i] / (β0 * β0 * E0)
                particle_states.ϕ[i] += buffers.coeff[i] * particle_states.ΔE[i]
            end
        else
            coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
            @turbo for i in 1:n_particles
                particle_states.ϕ[i] += coeff * particle_states.ΔE[i]
            end
        end
        
        # Update z coordinates using SIMD
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        @turbo for i in 1:n_particles
            particle_states.z[i] = (-particle_states.ϕ[i] + ϕs) / rf_factor
        end
        # copyto!(particle_trajectory, turn+1, particle_states)
        # @views particle_trajectory.states[turn+1] .= particle_states
        assign_to_turn!(particle_trajectory, particle_states, turn+1)

        if write_to_file
            h5open(output_file, "r+") do file
                file["z"][:, turn + 1] = particle_states.z
                file["phi"][:, turn + 1] = particle_states.ϕ
                file["dE"][:, turn + 1] = particle_states.ΔE
            end
        end
        next!(p)
    end
    
    if plot_potential && plot_WF
        return particle_trajectory, potential_plots, WF_plots
    elseif plot_potential && !plot_WF
        return particle_trajectory, potential_plots
    elseif !plot_potential && plot_WF
        return particle_trajectory, WF_plots
    else
        return particle_trajectory
    end
end ;




function longitudinal_evolve!(
    n_turns::Int,
    particle_states::StructArray{ParticleState{T}},
    ϕs::T,
    α_c::T,
    mass::T,
    voltage::T,
    harmonic::Int,
    acc_radius::T,
    freq_rf::T,
    pipe_radius::T,
    E0::T,
    σ_E::T;
    update_η::Bool=false,
    update_E0::Bool=false,
    SR_damping::Bool=false,
    use_excitation::Bool=false,
    use_wakefield::Bool=false,
    display_counter::Bool=true,
    plot_scatter::Bool=false
) where T<:AbstractFloat
    
    # Pre-compute constants
    γ0 = E0 / mass
    β0 = sqrt(1 - 1/γ0^2)
    η0 = α_c - 1/(γ0^2)
    sin_ϕs = sin(ϕs)
    rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
    
    # Initialize sizes and buffers
    n_particles = length(particle_states.z)
    buffers = create_simulation_buffers(n_particles,Int(n_particles/10), T)
    

    scatter_plots = plot_scatter ? Vector{Any}(undef, n_turns+1) : nothing
    if plot_scatter
        boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
        boundary_obs = Observable((boundary_points[1], boundary_points[2]))
        fig = Figure(size=(800, 500))
        ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
        Label(fig[1, 1, Top()], "Turn 1", fontsize = 20)
        scatter!(ax, 
            particle_states.ϕ,
            particle_states.ΔE / σ_E,
            markersize = 1, color = :black)
        lines!(ax, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
        xlims!(ax, 0, 3π/2)
        ylims!(ax, minimum(boundary_points[2]) / σ_E-5, maximum(boundary_points[2]) / σ_E+5)
        scatter_plots[1] = fig
    end
    # Initialize phases using pre-allocated buffer
    # @tturbo for i in 1:n_particles
    #     particle_states.ϕ[i] = -(particle_states.z[i] * rf_factor - ϕs)
    # end

    if use_wakefield
        kp = T(3e1)
        Z0 = T(120π)
        cτ = T(4e-3)
        wake_factor = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt = sqrt(2*kp/pipe_radius)
    end
    if display_counter
        p = Progress(n_turns, desc="Simulating Turns: ")
    end
    
    # Main evolution loop with minimal allocations
    @inbounds for turn in 1:n_turns
        # Energy kick from RF using SIMD
        @turbo for i in 1:n_particles
            particle_states.ΔE[i] += voltage * (sin(particle_states.ϕ[i]) - sin_ϕs)
        end
        
        # Apply SR damping if enabled
        if SR_damping
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] *= (1 - ∂U_∂E)
            end
        end
        
        # Apply quantum excitation if enabled
        if use_excitation
            ∂U_∂E = 4 * 8.85e-5 * (E0/1e9)^3 / acc_radius
            excitation = sqrt(1-∂U_∂E^2) * σ_E
            randn!(buffers.potential)
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] += excitation * buffers.potential[i]
            end
        end
        
        
        # Apply wakefield effects if enabled
        if use_wakefield
            ν_s = sqrt(voltage * harmonic * α_c / (2π * E0))
            curr = (abs(η) / η) * ELECTRON_CHARGE / (2 * π * ν_s * σ_E) * (1e11/(10.0^floor(Int, log10(n_particles)))) * n_particles
            apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles, curr, σ_z
                )
        end
        
        # Update reference energy if needed
        if update_E0
            E0 += voltage * sin_ϕs
            γ0 = E0/mass 
            β0 = sqrt(1 - 1/γ0^2)
        end
        
        # Update phase advance using pre-allocated buffers
        if update_η
            @turbo for i in 1:n_particles
                buffers.Δγ[i] = particle_states.ΔE[i] / mass
                buffers.η[i] = α_c - 1/(γ0 + buffers.Δγ[i])^2
                buffers.coeff[i] = 2π * harmonic * buffers.η[i] / (β0 * β0 * E0)
                particle_states.ϕ[i] += buffers.coeff[i] * particle_states.ΔE[i]
            end
        else
            coeff = 2π * harmonic * η0 / (β0 * β0 * E0)
            @turbo for i in 1:n_particles
                particle_states.ϕ[i] += coeff * particle_states.ΔE[i]
            end
        end
        
        # Update z coordinates using SIMD
        rf_factor = freq_rf * 2π / (β0 * SPEED_LIGHT)
        @turbo for i in 1:n_particles
            particle_states.z[i] = (-particle_states.ϕ[i] + ϕs) / rf_factor
        end

        # Store potential plot for this turn
        # fig = Figure(size=(800, 500))

        # Create axes that will be reused
        # ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
        if plot_scatter
            fig = Figure(size=(800, 500))
            ax = Axis(fig[1, 1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma_E}")
            Label(fig[1, 1, Top()], "Turn $(turn+1)", fontsize = 20)
            scatter!(ax, 
                particle_states.ϕ,
                particle_states.ΔE / σ_E,
                markersize = 1, color = :black)
            lines!(ax, boundary_obs[][1], boundary_obs[][2] / σ_E, color=:red)
            xlims!(ax, 0, 3π/2)
            ylims!(ax, minimum(boundary_points[2]) / σ_E-5, maximum(boundary_points[2]) / σ_E+5)
            scatter_plots[turn+1] = fig
        end
        if display_counter
            next!(p)
        end
    end
    return scatter_plots
end ;
"""
    generate_particles(μ_z::T, μ_E::T, σ_z::T, σ_E::T, num_particles::Int) 
                      -> StructArray{ParticleState{T}}

Generate initial particle distribution for simulation using multivariate normal distribution.

# Arguments
- `μ_z::T`: Mean longitudinal position
- `μ_E::T`: Mean energy deviation
- `σ_z::T`: Position spread (standard deviation)
- `σ_E::T`: Energy spread (standard deviation)
- `num_particles::Int`: Number of particles to generate

# Returns
- `StructArray{ParticleState{T}}`: Array of initial particle states

# Example
```julia
# Generate 10000 particles with given parameters
particles = generate_particles(0.0, 0.0, 1e-3, 1e-4, 10000)
```
"""
function generate_particles(
    μ_z::T,
    μ_E::T,
    σ_z::T,
    σ_E::T,
    num_particles::Int,
    energy::T,
    mass::T,
    ϕs::T,
    freq_rf::T,
) where T<:AbstractFloat
    
    # Pre-allocate arrays for initial sampling
    initial_sample_size = min(10000, num_particles)
    z_samples = Vector{T}(undef, initial_sample_size)
    E_samples = Vector{T}(undef, initial_sample_size)
    
    # Create initial distributions
    z_dist = Normal(μ_z, σ_z)
    E_dist = Normal(μ_E, σ_E)
    
    # Generate initial samples efficiently
    rand!(z_dist, z_samples)
    rand!(E_dist, E_samples)
    
    # Calculate covariance matrix efficiently
    Σ = @views begin
        cov_zz = cov(z_samples, z_samples)
        cov_zE = cov(z_samples, E_samples)
        cov_EE = cov(E_samples, E_samples)
        [cov_zz cov_zE; cov_zE cov_EE]
    end
    
    # Create multivariate distribution
    μ = SVector{2,T}(μ_z, μ_E)
    dist_total = MvNormal(μ, Symmetric(Σ))
    
    # Pre-allocate particle states array
    particle_states = StructArray{ParticleState{T}}((
        Vector{T}(undef, num_particles),  # z
        Vector{T}(undef, num_particles),  # ΔE
        Vector{T}(undef, num_particles)   # ϕ
    ))
    
    γ = energy / mass
    β = sqrt(1 - 1/γ^2)
    rf_factor = freq_rf * 2π / (β* SPEED_LIGHT)

    # # Generate particles in parallel
    # chunk_size = num_particles ÷ nthreads()
    
    # @threads for thread_idx in 1:nthreads()
    #     # Calculate range for this thread
    #     start_idx = (thread_idx - 1) * chunk_size + 1
    #     end_idx = thread_idx == nthreads() ? num_particles : thread_idx * chunk_size
        
    #     # Local RNG for thread safety
        local_rng = Random.default_rng()
        
    #     # Generate particles for this chunk
        for i in eachindex(particle_states.z)
            sample_vec = rand(local_rng, dist_total)
            particle_states.z[i] = sample_vec[1]
            particle_states.ΔE[i] = sample_vec[2]
            particle_states.ϕ[i] = -(particle_states.z[i] * rf_factor - ϕs)
        end
        
    # end
    return particle_states
end ;

#=
Data Management Functions
=#

"""
    write_particle_evolution(filename::String, particle_states::Vector{StructArray{ParticleState{T}}};
                           metadata::Dict{String,Any}=Dict{String,Any}()) -> Nothing

Write particle evolution data to HDF5 file with optimized chunking.

# Arguments
- `filename::String`: Output file path
- `particle_states::Vector{StructArray{ParticleState{T}}}`: Vector of particle states for each turn
- `metadata::Dict{String,Any}=Dict{String,Any}()`: Optional dictionary of metadata

# Example
```julia
# Save simulation results with metadata
metadata = Dict("voltage" => 1e6, "turns" => 1000)
write_particle_evolution("simulation_results.h5", states; metadata=metadata)
```
"""


function write_particle_evolution(filename::String, 
                                particle_states::Vector{StructArray{ParticleState{T}}};
                                metadata::Dict{String, Any}=Dict{String, Any}()) where T<:AbstractFloat
    n_turns = length(particle_states)
    n_particles = length(particle_states[1].z)

    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(n_particles)"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    filename = joinpath(folder_storage, filename)
    
    h5open(filename, "w") do file
        # Create datasets with optimal chunking
        chunk_size = min(n_particles, 10000)
        z_dset = create_dataset(file, "z", T, (n_particles, n_turns), 
                               chunk=(chunk_size, 1))
        phi_dset = create_dataset(file, "phi", T, (n_particles, n_turns), 
                                chunk=(chunk_size, 1))
        dE_dset = create_dataset(file, "dE", T, (n_particles, n_turns), 
                                chunk=(chunk_size, 1))
        
        # Pre-allocate buffers for batch writing
        z_buffer = Matrix{T}(undef, n_particles, n_turns)
        phi_buffer = Matrix{T}(undef, n_particles, n_turns)
        dE_buffer = Matrix{T}(undef, n_particles, n_turns)
        
        # Fill buffers
        @inbounds for turn in 1:n_turns
            z_buffer[:, turn] = particle_states[turn].z
            phi_buffer[:, turn] = particle_states[turn].ϕ
            dE_buffer[:, turn] = particle_states[turn].ΔE
        end
        
        # Write data in one go
        write!(z_dset, z_buffer)
        write!(phi_dset, phi_buffer)
        write!(dE_dset, dE_buffer)
        
        # Write metadata if provided
        if !isempty(metadata)
            meta_group = create_group(file, "metadata")
            for (key, value) in metadata
                meta_group[key] = value
            end
        end
    end
end ;

"""
    read_particle_evolution(filename::String; turn_range=nothing)
                          -> Tuple{Vector{StructArray{ParticleState{T}}}, Dict{String,Any}}

Read particle evolution data from HDF5 file.

# Arguments
- `filename::String`: Input file path
- `turn_range=nothing`: Optional range of turns to read

# Returns
- Tuple of particle states and metadata dictionary

# Example
```julia
# Read all turns
states, metadata = read_particle_evolution("simulation_results.h5")

# Read specific turn range
states, metadata = read_particle_evolution("simulation_results.h5", turn_range=1:100)
```
"""
function read_particle_evolution(filename::String; turn_range=nothing)
    h5open(filename, "r") do file
        # Get data dimensions and type
        z_dset = file["z"]
        n_particles, n_turns = size(z_dset)
        T = eltype(z_dset)
        
        # Determine turns to read
        turns_to_read = isnothing(turn_range) ? (1:n_turns) : turn_range
        n_turns_to_read = length(turns_to_read)
        
        # Pre-allocate vector of StructArrays for all turns
        particle_states = Vector{StructArray{ParticleState{T}}}(undef, n_turns_to_read)
        
        # Pre-allocate buffers for reading
        z_buffer = Matrix{T}(undef, n_particles, length(turns_to_read))
        phi_buffer = Matrix{T}(undef, n_particles, length(turns_to_read))
        dE_buffer = Matrix{T}(undef, n_particles, length(turns_to_read))
        
        # Read data in chunks for memory efficiency
        read!(file["z"], z_buffer)
        read!(file["phi"], phi_buffer)
        read!(file["dE"], dE_buffer)
        
        # Create StructArrays for each turn using views to avoid copying
        @inbounds for (idx, turn) in enumerate(turns_to_read)
            particle_states[idx] = StructArray{ParticleState{T}}((
                view(z_buffer, :, idx),
                view(phi_buffer, :, idx),
                view(dE_buffer, :, idx)
            ))
        end
        
        # Read metadata if it exists
        metadata = Dict{String, Any}()
        if haskey(file, "metadata")
            metadata_group = file["metadata"]
            for key in keys(metadata_group)
                metadata[key] = read(metadata_group[key])
            end
        end
        
        return particle_states, metadata
    end
end ;

"""
    read_turn_range(filename::String, turn_range::AbstractRange) -> StructArray{ParticleState{T}}

Efficiently read specific turn range from HDF5 file.

# Arguments
- `filename::String`: Input file path
- `turn_range::AbstractRange`: Range of turns to read

# Returns
- `StructArray{ParticleState{T}}`: Particle states for specified turns

# Example
```julia
# Read turns 100-200
states = read_turn_range("simulation_results.h5", 100:200)
```
"""
function read_turn_range(filename::String, turn_range::AbstractRange)
    h5open(filename, "r") do file
        z_dset = file["z"]
        T = eltype(z_dset)
        n_particles = size(z_dset, 1)
        
        # Read only the specified turns
        z = z_dset[:, turn_range]
        phi = file["phi"][:, turn_range]
        dE = file["dE"][:, turn_range]
        
        return StructArray{ParticleState{T}}((z, phi, dE))
    end
end ;

#=
Physics Calculation Functions
=#

"""
    make_separatrix(ϕs::Float64, voltage::Float64, energy::Float64, 
                   harmonic::Int64, η::Float64, β::Float64) 
                   -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate separatrix coordinates for phase space visualization using optimized numerical methods.

# Arguments
- `ϕs::Float64`: Synchronous phase
- `voltage::Float64`: RF voltage
- `energy::Float64`: Reference energy
- `harmonic::Int64`: RF harmonic number
- `η::Float64`: Slip factor
- `β::Float64`: Relativistic beta

# Returns
- Tuple of vectors containing phase and energy coordinates for separatrix

# Example
```julia
# Calculate separatrix for given parameters
phases, energies = make_separatrix(0.0, 1e6, 1e9, 400, 1e-3, 0.999999)
```
"""
function make_separatrix(ϕs::Float64, voltage::Float64, energy::Float64, 
    harmonic::Int64, η::Float64, β::Float64)
    # Pre-calculate constants to avoid repeated computation
    const_factor = voltage * energy * β^2 / (harmonic * π * η)

    # Improved root finding function with better numerical stability
    function fangle(ϕu)
        Δϕ = π - ϕu - ϕs
        return -cos(ϕu) - cos(ϕs) + sin(ϕs) * Δϕ
    end

    # More robust initial bracket for root finding
    ϕ_lower = max(-2π, ϕs - 2π)
    ϕ_upper = min(2π, ϕs + 2π)

    # Use more robust root finding method
    ϕ_unstable = find_zero(fangle, (ϕ_lower, ϕ_upper), Roots.Brent())

    # Optimize the number of points based on the region of interest
    Δϕ = π - ϕs - ϕ_unstable
    n_points = max(1000, round(Int, abs(Δϕ) * 500))  # Scale points with separatrix size

    # Use LinRange for more efficient memory allocation
    ϕtest = LinRange(ϕ_unstable, π-ϕs, n_points)

    # Preallocate arrays
    sep = Vector{Float64}(undef, n_points)

    # Vectorize the main calculation
    @. sep = sqrt(abs(const_factor * (cos(ϕtest) + cos(ϕs) - sin(ϕs) * (π - ϕtest - ϕs))))

    # Create the full separatrix more efficiently
    sep_total = Vector{Float64}(undef, 2n_points)
    ϕ_test_total = Vector{Float64}(undef, 2n_points)

    # Fill both halves simultaneously
    @views begin
        sep_total[1:n_points] = reverse(sep)
        sep_total[n_points+1:end] = -sep
        ϕ_test_total[1:n_points] = reverse(ϕtest)
        ϕ_test_total[n_points+1:end] = ϕtest
    end

    return (ϕ_test_total, sep_total)
end ;

"""
    make_separatrix_extended(ϕs, voltage, energy, harmonic, η, β) 
    -> Tuple{Vector{Float64}, Vector{Float64}, Float64, Float64}

Extended version of separatrix calculation including bucket parameters.

Additional returns:
- Bucket height
- Bucket area
"""
function make_separatrix_extended(ϕs::Float64, voltage::Float64, energy::Float64, 
    harmonic::Int64, η::Float64, β::Float64)
    # Calculate basic separatrix
    ϕ_total, sep_total = make_separatrix(ϕs, voltage, energy, harmonic, η, β)

    # Calculate bucket area and height
    const_factor = voltage * energy * β^2 / (harmonic * π * η)
    bucket_height = sqrt(abs(const_factor * (2)))  # Maximum height

    # Calculate bucket area (approximate)
    bucket_width = π - ϕs
    bucket_area = 4 * bucket_height * bucket_width  # Approximate area

    return (ϕ_total, sep_total, bucket_height, bucket_area)
end ;

#=
Visualization Functions
=#


"""
    all_animate(n_turns, particles_out, ϕs, α_c, mass, voltage, harmonic, E0, σ_E, σ_z, [filename]) -> Nothing

Generate an animation visualizing beam evolution over multiple turns. Creates histograms of particle 
distributions and phase space plots.

# Arguments
- `n_turns::Int64`: Number of turns to animate
- `particles_out::Union{Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}}, BeamTurn{Float64}}`: 
    Vector of particle states for each turn or BeamTurn object containing particle states
- `ϕs::Float64`: Synchronous phase
- `α_c::Float64`: Momentum compaction factor
- `mass::Float64`: Particle mass
- `voltage::Float64`: RF cavity voltage
- `harmonic::Int64`: RF harmonic number
- `E0::Float64`: Reference beam energy
- `σ_E::Float64`: Energy spread
- `σ_z::Float64`: Bunch length
- `filename::String="all_animation.mp4"`: Output animation file path

# Description
Creates an animation showing:
- Phase space distribution (φ vs ΔE/σ_E)
- Longitudinal distribution histogram (z/σ_z)
- Energy distribution histogram (ΔE/σ_E)
- Separatrix boundary in phase space

The animation is saved in a timestamped subfolder under "Haissinski/particle_sims/".

# Example
```julia
# For vector of tuples
all_animate(1000, particle_states, 0.1, 1.89e-4, 0.511e6, 1.0e6, 1320, 3.0e9, 1.0e6, 0.01)

# For BeamTurn object
all_animate(1000, beam_states, 0.1, 1.89e-4, 0.511e6, 1.0e6, 1320, 3.0e9, 1.0e6, 0.01)
```
"""
#the below version accepts the particle_out as a BeamTurn object
function all_animate(n_turns::Int64,
    particles_out::BeamTurn{Float64},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64,E0::Float64, σ_E::Float64, σ_z::Float64, filename::String="all_animation.mp4")

    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)

    filename = joinpath(folder_storage, filename)

    # Precompute separatrix
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    println("Starting animation generation...")
    
    # Create a figure
    fig = Figure(size=(1400, 900))
    
    # Create axes that will be reused
    ax_z = Axis(fig[1, 1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
    ax_E = Axis(fig[1, 2], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
    ax_phase = Axis(fig[2, 1:2], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
    title_label = Label(fig[0, :], "Turn 1", fontsize=20)

    # Set fixed y-limits for histograms - adjust this value as needed
    FIXED_Y_MAX = 2000  # You can change this to whatever maximum value you need
    
    # Create the animation using record
    record(fig, filename, 1:n_turns; framerate=60) do frame_idx
        # Clear previous plots
        empty!(ax_z)
        empty!(ax_E)
        empty!(ax_phase)
        
        # Get current frame data
        z_data = particles_out[frame_idx].z
        ϕ_data = particles_out[frame_idx].ϕ
        E_data = particles_out[frame_idx].ΔE
        
        # Plot phase space
        scatter!(ax_phase, ϕ_data, E_data/σ_E, color=:black, markersize=1)
        lines!(ax_phase, boundary_points[1], boundary_points[2]/σ_E, color=:red)
        
        # Plot histograms and KDEs
        z_normalized_mask = -5 .< z_data / σ_z .< 5 
        z_normalized = z_data[z_normalized_mask] / σ_z
        E_normalized_mask = -120 .< E_data / σ_E .< 120
        E_normalized = E_data[E_normalized_mask] / σ_E
        

        density!(ax_z, z_normalized, color = (:red, 0.3),
                strokecolor = :red, strokewidth = 3, strokearound = true)
        
        # Energy distribution with fixed bins
        density!(ax_E, E_normalized, color = (:red, 0.3),
        strokecolor = :red, strokewidth = 3, strokearound = true)
        # Set limits and labels
        xlims!(ax_z, (-5, 5))
        xlims!(ax_E, (-120, 120))
        xlims!(ax_phase, (0, 3π/2))
        ylims!(ax_phase, (-400, 400))
        
        # Set fixed y limits for both histogram plots
        ylims!(ax_z, (0, 1))
        ylims!(ax_E, (0, .1))
        
        # Update turn number
        title_label.text = "Turn $frame_idx"
        
        if frame_idx % 25 == 0
            println("Processed frame $frame_idx of $n_turns")
        end
    end
    println("Animation complete!")
end ;

"""
NEED TO ADD HERE

"""
# Modified histogram calculation function for fixed bins
function calculate_fixed_histogram(data::Vector{Float64}, bins::AbstractRange)
    hist = fit(Histogram, data, bins)
    centers = (bins[1:end-1] + bins[2:end]) ./ 2
    return centers, hist.weights
end



"""
    scatter_particle_data(particle_states, turn_number, ϕs, α_c, mass, voltage, harmonic, 
                         E0, σ_E, σ_z; filename="particle_scatter.png", ϕ_plot=true, save_fig=true) -> Figure

Generate scatter plots of particle distributions at a specific turn.

# Arguments
- `particle_states::BeamTurn{Float64}`: BeamTurn object containing particle states
- `turn_number::Int64`: Turn number to plot
- `ϕs::Float64`: Synchronous phase
- `α_c::Float64`: Momentum compaction factor
- `mass::Float64`: Particle mass
- `voltage::Float64`: RF cavity voltage
- `harmonic::Int64`: RF harmonic number
- `E0::Float64`: Reference beam energy
- `σ_E::Float64`: Energy spread
- `σ_z::Float64`: Bunch length

# Keywords
- `filename::String="particle_scatter.png"`: Output file path
- `ϕ_plot::Bool=true`: If true, plot φ vs ΔE/σ_E; if false, plot z/σ_z vs ΔE/σ_E
- `save_fig::Bool=true`: If true, save the figure to file

# Returns
- `Figure`: Makie figure object containing the scatter plot

# Example
```julia
# Plot phase space at turn 100
fig = scatter_particle_data(beam_states, 100, 0.1, 1.89e-4, 0.511e6, 
                            1.0e6, 1320, 3.0e9, 1.0e6, 0.01)
```
"""
function scatter_particle_data(particle_states::BeamTurn{Float64}, turn_number::Int64, 
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64, harmonic::Int64,
    E0::Float64, σ_E::Float64, σ_z::Float64,; filename::String="particle_scatter.png", ϕ_plot::Bool=true, save_fig::Bool=true)
    
    # Precompute separatrix
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)

    z_data = particle_states[turn_number].z
    ϕ_data = particle_states[turn_number].ϕ
    E_data = particle_states[turn_number].ΔE
    
    # Create a figure
    fig = Figure(size=(800, 500))
    if ϕ_plot
        # Create axes
        ax = Axis(fig[1,1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        
        # Plot phase space
        scatter!(ax, particle_states[turn_number].ϕ, particle_states[turn_number].ΔE/σ_E, color=:black, markersize=1)
        lines!(ax, boundary_points[1], boundary_points[2]/σ_E, color=:red)
        
        # Save the figure
        save(filename)
    else
        # Create axes
        ax = Axis(fig[1,1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        # Plot phase space
        scatter!(ax, particle_states[turn_number].z/σ_z, particle_states[turn_number].ΔE/σ_E, color=:black, markersize=1)
        
        # Save the figure
        save(filename)
    end 
    return fig
end;


function scatter_particle_data(x_axis::Vector{Float64}, y_axis::Vector{Float64},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64, harmonic::Int64,
    E0::Float64, σ_E::Float64, σ_z::Float64, ;filename::String="particle_scatter.png", ϕ_plot=true, save_fig::Bool=true)
    
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)

    filename = joinpath(folder_storage, filename)
    # Precompute separatrix
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    
    # Create a figure
    fig = Figure(size=(800, 800))

    if ϕ_plot
        # Create axes
        ax = Axis(fig[1,1], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        
        # Plot phase space
        scatter!(ax, x_axis, y_axis, color=:black, markersize=1)
        lines!(ax, boundary_points[1], boundary_points[2]/σ_E, color=:red)
        
        # Save the figure
        if save_fig
            save(filename)
        end
    else
        # Create axes
        ax = Axis(fig[1,1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
        # Plot phase space
        scatter!(ax, x_axis, y_axis, color=:black, markersize=1)
        
        # Save the figure
        if save_fig
            save(filename)
        end
    end
    return fig
end; 


"""
    histogram_particle_data(particles_out, turn_number; z_hist=true, e_hist=true, 
                        filename="histogram_particle.png", save_figs=true) -> Union{Figure, Tuple{Figure, Figure}}

Generate histograms of particle distributions at a specific turn.

# Arguments
- `particles_out::BeamTurn`: BeamTurn object containing particle states
- `turn_number::Int64`: Turn number to plot

# Keywords
- `z_hist::Bool=true`: If true, generate longitudinal distribution histogram
- `e_hist::Bool=true`: If true, generate energy distribution histogram
- `filename::String="histogram_particle.png"`: Base filename for output
- `save_figs::Bool=true`: If true, save figures to files

# Returns
- If both `z_hist` and `e_hist` are true: Tuple of two Figure objects (z and E histograms)
- If only one histogram is selected: Single Figure object
- If neither histogram is selected: Nothing

# Example
```julia
# Generate both histograms at turn 100
z_fig, e_fig = histogram_particle_data(beam_states, 100)

# Generate only energy histogram
e_fig = histogram_particle_data(beam_states, 100, z_hist=false)
```
"""

function histogram_particle_data(particles_out::BeamTurn,turn_number::Int64, ;z_hist::Bool = true,e_hist::Bool=true, filename::String="histogram_particle.png", save_figs::Bool=true)
    
    function calculate_fixed_histogram(data::Vector{Float64}, bins::AbstractRange)
        hist = fit(Histogram, data, bins)
        centers = (bins[1:end-1] + bins[2:end]) ./ 2
        return centers, hist.weights
    end

    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)

    
    n_bins = 10^(ceil(Int, log10(length(particles_out[1])))-2)

    if z_hist
        
        z_bin_width = (5 - (-5)) / n_bins  # Since xlims is (-5, 5)
        z_bins = range(-5, 5, step=z_bin_width)
        z_normalized = particles_out[turn_number].z / σ_z
        z_centers, z_counts = calculate_fixed_histogram(z_normalized, z_bins)
        z_kde_x, z_kde_y = calculate_kde(z_normalized)
        fig_z = Figure()
        ax_z = Axis(fig_z[1,1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
        barplot!(ax_z, z_centers, z_counts, color=(:red, 0.5))
        lines!(ax_z, z_kde_x, z_kde_y .* length(z_normalized) .* z_bin_width,
            color=:green, linewidth=2)
        
        if save_figs
            filename = joinpath(folder_storage, filename)
            if z_hist && e_hist
                filename = "histogram_particle_z.png"
                filename = joinpath(folder_storage, filename)
            end
            save(filename)
        end
    end
    if e_hist
        E_bin_width = (120 - (-120)) / n_bins  # Since xlims is (-120, 120)
        E_bins = range(-120, 120, step=E_bin_width)
        E_normalized = particles_out[turn_number].ΔE / σ_E
        E_centers, E_counts = calculate_fixed_histogram(E_normalized, E_bins)
        E_kde_x, E_kde_y = calculate_kde(E_normalized)

        fig_E = Figure()
        ax_E = Axis(fig_E[1,1], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
        barplot!(ax_E, E_centers, E_counts, color=(:red, 0.5))
        lines!(ax_E, E_kde_x, E_kde_y .* length(E_normalized) .* E_bin_width,
            color=:green, linewidth=2)
        if save_figs
            filename = joinpath(folder_storage, filename)
            if z_hist && e_hist
                filename = "histogram_particle_E.png"
                filename = joinpath(folder_storage, filename)
            end
            save(filename)
        end
    end

    if z_hist && e_hist
        return fig_z, fig_E
    elseif z_hist && !e_hist
        return fig_z
    elseif e_hist && !z_hist
        return fig_E
    else
        println("No histogram selected")
    end
end;



"""
    calculate_histogram(data, bins) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate histogram for distribution visualization.

# Arguments
- `data::Vector{Float64}`: Input data vector
- `bins::Int64`: Number of histogram bins

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple of (bin_centers, bin_counts)

# Example
```julia
centers, counts = calculate_histogram(particle_energies, 50)
```
"""
@inline function calculate_histogram(data::Vector{Float64}, bins::Int64)
    # hist = fit(Histogram, data, nbins=bins)
    # centers = (hist.edges[1][1:end-1] + hist.edges[1][2:end]) ./ 2
    # return collect(centers), hist.weights
    histo = Hist1D(data, nbins=bins)
    centers = (histo.binedges[1][1:end-1] + histo.binedges[1][2:end]) ./ 2
    return collect(centers), histo.bincounts
end ;

"""
    calculate_kde(data, bandwidth=nothing) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate kernel density estimate for distribution visualization.

# Arguments
- `data::Vector{Float64}`: Input data vector
- `bandwidth::Union{Float64, Nothing}=nothing`: Bandwidth parameter for KDE. If nothing, uses 
    Silverman's rule of thumb: bandwidth = 1.06 * σ * n^(-1/5)

# Returns
- `Tuple{Vector{Float64}, Vector{Float64}}`: Tuple of (x_coordinates, density_values)

# Example
```julia
x_coords, density = calculate_kde(particle_positions)
x_coords, density = calculate_kde(particle_positions, bandwidth=0.1)
```
"""
@inline function calculate_kde(data::Vector{Float64}, bandwidth=nothing)
    if isnothing(bandwidth)
        bandwidth = 1.06 * std(data) * length(data)^(-0.2)
    end
    kde_obj = kde(data; bandwidth=bandwidth)
    return kde_obj.x, kde_obj.density
end ;

#=
Helper Functions
=#

"""
    create_simulation_buffers(n_particles, T=Float64) -> SimulationBuffers{T}

Create pre-allocated buffers for simulation calculations.

# Arguments
- `n_particles::Int`: Number of particles in simulation
- `T::Type=Float64`: Numeric type for calculations

# Returns
- `SimulationBuffers{T}`: Struct containing pre-allocated arrays for:
    - WF: Wakefield values
    - potential: Wakefield potential
    - Δγ: Change in Lorentz factor
    - η: Slip factor
    - coeff: Temporary coefficients
    - temp_z: Temporary z positions
    - temp_ΔE: Temporary energy deviations
    - temp_ϕ: Temporary phase values

# Example
```julia
buffers = create_simulation_buffers(10000)
```
"""
function create_simulation_buffers(n_particles::Int,nbins::Int, T::Type=Float64)
    SimulationBuffers{T}(
        Vector{T}(undef, n_particles),  # WF
        Vector{T}(undef, n_particles),  # potential
        Vector{T}(undef, n_particles),  # Δγ
        Vector{T}(undef, n_particles),  # η
        Vector{T}(undef, n_particles),  # coeff
        Vector{T}(undef, n_particles),  # temp_z
        Vector{T}(undef, n_particles),  # temp_ΔE
        Vector{T}(undef, n_particles),   # temp_ϕ
        Vector{T}(undef, nbins),   # WF_temp
        Vector{T}(undef, nbins),   # λ
        Vector{Complex{T}}(undef, nbins)   # convol

    )
end ;



@inline function dierckx_interpolation(x, y, query_points)
    spl = Spline1D(x, y; k=1, bc="extrapolate")  # k=1 for linear interpolation
    return evaluate(spl, query_points)
end

"""
    apply_wakefield_inplace!(particle_states, buffers, wake_factor, wake_sqrt, cτ, 
                            E0, acc_radius, n_particles) -> Nothing

Apply wakefield effects to particle states in-place using optimized calculations.

# Arguments
- `particle_states::StructArray{ParticleState{T}}`: Current particle states
- `buffers::SimulationBuffers{T}`: Pre-allocated calculation buffers
- `wake_factor::T`: Wakefield strength factor
- `wake_sqrt::T`: Square root term for wakefield calculation
- `cτ::T`: Characteristic time
- `E0::T`: Reference beam energy
- `acc_radius::T`: Accelerator radius
- `n_particles::Int`: Number of particles

# Implementation Notes
- Uses branchless operations for performance
- Implements SIMD optimizations
- Sorts particles by z-position before wakefield calculation
- Uses linear interpolation for potential calculation

# Example
```julia
apply_wakefield_inplace!(particles, buffers, 1.0e6, sqrt(2.0), 1.0e-12, 
                        3.0e9, 75.0, 10000)
```
"""

function apply_wakefield_inplace!(
    particle_states::StructArray{ParticleState{T}}, 
    buffers::SimulationBuffers{T}, 
    wake_factor::T, 
    wake_sqrt::T, 
    cτ::T, 
    E0::T, 
    acc_radius::T, 
    n_particles::Int,
    current::T,
    σ_z::T) where T<:AbstractFloat
    
    # Fast buffer clearing with broadcast
    buffers.WF .= zero(T)
    buffers.potential .= zero(T)
    
    z_positions = @views particle_states.z
    inv_cτ = 1 / cτ
    
    # Use fixed number of bins for stability and speed
    nbins = next_power_of_two(Int(10^(ceil(Int, log10(length(z_positions))-2)) ))  # Power of 2 for FFT efficiency
    # nbins  = 2048
    # Fast histogram calculation
    bin_centers, bin_amounts = calculate_histogram(z_positions, nbins)
    nbins = length(bin_centers)
    bin_size = bin_centers[2] - bin_centers[1]
    power_2_length = next_power_of_two(2*nbins-1)
    
    # Fast buffer clearing
    buffers.λ .= zero(T)
    buffers.WF_temp .= zero(T)
    buffers.convol .= zero(T)
    # z_mean = 0#mean(bin_centers)
    # Vectorize WF calculation
    @turbo for i in eachindex(bin_centers)
        z = bin_centers[i]
        buffers.WF_temp[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    # Fast λ calculation
    # log_bin_size = nbins/n_particles*100#T(10.0)^(ceil(Int, log10(bin_size)+5))
    # log_bin_size = round(nbins/100, digits=2)/2.355
    log_bin_size = (maximum(bin_centers) - minimum(bin_centers))/ σ_z  / 100
    @turbo for i in eachindex(bin_centers)
        buffers.λ[i] = delta(bin_centers[i], log_bin_size)
    end
    
    # Fast normalization with views
    normalized_amounts = bin_amounts .* (1/n_particles)
    λ = buffers.λ[1:nbins]
    WF_temp = buffers.WF_temp[1:nbins]
    convol = buffers.convol[1:power_2_length]
    
    # Fast convolution
    convol .= FastLinearConvolution(WF_temp, λ .* normalized_amounts, power_2_length) .* current
    
    # Efficient interpolation setup
    temp_z = range(minimum(z_positions), maximum(z_positions), length=length(convol))
    resize!(buffers.potential, length(z_positions))
    
    # Use fast linear interpolation with bounds checking disabled for speed
    buffers.potential .= LinearInterpolation(temp_z, real.(convol), extrapolation_bc=Line()).(z_positions)
    
    # Fast energy and wake function update
    # mean_z_val = 0#mean(z_positions)
    @turbo for i in eachindex(z_positions)
        z = z_positions[i]
        particle_states.ΔE[i] -= buffers.potential[i]
        # Avoid branching with multiplication by condition
        buffers.WF[i] = z > 0 ? zero(T) : wake_factor * exp(z * inv_cτ) * cos(wake_sqrt * z)
    end
    
    return nothing
end



@inline function FastConv1D(f::AbstractVector{T},g::AbstractVector{T}) where T
    return ifft(fft(f).*fft(g))
end

@inline function FastLinearConvolution(f::AbstractVector{T},g::AbstractVector{T}, power_2_length::Int) where T

    pad_and_ensure_power_of_two!(f, g, power_2_length)

    # return FastConv1D( f_pad, g_pad )
    return FastConv1D( f, g )
end



# Check if a number is a power of two
function is_power_of_two(n::Int)
    return (n & (n - 1)) == 0 && n > 0
end

# Get the next power of two greater than or equal to a number
function next_power_of_two(n::Int)
    return Int(2^(ceil(log2(n))))  # Nearest power of two greater than or equal to n
end

# Pad both vectors to length N*M - 1 and then ensure both are powers of two
function pad_and_ensure_power_of_two!(f::AbstractVector{T}, g::AbstractVector{T}, power_two_length::Int) where T
    N = length(f)
    M = length(g)
    # Resize and pad f
    original_f = copy(f)#@views f[:]
    resize!(f, power_two_length)
    f[1:N] = original_f
    f[N+1:end] .= zero(T)
    
    # Resize and pad g
    original_g = copy(g)#@views g[:]
    resize!(g, power_two_length)
    g[1:M] = original_g
    g[M+1:end] .= zero(T)
    
    return nothing
end



"""
    delta(x::T, ϵ::T) where T<:AbstractFloat -> T

Calculate smoothed delta function for wakefield calculations.

# Arguments
- `x::T`: Input value
- `ϵ::T`: Smoothing parameter

# Returns
- `T`: Smoothed delta function value

# Implementation Notes
- Uses pre-computed 1/π for efficiency
- Implements Lorentzian smoothing

# Example
```julia
smoothed_value = delta(0.1, 1e-3)
```
"""
@inline function delta(x::T, σ::T) where T<:AbstractFloat
    σ_inv = INV_SQRT_2π / σ
    exp_factor = -0.5 / (σ^2)
    return σ_inv * exp(x^2 * exp_factor)
end

"""

"""
function precompute_densities(particles_out, σ_z, σ_E)
    n_turns = length(particles_out)
    z_densities = Vector{KernelDensity.UnivariateKDE}(undef, n_turns)
    E_densities = Vector{KernelDensity.UnivariateKDE}(undef, n_turns)
    p = Progress(n_turns, desc="Precomputing densities: ")
    for i in 1:n_turns
        z_data = particles_out[i].z
        E_data = particles_out[i].ΔE
        
        # Apply masks and normalize
        z_normalized = @view(z_data[.-5 .< z_data ./ σ_z .< 5]) ./ σ_z
        E_normalized = @view(E_data[.-120 .< E_data ./ σ_E .< 120]) ./ σ_E
        
        # Compute KDE with optimized parameters
        z_densities[i] = kde(z_normalized, boundary=(-5,5), npoints=100)
        E_densities[i] = kde(E_normalized, boundary=(-120,120), npoints=200)
        next!(p)
    end
    
    return z_densities, E_densities
end
"""

"""
function all_animate_optimized(
    n_turns::Int64,
    particles_out::BeamTurn{Float64},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64,
    filename::String="all_animation_optimized.mp4")
    # Create output directory
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    filename = joinpath(folder_storage, filename)

    # Precompute separatrix (unchanged)
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    
    # Precompute densities
    z_densities, E_densities = precompute_densities(particles_out, σ_z, σ_E)
    
    # Create static figure components
    println("Setting up figure...")
    fig = Figure(;size=(1400, 900), font="Arial")
    ax_z = Axis(fig[1, 1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
    ax_E = Axis(fig[1, 2], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
    ax_phase = Axis(fig[2, 1:2], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
    title_label = Label(fig[0, :], "Turn 1", fontsize=20, halign=:center)
    
    # Set static limits
    ylims!(ax_z, (0, 1))
    ylims!(ax_E, (0, .1))
    xlims!(ax_z, (-5, 5))
    xlims!(ax_E, (-120, 120))
    xlims!(ax_phase, (0, 3π/2))
    ylims!(ax_phase, (minimum(boundary_points[2])/σ_E - 4,maximum(boundary_points[2])/σ_E + 4))
    
    
    # Prepare static observables for better performance
    boundary_obs = Observable((boundary_points[1], boundary_points[2]/σ_E))
    
    # Create progress meter
    p = Progress(n_turns, desc="Generating animation: ")
    
    # Record animation with optimized frame generation
    @inbounds record(fig, filename, 1:n_turns; framerate=60) do frame_idx
        empty!(ax_z)
        empty!(ax_E)
        empty!(ax_phase)
        
        # Get current frame data
        frame_data = particles_out[frame_idx]
        
        # Plot phase space
        scatter!(ax_phase, frame_data.ϕ, frame_data.ΔE/σ_E, color=:black, markersize=1)
        
        # Plot separatrix
        
        lines!(ax_phase, boundary_obs[][1], boundary_obs[][2], color=:red)
        
        # Plot pre-computed densities
        z_density = z_densities[frame_idx]
        E_density = E_densities[frame_idx]
        
        lines!(ax_z, z_density.x, z_density.density, color=:red, linewidth=2)
        lines!(ax_E, E_density.x, E_density.density, color=:red, linewidth=2)
        
        # Update turn number
        title_label.text = "Turn $frame_idx"
        
        # Update progress
        next!(p)
    end
    println("Animation complete!")
end ;

function create_animation_from_pngs(plots_vector, n_particles; fps=60, filename="animation.mp4")
    # Create the output directory if it doesn't exist
    dir_frames = "frames"
    folder_storage = "Haissinski/particle_sims/turns$(length(plots_vector)-1)_particles$(n_particles)"
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    dir = joinpath(folder_storage, timestamp, dir_frames)
    mkpath(dir)
    filename = joinpath(folder_storage, timestamp, filename)
    
    p2 = Progress(length(plots_vector), desc="Generating Frames: ")
    # Save each plot as a PNG
    n = length(plots_vector)
    chunks = Iterators.partition(1:n, ceil(Int, n/Threads.nthreads()))
    
    Threads.@threads for chunk in collect(chunks)
        for i in chunk
            save(joinpath(dir, "frame_$i.png"), plots_vector[i])
            next!(p2)
        end
    end
    
    # Create a figure for the animation
    fig = Figure()
    ax = Axis(fig[1,1])
    hidedecorations!(ax)  # Hide axis decorations
    hidespines!(ax)      # Hide spines
    p = Progress(length(plots_vector), desc="Generating animation: ")
    # Create animation from saved PNGs
    record(fig, filename, 1:length(plots_vector), framerate=fps) do frame
        empty!(ax)
        img = load(joinpath(dir, "frame_$frame.png"))
        image!(ax, rotr90(img))  # rotate if needed
        next!(p)
    end
    
    # Optional: remove the temporary PNG files
    rm(dir, recursive=true)
    println("Animation complete!")
end

#this one allows you to skip any number of turns between frames  easily
function animate_one_by_one(
    n_turns::Int64, particle_states::StructArray{ParticleState{Float64}},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64, E0::Float64, σ_E::Float64, σ_z::Float64, acc_radius::Float64,
    freq_rf::Float64,
    pipe_radius::Float64,; frame_itv::Int=1,
    filename::String="anim_1_by_1.mp4")

    # Create output directory
    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(particle_states))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)
    filename = joinpath(folder_storage, filename)

    # Precompute separatrix (unchanged)
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    
    # Create static figure components
    println("Setting up figure...")
    fig = Figure(;size=(1400, 900), font="Arial")
    ax_z = Axis(fig[1, 1], xlabel=L"\frac{z}{\sigma _{z}}", ylabel="Count")
    ax_E = Axis(fig[1, 2], xlabel=L"\frac{\Delta E}{\sigma _{E}}", ylabel="Count")
    ax_phase = Axis(fig[2, 1:2], xlabel=L"\phi", ylabel=L"\frac{\Delta E}{\sigma _{E}}")
    title_label = Label(fig[0, :], "Turn 1", fontsize=20, halign=:center)
    
    # Set static limits
    ylims!(ax_z, (0, 1))
    ylims!(ax_E, (0, .1))
    xlims!(ax_z, (-5, 5))
    xlims!(ax_E, (-120, 120))
    boundary_obs = Observable((boundary_points[1], boundary_points[2]/σ_E))
    xlims!(ax_phase, (0, 3π/2))
    ylims!(ax_phase, (minimum(boundary_obs[][2]), maximum(boundary_obs[][2])))
    
    
    n_turns = Int(n_turns/frame_itv)
    p = Progress(n_turns, desc="Generating animation: ")
    # Record animation with optimized frame generation
    @inbounds record(fig, filename, 1:n_turns; framerate=60) do frame_idx
        empty!(ax_z)
        empty!(ax_E)
        empty!(ax_phase)
        
        # Get current frame data
        z_data = particle_states.z
        E_data = particle_states.ΔE
        
        # Apply masks and normalize
        z_normalized = @view(z_data[.-5 .< z_data ./ σ_z .< 5]) ./ σ_z
        E_normalized = @view(E_data[.-120 .< E_data ./ σ_E .< 120]) ./ σ_E
        
        # Compute KDE with optimized parameters
        z_densities = kde(z_normalized, boundary=(-5,5), npoints=100)
        E_densities = kde(E_normalized, boundary=(-120,120), npoints=200)
        
        # Plot phase space
        scatter!(ax_phase, particle_states.ϕ, particle_states.ΔE/σ_E, color=:black, markersize=1)
        
        # Plot separatrix
        lines!(ax_phase, boundary_obs[][1], boundary_obs[][2], color=:red)

        z_density = z_densities
        E_density = E_densities
        
        lines!(ax_z, z_density.x, z_density.density, color=:red, linewidth=2)
        lines!(ax_E, E_density.x, E_density.density, color=:red, linewidth=2)
        
        # Update turn number
        title_label.text = "Turn $frame_idx"
        longitudinal_evolve!(
        frame_itv, particle_states, ϕs, α_c, mass, voltage,
        harmonic, acc_radius, freq_rf, pipe_radius, E0, σ_E,
        use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
        use_excitation=true, display_counter=false)
        # Update progress
        next!(p)
    end
    println("Animation complete!")
end ;


##############################################################################################################


energy = 4e9 ;
mass = MASS_ELECTRON ;
voltage = 5e6 ;
harmonic = 360 ;
radius = 250. ;
pipe_radius = .00025 ;


α_c = 3.68e-4 ;
γ = energy/mass ;
β = sqrt(1 - 1/γ^2) ;
η = α_c - 1/γ^2 ;
sin_ϕs = 0.5 ;
ϕs = 5π/6 ;
freq_rf = 180.15e7 ;

σ_E = 1e6 ;
μ_E = 0. ;
μ_z = 0. ;
ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT)) ;
σ_z = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*energy/harmonic/voltage/abs(cos(ϕs))) * σ_E / energy ;


n_turns = 1000;
particle_states = generate_particles(μ_z, μ_E, σ_z,σ_E, Int64(1e5),energy,mass,ϕs, freq_rf) ;

particles_out, plot_potential, plot_WF= longitudinal_evolve(
    n_turns, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=true,plot_WF=true, write_to_file=false, output_file="test1.h5") ; 
plot_potential[1]
plot_potential[65] 
plot_potential[100]
plot_WF[1]
plot_WF[65]
plot_WF[100]


path = "Haissinski/particle_sims/turns$(n_turns)_particles$(length(results[1]))"
timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
path = joinpath(path, timestamp)
filename = joinpath(path, "histogram_method.png")
mkpath(path)
save(filename, plot_potential[10])


particles_out = longitudinal_evolve(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5") ;

all_animate(100, results, ϕs, α_c, mass, voltage, harmonic,energy, σ_E, σ_z, "1e5p_1e2t_4.mp4")
@btime all_animate(100, results, ϕs, α_c, mass, voltage, harmonic,energy, σ_E, σ_z, "1e5p_1e2t_3.mp4") #73.517 s (80880209 allocations: 3.24 GiB), 1e5 particles, 1e2 turns

all_animate_optimized(1000, particles_out, ϕs, α_c, mass, voltage, harmonic,energy, σ_E, σ_z, "opt_anim.mp4")
#Maybe only plot every 5 or 10 turns, also redo how I find the mask z and E values, and use phi to find the mask values.




function filter_outliers(trial::BenchmarkTools.Trial; percentile=0.90)
    cutoff = quantile(trial.times, percentile)
    keep = trial.times .<= cutoff
    
    filtered = deepcopy(trial)
    filtered.times = trial.times[keep]
    filtered.gctimes = trial.gctimes[keep]
    
    return filtered
end

@btime generate_particles(μ_z, μ_E, σ_z,σ_E, Int64(1e5),energy,mass,ϕs, freq_rf) ; #187.400 ms (200351 allocations: 12.09 MiB)
particle_states = generate_particles(μ_z, μ_E, σ_z,σ_E, Int64(1e5),energy,mass,ϕs, freq_rf) 

@btime longitudinal_evolve(
    $100, $particle_states, $ϕs, $α_c, $mass, $voltage,
    $harmonic, $radius, $freq_rf, $pipe_radius, $energy, $σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5");
    #251.623 ms (19274 allocations: 256.85 MiB), 1e5 particles, 1e2 turns
    #2.783 s (22648 allocations: 2.61 GiB), 1e6 particles, 1e2 turns
    #3.400 s (203094 allocations: 2.55 GiB), 1e5 particles, 1e3 turns
    #33.730 s (236678 allocations: 25.47 GiB), 1e6 particles, 1e3 turns !! Takes a bit to get started, but once it does, it's fast


benchmark_result = @benchmark longitudinal_evolve(
        $100, $particle_states, $ϕs, $α_c, $mass, $voltage,
        $harmonic, $radius, $freq_rf, $pipe_radius, $energy, $σ_E,
        use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
        use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5")

filtered_result = filter_outliers(benchmark_result)


@ProfileSVG.profview longitudinal_evolve(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false,plot_WF=false, write_to_file=false, output_file="test1.h5") 
# z_hist, E_hist = histogram_particle_data(particles_out,10 ,save_figs = false)
# z_hist

particle_states = generate_particles(μ_z, μ_E, σ_z,σ_E, Int64(1e6),energy,mass,ϕs, freq_rf) ;
longitudinal_evolve!(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true);


@btime longitudinal_evolve!(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true);
    #208.964 ms (10266 allocations: 36.96 MiB), 1e5 particles, 1e2 turns
    #2.533 s (13141 allocations: 340.81 MiB), 1e6 particles, 1e2 turns
    #2.072 s (106745 allocations: 319.96 MiB), 1e5 particles, 1e3 turns
    #25.148 s (138145 allocations: 3.29 GiB), 1e6 particles, 1e3 turns 

@benchmark longitudinal_evolve!(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true) samples=100 seconds = 500

filtered_result = filter_outliers(benchmark_result)

Profile.clear()
@ProfileSVG.profview longitudinal_evolve!(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true)



particle_states = generate_particles(μ_z, μ_E, σ_z,σ_E, Int64(1e5),energy,mass,ϕs, freq_rf) ;

animate_one_by_one(1000, particle_states, ϕs, α_c, mass, voltage, harmonic,energy, σ_E, σ_z, radius, freq_rf, pipe_radius, frame_itv=1, filename = "anim_1_by_1.mp4")


scatter(particle_states.z / σ_z, particle_states.ΔE / σ_E, color=:black, markersize=1)
scatter(particle_states.ϕ, particle_states.ΔE / σ_E, color=:black, markersize=1)

scatter_plots = longitudinal_evolve!(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_scatter=true);

scatter_plots[100]







create_animation_from_pngs(scatter_plots, 100000, filename="scatter_fast.mp4")
