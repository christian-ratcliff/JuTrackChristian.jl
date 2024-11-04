"""
    Module: Beam Evolution Simulation
    
This code implements a high-performance beam evolution simulation for particle accelerators,
including functionality for particle generation, evolution tracking, and data visualization.
"""

using Distributions ;
using Random ;
Random.seed!(061101) ;
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

# Physical constants
const SPEED_LIGHT = 299792458 ;
const ELECTRON_CHARGE = 1.602176634e-19 ;
const MASS_ELECTRON = 0.51099895069e6 ;

#=
Core Data Structures
=#

"""
    ParticleState{T<:AbstractFloat}

Immutable structure representing the state of a single particle in the beam.

Fields:
- `z::T`: Longitudinal position
- `ΔE::T`: Energy deviation from reference
- `ϕ::T`: Phase relative to RF
"""
struct ParticleState{T<:AbstractFloat}
    z::T
    ΔE::T
    ϕ::T
end ;

"""
    SimulationBuffers{T<:AbstractFloat}

Pre-allocated buffers for efficient computation during simulation.

Fields:
- `WF::Vector{T}`: Wakefield calculations
- `potential::Vector{T}`: Potential energy calculations
- `Δγ::Vector{T}`: Gamma factor deviations
- `η::Vector{T}`: Slip factor calculations
- `coeff::Vector{T}`: Temporary coefficients
- `temp_z`, `temp_ΔE`, `temp_ϕ`: General temporary storage
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
end ;

#=
High-Level Simulation Functions
=#

"""
    longitudinal_evolve(n_turns, particle_states, ϕs, α_c, mass, voltage, harmonic, 
                       acc_radius, freq_rf, pipe_radius, E0, σ_E; kwargs...) 
                       -> Union{Vector{Tuple}, Tuple{Vector{Tuple}, Vector{Any}}}

Main simulation function for longitudinal beam evolution.

Parameters:
- `n_turns`: Number of turns to simulate
- `particle_states`: Initial particle states
- `ϕs`: Synchronous phase
- `α_c`: Momentum compaction factor
- `mass`: Particle mass
- `voltage`: RF voltage
- `harmonic`: RF harmonic number
- `acc_radius`: Accelerator radius
- `freq_rf`: RF frequency
- `pipe_radius`: Beam pipe radius
- `E0`: Reference energy
- `σ_E`: Energy spread

Optional kwargs:
- `update_η`: Enable slip factor updates
- `update_E0`: Enable reference energy updates
- `SR_damping`: Enable synchrotron radiation damping
- `use_excitation`: Enable quantum excitation
- `use_wakefield`: Enable wakefield effects
- `plot_potential`: Enable potential plotting
- `write_to_file`: Enable data writing to file
- `output_file`: Output file path
- `additional_metadata`: Additional simulation metadata

Returns:
- Vector of particle states for each turn
- Optional potential plots if enabled
"""
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
    buffers = create_simulation_buffers(n_particles, T)
    
    # Pre-allocate master storage with fixed size arrays
    master_storage = [
        (Vector{T}(undef, n_particles),
        Vector{T}(undef, n_particles),
        Vector{T}(undef, n_particles))
        for _ in 1:n_turns+1
    ]
    
    

    # Initialize phases using pre-allocated buffer
    @turbo for i in 1:n_particles
        particle_states.ϕ[i] = -(particle_states.z[i] * rf_factor - ϕs)
    end
    
    # Store initial state
    copyto!(master_storage[1][1], particle_states.z)
    copyto!(master_storage[1][2], particle_states.ϕ)
    copyto!(master_storage[1][3], particle_states.ΔE)

# Initialize file and write metadata if writing is enabled
    if write_to_file
        timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
        folder_storage = "/home/ratcliff/JuTrackChristian.jl/Haissinski/particle_sims/turns$(n_turns)_particles$(n_particles)"
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
    
    # Pre-allocate wakefield parameters if needed
    if use_wakefield
        kp = T(3e1)
        Z0 = T(120π)
        cτ = T(4e-3)
        wake_factor = Z0 * SPEED_LIGHT / (π * pipe_radius^2)
        wake_sqrt = sqrt(2*kp/pipe_radius)
    end

    potential_plots = plot_potential ? Vector{Any}(undef, n_turns) : nothing
    
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
            println(randn!(buffers.potential))
            @turbo for i in 1:n_particles
                particle_states.ΔE[i] += excitation * buffers.potential[i]
            end
        end
        
        # Apply wakefield effects if enabled
        if use_wakefield
            if plot_potential
                # Store potential plot for this turn
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles
                )
                potential_plots[turn] = scatter(
                    particle_states.z,
                    buffers.potential / σ_E,
                    markersize = 3
                )
            else
                apply_wakefield_inplace!(
                    particle_states, buffers, wake_factor, wake_sqrt, cτ,
                    E0, acc_radius, n_particles
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


        # Store current state using pre-allocated arrays
        copyto!(master_storage[turn+1][1], particle_states.z)
        copyto!(master_storage[turn+1][2], particle_states.ϕ)
        copyto!(master_storage[turn+1][3], particle_states.ΔE)
        if write_to_file
            h5open(output_file, "r+") do file
                file["z"][:, turn + 1] = particle_states.z
                file["phi"][:, turn + 1] = particle_states.ϕ
                file["dE"][:, turn + 1] = particle_states.ΔE
            end
        end

    end
    if plot_potential
        return master_storage, potential_plots
    else
        return master_storage
    end
end ;

"""
    generate_particles(μ_z, μ_E, σ_z, σ_E, num_particles, α_c, E_ini, harmonic, 
                      voltage, ϕs, radius) -> StructArray{ParticleState{T}}

Generate initial particle distribution for simulation.

Parameters:
- `μ_z`, `μ_E`: Mean position and energy
- `σ_z`, `σ_E`: Position and energy spread
- `num_particles`: Number of particles to generate

Returns:
- StructArray of initial particle states
"""
function generate_particles(
    μ_z::T,
    μ_E::T,
    σ_z::T,
    σ_E::T,
    num_particles::Int
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
    
    # Generate particles in parallel
    chunk_size = num_particles ÷ nthreads()
    
    @threads for thread_idx in 1:nthreads()
        # Calculate range for this thread
        start_idx = (thread_idx - 1) * chunk_size + 1
        end_idx = thread_idx == nthreads() ? num_particles : thread_idx * chunk_size
        
        # Local RNG for thread safety
        local_rng = Random.default_rng()
        
        # Generate particles for this chunk
        for i in start_idx:end_idx
            sample_vec = rand(local_rng, dist_total)
            particle_states.z[i] = sample_vec[1]
            particle_states.ΔE[i] = sample_vec[2]
        end
    end
    return particle_states
end ;

#=
Data Management Functions
=#

"""
    write_particle_evolution(filename, particle_states; metadata) -> Nothing

Write particle evolution data to HDF5 file.

Parameters:
- `filename`: Output file path
- `particle_states`: Vector of particle states for each turn
- `metadata`: Optional dictionary of metadata
"""
function write_particle_evolution(filename::String, 
                                particle_states::Vector{StructArray{ParticleState{T}}};
                                metadata::Dict{String, Any}=Dict{String, Any}()) where T<:AbstractFloat
    n_turns = length(particle_states)
    n_particles = length(particle_states[1].z)

    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "/home/ratcliff/JuTrackChristian.jl/Haissinski/particle_sims/turns$(n_turns)_particles$(n_particles)"
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
    read_particle_evolution(filename; turn_range) 
    -> Tuple{Vector{StructArray{ParticleState{T}}}, Dict{String, Any}}

Read particle evolution data from HDF5 file.

Parameters:
- `filename`: Input file path
- `turn_range`: Optional range of turns to read

Returns:
- Tuple of particle states and metadata
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
    read_turn_range(filename, turn_range) -> StructArray{ParticleState{T}}

Read specific turn range from HDF5 file.

Parameters:
- `filename`: Input file path
- `turn_range`: Range of turns to read

Returns:
- StructArray of particle states for specified turns
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
    make_separatrix(ϕs, voltage, energy, harmonic, η, β) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate separatrix coordinates for phase space visualization.

Parameters:
- `ϕs`: Synchronous phase
- `voltage`: RF voltage
- `energy`: Reference energy
- `harmonic`: RF harmonic number
- `η`: Slip factor
- `β`: Relativistic beta

Returns:
- Tuple of phase and energy coordinates for separatrix
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
    all_animate(n_turns, particles_out, ϕs, α_c, mass, voltage, harmonic, acc_radius, 
               freq_rf, pipe_radius, E0, σ_E, σ_z, filename) -> Nothing

Generate animation of beam evolution.

Parameters:
- `n_turns`: Number of turns to animate
- `particles_out`: Vector of particle states for each turn
- Additional parameters matching simulation parameters
- `filename`: Output animation file path
"""
function all_animate(n_turns::Int64,
    particles_out::Vector{Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}}},
    ϕs::Float64, α_c::Float64, mass::Float64, voltage::Float64,
    harmonic::Int64, acc_radius::Float64, freq_rf::Float64,
    pipe_radius::Float64, E0::Float64, σ_E::Float64, σ_z::Float64, filename::String="all_animation.mp4")

    timestamp = string(Dates.format(Dates.now(), "yyyy-mm-dd"))
    folder_storage = "/home/ratcliff/JuTrackChristian.jl/Haissinski/particle_sims/turns$(n_turns)_particles$(length(particles_out[1][1]))"
    folder_storage = joinpath(folder_storage, timestamp)
    mkpath(folder_storage)

    filename = joinpath(folder_storage, filename)

    # Precompute separatrix
    γ = E0/mass
    β = sqrt(1 - 1/γ^2)
    η = α_c - 1/γ^2
    boundary_points = make_separatrix(ϕs, voltage, E0, harmonic, η, β)
    println("Starting animation generation...")
    
    # Get initial data to calculate fixed bin widths
    # initial_z_data = particles_out[1][1] / σ_z  # Normalized z data
    # initial_E_data = particles_out[1][3] / σ_E  # Normalized E data
    
    # Calculate fixed bin edges using the display limits instead of just initial data
    n_bins = 10^(ceil(Int, log10(length(particles_out[1][1])))-2)
    z_bin_width = (5 - (-5)) / n_bins  # Since xlims is (-5, 5)
    E_bin_width = (120 - (-120)) / n_bins  # Since xlims is (-120, 120)
    
    # Create fixed bin edges spanning the full display range
    z_bins = range(-5, 5, step=z_bin_width)
    E_bins = range(-120, 120, step=E_bin_width)
    
    println("Z bin width: ", z_bin_width)
    println("E bin width: ", E_bin_width)
    println("Number of Z bins: ", length(z_bins)-1)
    println("Number of E bins: ", length(E_bins)-1)
    
    # Modified histogram calculation function for fixed bins
    function calculate_fixed_histogram(data::Vector{Float64}, bins::AbstractRange)
        hist = fit(Histogram, data, bins)
        centers = (bins[1:end-1] + bins[2:end]) ./ 2
        return centers, hist.weights
    end
    
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
        z_data, ϕ_data, E_data = particles_out[frame_idx]
        
        # Plot phase space
        scatter!(ax_phase, ϕ_data, E_data/σ_E, color=:black, markersize=1)
        lines!(ax_phase, boundary_points[1], boundary_points[2]/σ_E, color=:red)
        
        # Plot histograms and KDEs
        z_normalized = z_data / σ_z
        E_normalized = E_data / σ_E
        
        # Z distribution with fixed bins
        z_centers, z_counts = calculate_fixed_histogram(z_normalized, z_bins)
        z_kde_x, z_kde_y = calculate_kde(z_normalized)
        barplot!(ax_z, z_centers, z_counts, color=(:red, 0.5))
        lines!(ax_z, z_kde_x, z_kde_y .* length(z_normalized) .* z_bin_width,
            color=:green, linewidth=2)
        
        # Energy distribution with fixed bins
        E_centers, E_counts = calculate_fixed_histogram(E_normalized, E_bins)
        E_kde_x, E_kde_y = calculate_kde(E_normalized)
        barplot!(ax_E, E_centers, E_counts, color=(:red, 0.5))
        lines!(ax_E, E_kde_x, E_kde_y .* length(E_normalized) .* E_bin_width,
            color=:green, linewidth=2)
        
        # Set limits and labels
        xlims!(ax_z, (-5, 5))
        xlims!(ax_E, (-120, 120))
        xlims!(ax_phase, (0, 3π/2))
        ylims!(ax_phase, (-400, 400))
        
        # Set fixed y limits for both histogram plots
        ylims!(ax_z, (0, FIXED_Y_MAX))
        ylims!(ax_E, (0, FIXED_Y_MAX))
        
        # Update turn number
        title_label.text = "Turn $frame_idx"
        
        if frame_idx % 25 == 0
            println("Processed frame $frame_idx of $n_turns")
        end
    end
    println("Animation complete!")
end ;

"""
    calculate_histogram(data, bins) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate histogram for distribution visualization.

Parameters:
- `data`: Input data vector
- `bins`: Number of histogram bins

Returns:
- Tuple of bin centers and counts
"""
function calculate_histogram(data::Vector{Float64}, bins::Int64)
    hist = fit(Histogram, data, nbins=bins)
    centers = (hist.edges[1][1:end-1] + hist.edges[1][2:end]) ./ 2
    return centers, hist.weights
end ;

"""
    calculate_kde(data, bandwidth=nothing) -> Tuple{Vector{Float64}, Vector{Float64}}

Calculate kernel density estimate for distribution visualization.

Parameters:
- `data`: Input data vector
- `bandwidth`: Optional bandwidth parameter

Returns:
- Tuple of x coordinates and density values
"""
function calculate_kde(data::Vector{Float64}, bandwidth=nothing)
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

Parameters:
- `n_particles`: Number of particles
- `T`: Numeric type for calculations

Returns:
- SimulationBuffers struct with pre-allocated arrays
"""
function create_simulation_buffers(n_particles::Int, T::Type=Float64)
    SimulationBuffers{T}(
        Vector{T}(undef, n_particles),  # WF
        Vector{T}(undef, n_particles),  # potential
        Vector{T}(undef, n_particles),  # Δγ
        Vector{T}(undef, n_particles),  # η
        Vector{T}(undef, n_particles),  # coeff
        Vector{T}(undef, n_particles),  # temp_z
        Vector{T}(undef, n_particles),  # temp_ΔE
        Vector{T}(undef, n_particles),   # temp_ϕ
    )
end ;

"""
    apply_wakefield_inplace!(particle_states, buffers, wake_factor, wake_sqrt, cτ, 
                           E0, acc_radius, n_particles) -> Nothing

Apply wakefield effects to particle states in-place.

Parameters:
- `particle_states`: Current particle states
- `buffers`: Pre-allocated calculation buffers
- Additional wakefield parameters
"""
function apply_wakefield_inplace!(
    particle_states::StructArray{ParticleState{T}},
    buffers::SimulationBuffers{T},
    wake_factor::T,
    wake_sqrt::T,
    cτ::T,
    E0::T,
    acc_radius::T,
    n_particles::Int) where T<:AbstractFloat
    
    sort!(particle_states, by=x->x.z)
    # Calculate wakefield using branchless operations
    @tturbo for i in 1:n_particles
        z = particle_states.z[i]
        # Use multiplication by boolean instead of if/else
        is_negative = z < zero(T)
        wake_term = wake_factor * 
                   exp(z / cτ) * 
                   cos(wake_sqrt * (-z))
        buffers.WF[i] = is_negative * wake_term
    end
    
    
    
    # Apply wakefield kick using SIMD
    scale_factor = (n_particles * ELECTRON_CHARGE) / (E0 * 2π * acc_radius) * 
                    (1e11/floor(Int, log10(n_particles)))

    # Efficient in-place convolution
    convolve_inplace!(buffers.potential, buffers.WF, particle_states.z, n_particles)

    # perm = sortperm(particle_states, by=x->x.z)
    # sorted_particles = particle_states[perm]
    # sorted_potential = buffers.potential[perm]


    # itp = LinearInterpolation(sorted_particles.z, sorted_potential, extrapolation_bc=Flat())
    itp = LinearInterpolation(particle_states.z, buffers.potential, extrapolation_bc=Flat())
    # itpz = itp.(sorted_particles.z)
    # itpz = itp.(particle_states.z)
    @turbo for i in 1:n_particles
        # particle_states.ΔE[i] -= buffers.potential[i] * scale_factor
        # sorted_particles.ΔE[i] -= itpz[i] * scale_factor
        # particle_states.ΔE[i] -= itpz[i] * scale_factor
        particle_states.ΔE[i] -= itp(particle_states.z[i]) * scale_factor
    end
    # inv_perm = invperm(perm)
    # particle_states = sorted_particles[inv_perm]
end ;

# function interp(
#     particle_states::StructArray{ParticleState{T}},
#     buffers::SimulationBuffers{T}) where T<:AbstractFloat
    
#     spl = Spline1D(particle_states.z, buffers.potential; k=2)  # k=1 for linear interpolation
#     return evaluate(spl, particle_states.z)
# end ;


"""
    convolve_inplace!(output, signal, positions, n_particles) -> Nothing

Perform in-place convolution for wakefield calculations.

Parameters:
- `output`: Pre-allocated output buffer
- `signal`: Input signal vector
- `positions`: Particle positions
- `n_particles`: Number of particles
"""
function convolve_inplace!(
    output::Vector{T},
    signal::Vector{T},
    positions::Vector{T},
    n_particles::Int
) where T<:AbstractFloat
    
    fill!(output, zero(T))
    Threads.@threads for i in 1:n_particles
        sum_val = zero(T)
        for j in 1:i-1
            # dx = positions[i] - positions[j]
            dx = delta( positions[i] - positions[j], 9.9999e-3)
            # is_positive = dx > zero(T)
            # is_positive = 1
            # sum_val += signal[j] * is_positive
            sum_val += signal[j] * dx
        end
        output[i] = sum_val  #curve up
        # output[i] = -sum_val  #curve down
    end
end ;

"""
    delta(x::T, ϵ::T) where T<:AbstractFloat -> T

Calculate delta function for wakefield smoothing.

Parameters:
- `x`: Input value
- `ϵ`: Smoothing parameter

Returns:
- Smoothed delta function value

"""
@inline function delta(x::T, ϵ::T) where T<:AbstractFloat
    inv_π = T(0.31830988618379067)  # 1/π pre-computed
    return (ϵ * inv_π) / (x * x + ϵ * ϵ)
end ;




energy = 4e9 ;
mass = MASS_ELECTRON ;
voltage = 5e6 ;
harmonic = 360 ;
radius = 250. ;
pipe_radius = .00025 ;
n_turns = 100 ;

α_c = 3.68e-4 ;
γ = energy/mass ;
β = sqrt(1 - 1/γ^2) ;
η = α_c - 1/γ^2 ;
sin_ϕs = 0.5 ;
ϕs = 5π/6 ;
freq_rf = 280.15e7 ;

σ_E = 1e6 ;
μ_E = 0. ;
μ_z = 0. ;
ω_rev = 2 * π / ((2*π*radius) / (β*SPEED_LIGHT)) ;
σ_z = sqrt(2 * π) * SPEED_LIGHT / ω_rev * sqrt(α_c*energy/harmonic/voltage/abs(cos(ϕs))) * σ_E / energy ;



particle_states = generate_particles(μ_z, μ_E, σ_z,σ_E, 100000) ;
results, plots= longitudinal_evolve(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=true, write_to_file=false, output_file="test1.h5") ;
all_animate(1000, results, ϕs, α_c, mass, voltage, harmonic, radius, freq_rf, pipe_radius, energy, σ_E, σ_z, "test2.mp4")

plots[1]

@btime generate_particles(μ_z, μ_E, σ_z,σ_E, 10000) ; #14.216 ms (20351 allocations: 1.79 MiB)
@btime generate_particles(μ_z, μ_E, σ_z,σ_E, 100000) ; #198.547 ms (200351 allocations: 12.09 MiB)

@btime longitudinal_evolve(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false, write_to_file=false, output_file="test1.h5");  
    #975.307 ms (43565 allocations: 84.34 MiB), 1e4 particles, 1e2 turns

@btime longitudinal_evolve(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false, write_to_file=false, output_file="test1.h5");
    #11.548 s (436323 allocations: 835.91 MiB), 1e4 particles, 1e3 turns

@btime longitudinal_evolve(
    100, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false, write_to_file=false, output_file="test1.h5"); 
    #43.233 s (47249 allocations: 852.39 MiB), 1e5 particles, 1e2 turns

@btime longitudinal_evolve(
    1000, particle_states, ϕs, α_c, mass, voltage,
    harmonic, radius, freq_rf, pipe_radius, energy, σ_E,
    use_wakefield=true, update_η=true, update_E0=true, SR_damping=true,
    use_excitation=true, plot_potential=false, write_to_file=false, output_file="test1.h5"); 
    #659.240 s (435066 allocations: 8.25 GiB), 1e5 particles, 1e3 turns

time_taken = [.975307, 11.548, 43.233, 659.240]
particle_amount = [1e4, 1e4, 1e5, 1e5]
turn_amount = [1e2, 1e3, 1e2, 1e3]
allocations = [43565, 436323, 47249, 435066]
memory = [84.34, 835.91, 852.39, 8250]