module BuildAggregate

include("Metrics.jl")
include("RandomWalk.jl")
using .Metrics
using .RandomWalk
using Random
using Statistics
using LinearAlgebra

export individuallyadded_aggregate!

function individuallyadded_aggregate!(cubes::Matrix{Integer},number_of_cubes::Integer,dimensionality::Integer,
    deltaR::Float64,steplength::Integer,attaching_distance::Integer)
        
    """
        individuallyadded_aggregate!(cubes::Matrix{Integer},number_of_cubes::Integer,dimensionality::Integer,
            deltaR::Float64,steplength::Integer,attaching_distance::Integer)

    This function implements a basic diffusion-limited individually-added aggregation routine on a 3D lattice
    Relevant reference: https://journals.aps.org/prfluids/abstract/10.1103/PhysRevFluids.5.044305
    # Arguments
    - cubes::Matrix{Integer}        : matrix containing the cubes that make up one aggregate
    - number_of_cubes::Integer      : number of cubes in the aggregate
    - dimensionality::Integer       : dimensionality of the problem (3D in the current setting)
    - deltaR::Float64               : number to be added to the radius of the sphere where we generate each random walker as aggregates increases in size
    - steplength::Integer           : length of each step that any single cube is allowed to take as it walks towards the stationary aggregate
    - attaching_distance::Integer   : how far we allow the walking cube to be in order for it to attach to a cube that is part of the aggregate
    """

    current_center_of_mass::Vector{Float64} = zeros(dimensionality)
    for current_cube = 2:number_of_cubes # loop starts at 2, because cube-1 is stationary at the origin
        current_aggregate::Matrix{Integer} = cubes[1:current_cube,:]
        current_center_of_mass = vec(compute_centerofmass(current_aggregate,dimensionality))

        radius::Float64 = compute_maxdistance(current_aggregate,current_center_of_mass)
        my_radius::Float64 = radius + deltaR
        
        current_walker::Vector{Integer} = zeros(dimensionality)
        cube_isfar::Bool = true
        while cube_isfar == true
            randomnumber::Float64 = rand(Float64)
            current_walker = select_initialposition(current_center_of_mass,dimensionality,my_radius,randomnumber)
            cube_isapproaching::Bool = true
            while cube_isapproaching == true
                randomstep::Float64 = rand(Float64)
                direction::String = throwdice(randomstep)
                step_taken::Vector{Integer} = stepvector(direction,steplength,dimensionality)
                current_walker = takestep!(current_walker,step_taken)
                current_distance::Float64 = norm(current_walker-current_center_of_mass)
                if current_distance > (my_radius + deltaR)
                    cube_isfar = true
                    cube_isapproaching = false
                    break
                end
                # check distance between cube-walker and all the cubes in the aggregates
                for closest_cube = 1:number_of_cubes
                    if round(norm(current_walker-cubes[closest_cube,:])) == attaching_distance 
                        cube_isapproaching = false  # cube-walker stops walking
                        cube_isfar = false          # cube-walker has attached
                        break
                    end
                end
            end
        end
        cubes[current_cube,:] = current_walker
    end

    return cubes
end

end