using DifferentialEquations, Dates

include("asm1.jl")

DEFAULT_INFLUENT_FILE_PATH = datadir("influent_files", "dryinfluent.ascii")

mutable struct ODECore

    current_t::Float64
    fixed_dt::Float64
    ode_fct!::Function
    state::Vector{Float64}
    params_vec::NamedTuple
    index_u::Vector{Int64}

    # Constructor with all the arguments
    function ODECore(current_t, fixed_dt, ode_fct!, state, params_vec, index_u)

        return new(current_t, fixed_dt, ode_fct!, state, params_vec, index_u)

    end

    # Default constructor with no arguments
    function ODECore()

        # Initialize simulator
        simulator = new(0.0, 1/1440, asm1!, [], NamedTuple(), [14])

        # Reset simulator
        reset!(simulator)

        return simulator

    end

end

function reset!(env::ODECore; init_state=Nothing, params_vec=Nothing, influent_file_path=DEFAULT_INFLUENT_FILE_PATH)

    default_p, default_state_init = get_default_parameters_asm1(influent_file_path=influent_file_path)

    env.current_t = 0.0
    env.state = init_state == Nothing ? default_state_init : init_state
    env.params_vec = params_vec == Nothing ? default_p : params_vec

end

function step!(env::ODECore, action::Union{Vector{Any}, CallbackSet})

    # Copy init state
    init_state = deepcopy(env.state)

    if isa(action, Vector{Float64}) || isa(action, Vector{Int64})
        
        # Check if the size of action is the same as env.index_u
        @assert size(action, 1) == size(env.index_u, 1) "The size of the action need to be the same as the index_u of env."

        # Update control index of init state with choosen action
        init_state[env.index_u] = action

    end

    # Define ODEProblem
    ode_problem = ODEProblem(env.ode_fct!, init_state, (env.current_t,env.current_t + env.fixed_dt), env.params_vec)

    # Compute the new state
    if isa(action, CallbackSet)
        sim_results = solve(ode_problem, saveat=[env.current_t + env.fixed_dt], alg_hints=[:stiff], callback=action)
    else
        sim_results = solve(ode_problem, saveat=[env.current_t + env.fixed_dt], alg_hints=[:stiff])
    end

    # Update the time
    env.current_t += env.fixed_dt

    # Update the state
    env.state = sim_results.u[1]

    return sim_results

end

function multi_step!(env::ODECore, action::Union{Vector{Vector{Float64}}, CallbackSet}, n_steps::Union{Int64, Period})

    # Convert period to timestep if it is the case
    n_steps = isa(n_steps, Period) ? Int(Dates.value(convert(Dates.Second, n_steps))/(env.fixed_dt*(24*60*60))) : n_steps

    if isa(action, Vector{Vector{Float64}}) || isa(action, Vector{Vector{Int64}})
        
        # Check if the size of action is the same as env.index_u
        @assert size(action, 1) == n_steps "The size of first dimension of the action need to be equal to n_steps."

        # Check if all the shape of each action is compatible with the size of index_u for all the timestep
        @assert all([size(a_t, 1) for a_t in action] .== size(env.index_u, 1)) "The size of all the second dimension of the action need to be equal to the size of env.index_u."

        # Update control index of init state with choosen action
        action = external_control([env.current_t + (i-1)*env.fixed_dt for i in 1:n_steps], vcat(action...); index_u = env.index_u[1])

    end

    # Define ODEProblem
    ode_problem = ODEProblem(env.ode_fct!, env.state, (env.current_t,env.current_t + env.fixed_dt*n_steps), env.params_vec)

    # Compute the new states
    sim_results = solve(ode_problem, saveat=[env.current_t + i*env.fixed_dt for i in 1:n_steps], alg_hints=[:stiff], callback=action)

    # Update the time
    env.current_t += env.fixed_dt*n_steps

    # Update the state
    env.state = sim_results.u[end]

    return sim_results

end