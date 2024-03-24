# export GaussianNonLinearStateSpaceSystem

mutable struct GaussianNonLinearStateSpaceSystem <: StateSpaceSystem

    # General components of gaussian non linear state space systems 
    M_t::Function
    H_t::Function
    R_t::Function
    Q_t::Function

    #Size of observation and latent space
    n_X::Int64
    n_Y::Int64

    # Time between two states
    dt::Float64

    # Derivatives if defined
    dM_t
    dH_t

    function GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt)

        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, Nothing, Nothing)
    end

    function GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, dM_t, dH_t)

        return new(M_t, H_t, R_t, Q_t, n_X, n_Y, dt, dM_t, dH_t)
    end

end


function transition(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, control_variables, parameters) 

    return ssm.M_t(current_x, exogenous_variables, control_variables, parameters)

end


function observation(ssm::GaussianNonLinearStateSpaceSystem, current_x, exogenous_variables, parameters) 

    return ssm.H_t(current_x, exogenous_variables, parameters)

end


function default_filter(model::ForecastingModel{GaussianNonLinearStateSpaceSystem})

    return ParticleFilter(model.current_state, model.system.n_X, model.system.n_Y, 50)

end

