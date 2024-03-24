include("time_series.jl")
include("models.jl")
include("systems/base.jl")
include("systems/non_linear_gaussian_state_space.jl")
include("filters/base.jl")
include("smoothers/base.jl")

abstract type FilterOutput end


function filter(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter::AbstractFilter=default_filter(model), parameters=model.parameters)

    filter_output = get_filter_output(filter, model, y_t)

    return filter!(filter_output, model.system, filter, y_t, exogenous_variables, control_variables, parameters)

end


function loglike(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter::AbstractFilter=default_filter(model), parameters=model.parameters)

    return filter!(model.system, filter, y_t, exogenous_variables, control_variables, parameters)

end


abstract type SmootherOutput end


function smoother(model::ForecastingModel, y_t, exogenous_variables, control_variables; filter_method::AbstractFilter=default_filter(model), smoother_method::AbstractSmoother=default_smoother(model), parameters=model.parameters)

    # Apply filtering
    filter_output = filter(model, y_t, exogenous_variables, control_variables; filter=filter_method, parameters=parameters)
    
    # Smoothing step
    return smoother(model, y_t, exogenous_variables, control_variables, filter_output; smoother_method=smoother_method, parameters=parameters)

end


function smoother(model::ForecastingModel, y_t, exogenous_variables, control_variables, filter_output::FilterOutput; smoother_method::AbstractSmoother=default_smoother(model), parameters=model.parameters)

    smoother_output = get_smoother_output(smoother_method, model, y_t)

    return smoother!(smoother_output, filter_output, model.system, smoother_method, y_t, exogenous_variables, control_variables, parameters)

end


include("filters/particle_filter.jl")
include("filters/extended_kalman_filter.jl")
include("smoothers/backward_smoothing.jl")
include("smoothers/extended_kalman_smoother.jl")