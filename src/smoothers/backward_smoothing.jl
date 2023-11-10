function backward_smoothing(y_t, exogenous_variables, filter_output, model, parameters; n_smoothing=30)

    n_obs = size(y_t, 1)
    n_X = model.system.n_X

    # Create output structure
    t_index = [model.current_state.t + (model.system.dt)*(t-1) for t in 1:(n_obs+1)]
    smoothed_particles_swarm = TimeSeries{ParticleSwarmState}(n_obs+1, 1, t_index; n_particles=n_smoothing)

    # Get output filter
    sampling_weight = filter_output.sampling_weights
    predicted_particles_swarm = filter_output.predicted_particles_swarm
    predicted_particles_swarm_mean = filter_output.predicted_particles_swarm_mean
    n_filtering = size(sampling_weight, 2)

    Xs = zeros(Float64, n_obs+1, n_X, n_smoothing)
    
    ind_smoothing = sample_discrete((1/n_filtering).*ones(n_filtering), n_smoothing)

    Xs[end, :, :] .= predicted_particles_swarm[end].particles_state[:, ind_smoothing']
    smoothed_particles_swarm[end].particles_state = Xs[end, :, :]

    @inbounds for t in (n_obs):-1:1

        # println(t)

        σ = pinv(Matrix(model.system.R_t(exogenous_variables[t, :], parameters)))

        v = Xs[t+1, :, :, [CartesianIndex()]] .- predicted_particles_swarm_mean[t+1, :, [CartesianIndex()], :]

        smoothing_weights = zeros(n_smoothing, n_filtering)

        @inbounds for i in 1:n_X
            @inbounds for j in 1:n_X
                smoothing_weights += exp.((-1/2)*v[i, :, :]*σ[i, j].*v[j, :, :])
            end
        end

        smoothing_weights = smoothing_weights.*sampling_weight[t+1, [CartesianIndex()], :]
        smoothing_weights ./= sum(smoothing_weights, dims=2) 

        ind_smoothing = sample_discrete(smoothing_weights', 1, n_exp=n_smoothing)

        Xs[t, :, :] .= predicted_particles_swarm[t].particles_state[:, ind_smoothing]

        smoothed_particles_swarm[t].particles_state = Xs[t, :, :]

    end

    return smoothed_particles_swarm

end



