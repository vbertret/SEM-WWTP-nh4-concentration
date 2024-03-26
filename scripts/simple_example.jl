using DrWatson
@quickactivate "SEM-WTTP-nh4-concentration"

include(srcdir("env.jl"))
include(srcdir("asm1.jl"))
include(srcdir("fit.jl"))

using Plots
using SparseArrays
using Optim, OptimizationNLopt, PDMats
using OptimizationOptimisers
using IterTools: ncycle
using LaTeXStrings
using Measures
using Flux

#######################################################
################### Generate data #####################
#######################################################

X_in_args = 7.0
include(scriptsdir("generate_data.jl"))

###############################################################################
################### Define systems for state space models #####################
###############################################################################

function M_t(x, exogenous, u, params)

    # Get Q_in and V
    Q_in = exogenous[1]
    X_in = exogenous[2]
    V = params[1]
    β = params[2]
    K = params[5]

    A = [1 - Q_in/V*(dt_model/1440);;]
    B = [- β*(dt_model/1440);;]

    return A*x + B*u.*(x./(x .+ K)) .+ [(X_in*Q_in)/V*(dt_model/1440)]

end

function dM_t(x, exogenous, u, params)

    # Get Q_in and V
    Q_in = exogenous[1]
    V = params[1]
    β = params[2]
    K = params[5]

    A = sparse([1 - Q_in/V*(dt_model/1440);;])
    B = sparse([- β*(dt_model/1440);;])

    return A + B*u.*(K./((x .+ K).^2))

end

dH_t(x, exogenous, params) = I(1)*1

H_t(x, exogenous, params) = x

function R_t(exogenous, params)

    # Get params
    # η = exp(params[3])
    η = exp(params[3])


    # Define R
    # R = hcat([[η]]...)

    return PDiagMat([η])

end

function Q_t(exogenous, params)

    # Get params
    # ϵ = exp(params[4])
    ϵ = exp(params[4])

    # Define Q
    # Q = hcat([[ϵ]]...)

    return PDiagMat([ϵ])

end

# Define the system
n_X = 1
n_Y = 1
gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt_model/(1440), dM_t, dH_t)

# Define init state
init_P_0 = zeros(1, 1) .+   1 #0.001
init_state = GaussianStateStochasticProcess(T_steady_state, [x_train[10,1]], init_P_0)

###############################################################################
########################### Generate train/test data ##########################
###############################################################################

# Set exogenous predictors
Q_in = sim.params_vec.Q_in
X_in = sim.params_vec.X_in[10]

#### Generate training data ####
Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:size(y_train, 1)]
X_in_t = [X_in for t in 1:size(y_train, 1)]
E_train = hcat([Q_in_t, X_in_t]...)

#### Generate test data ####
Q_in_t = [Q_in(T_steady_state + T_training + t*dt_model/(24*60)) for t in 1:size(y_test, 1)]
X_in_t = [X_in for t in 1:size(y_test, 1)]
E_test = hcat([Q_in_t, X_in_t]...)

#### Generate total data ####
Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
E_total = hcat([Q_in_t, X_in_t]...)
y_total = vcat(y_train, similar(y_test).*NaN)
u_total = vcat(U_train, U_test)
x_total = hcat(x_train, x_test)


#################################################################
########################### Fit model ###########################
#################################################################

# Define model for EM-PFBS
parameters = [1333.0, 200.0, -2.30, -2.30, 1.0]
model = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, init_state, parameters)

# Optimize with EM using approximate PFBS smoother
lb = [1e-2, 1e-2, -Inf, -Inf, 1e-2]
ub = [Inf, Inf, Inf, Inf, Inf]
@timed optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; lb=lb, ub=ub, n_filtering = 300, n_smoothing = 300, maxiters_em=30, optim_method=Opt(:LD_LBFGS, 5), maxiters=30);
model.parameters = optim_params_pfbs_em

# Define model for EM-EKS
parameters2 = [1333.0, 200.0, -2.30, -2.30, 1.0]
model2 = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, init_state, parameters2)

# Optimize with EM using approximate EKS smoother
lb = [1e-2, 1e-2, -Inf, -Inf, 1e-2]
ub = [Inf, Inf, Inf, Inf, Inf]
@timed optim_params_eks_em, results_eks = EM_EKS(model2, y_train, E_train, U_train; lb=lb, ub=ub, maxiters_em=30, optim_method=Opt(:LD_LBFGS, 5), maxiters=30);
model2.parameters = optim_params_eks_em

# Define model for PEM
function model_nh4(params, x, u, exogenous)

    return x .+ ((exogenous[1:1, :]./params[1]).*(exogenous[2:2, :] .- x) .- params[2].*u.*(x./(x .+ params[3]))).*(dt_model/1440)

end

function loss(params, y, x, u, exogenous)
    return mean((y .- model_nh4(params, x, u, exogenous)).^2)
end

callback = function (p, l)
    println("Loss : ", l)
    return false
end

# Optimize with PEM
nb_obs = size(y_train, 1)
numEpochs = 50000
k = nb_obs - 1

y = reshape(y_train[2:end], 1, nb_obs-1) ; x = reshape(y_train[1:(end-1)], 1, nb_obs-1)
u = U_train[1:(end-1), :]' ; e = E_train[1:(end-1), :]'
train_loader = Flux.Data.DataLoader((y, x, u, e), batchsize = k)

init_p = parameters[[1, 2, 5]]
optfun = OptimizationFunction((θ, params, y, x, u, exogenous) -> loss(θ, y, x, u, exogenous), Optimization.AutoForwardDiff())
optprob = OptimizationProblem(optfun, init_p)
optim_params_pem = Optimization.solve(optprob, Optimisers.ADAM(0.1), ncycle(train_loader, numEpochs), callback = callback)

println("--------------------------------------------------------------------")
println("----------------------- OPTIMIZATION RESULTS -----------------------")
println("--------------------------------------------------------------------\n")

println("   V    | Estimated PFBS = ", round(optim_params_pfbs_em[1], digits=3), " | Estimated EKS = ", round(optim_params_eks_em[1], digits=3), " | Estimated PEM = ", round(optim_params_pem[1], digits=3))
println("   β    | Estimated PFBS = ", round(optim_params_pfbs_em[2], digits=3), " | Estimated EKS = ", round(optim_params_eks_em[2], digits=3), " | Estimated PEM = ", round(optim_params_pem[2], digits=3))
println("   K    | Estimated PFBS = ", round(optim_params_pfbs_em[5], digits=3), " | Estimated EKS = ", round(optim_params_eks_em[5], digits=3), " | Estimated PEM = ", round(optim_params_pem[3], digits=3))
println("σ_model | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[3])), digits=3), " | Estimated EKS = ", round(sqrt(exp(optim_params_eks_em[3])), digits=3))
println("σ_obs   | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[4])), digits=3), " | Estimated EKS = ", round(sqrt(exp(optim_params_eks_em[4])), digits=3))

println("\n--------------------------------------------------------------------")
println("--------------------------------------------------------------------")
println("--------------------------------------------------------------------")

##############################################
################# Prediction #################
##############################################

index_x = [T_steady_state + (1/1440)*t for t in 1:Int((T_training+T_testing)*1440)]
index_y = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training)*1440/dt_model)]
index_u = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training + T_testing)*1440/dt_model)]

Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
E_graph = hcat([Q_in_t, X_in_t]...)

y_graph = vcat(y_train, similar(y_test).*NaN)
u_graph = vcat(U_train, U_test)
x_graph = hcat(x_train, x_test)

# Prediction with EM-PF
n_smoothing = 1000
filter_output_pf, _, _ = filter(model, y_graph, E_graph, u_graph, filter=ParticleFilter(model, n_particles = 1000))
smoother_output_bs1 = backward_smoothing(y_graph, E_graph, filter_output_pf, model, model.parameters; n_smoothing=n_smoothing)

# Prediction with EM-EKF
filter_output_ekf = filter(model2, y_graph, E_graph, u_graph, filter=ExtendedKalmanFilter(model2))
smoother_output_ekf = smoother(model2, y_graph, E_graph, u_graph, filter_output_ekf, smoother_method=ExtendedKalmanSmoother(model2));

# Prediction with PEM
x_pred_pem = zeros(Int(1440/dt_model+1));
x_pred_pem[1] = y_train[end, 1];
for i in 1:Int(1440/dt_model)
    x_pred_pem[i+1] = model_nh4(optim_params_pem, x_pred_pem[i], U_test[i, :], E_test[i, :])[1, 1]
end

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)
scalefontsizes(1.5)

plot(index_u[Int((1440/dt_model)*(T_training-1)+1):end], u_graph[Int((1440/dt_model)*(T_training-1)+1):end, :], alpha=0.5, color=:grey, label=L"u(t)")
plot!(filter_output_pf.predicted_particles_swarm[Int((1440/dt_model)*(T_training-1)+1):end], label= ["PF"], linewidth=0.95, color=:deepskyblue1, linestyle=:dashdot)
plot!(filter_output_ekf.predicted_state[Int((1440/dt_model)*(T_training-1)+1):end], label= "EKF", linestyle=:dashdot, linewidth=0.95, color=:indianred3)
plot!(size=(1000, 320), xlabel="Time (in days)", ylabel=L"S_{NH}"*" (mg/L)", margin=6mm)
plot!(index_x[Int((1440)*(T_training-1)+1):end], x_graph[10, Int((1440)*(T_training-1)+1):end], label="True NH4", linestyle=:solid, linewidth=1, color=:purple)
scatter!(index_y[Int((1440/dt_model)*(T_training-1)+1):end], y_train[Int((1440/dt_model)*(T_training-1)+1):end], label="Observations", markersize=1.0)
plot!(index_u[Int((1440/dt_model)*(T_training)):end], x_pred_pem, label="Mean PEM", color=:orange, linewidth=0.75, linestyle=:dash)
plot!(legend=:topright)
vline!([T_steady_state + T_training], color=:black)
annotate!(T_steady_state + T_training - 1 + 0.02,6.5,text("Past",plot_font,15))
annotate!(T_steady_state + T_training + 0.1,6.5,text("Future",plot_font,15))
fig = plot!(legend=:bottomleft)
safesave(plotsdir("model_prediction.pdf"), fig)