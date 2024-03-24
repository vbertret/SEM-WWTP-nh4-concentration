using DrWatson
@quickactivate "SEM-WTTP-nh4-concentration"

include(srcdir("env.jl"))
include(srcdir("asm1.jl"))
include(srcdir("fit.jl"))

using Plots
using SparseArrays
using Optim, PDMats, OptimizationNLopt
using OptimizationOptimisers
using IterTools: ncycle
using LaTeXStrings
using Measures
using Flux

nb_exp = 1
X_in_args = 6.8924
try
    global nb_exp = parse(Int, ARGS[1])
    global X_in_args = parse(Float64, ARGS[2])
catch e
    println(e)
end

params_exp = @strdict nb_exp X_in_args


println("################ RUN N°$nb_exp EXPERIENCE FOR X_in=$X_in_args ################")

#######################################################
################### Generate data #####################
#######################################################

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

H_t(x, exogenous, params) = x

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
########################### Define systems for PEM ############################
###############################################################################

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


##################################################################
########################### Fit models ###########################
##################################################################

######## EM-PFBS ########

# Define model for EM-PFBS
parameters = [1333.0, 200.0, -2.30, -2.30, 1.0]
model = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, init_state, parameters)

# Optimize with EM using approximate PFBS smoother
lb = [1e-2, 1e-2, -Inf, -Inf, 1e-2]
ub = [Inf, Inf, Inf, Inf, Inf]
@timed optim_params_pfbs_em, results_pfbs = SEM(model, y_train, E_train, U_train; lb=lb, ub=ub, n_filtering = 300, n_smoothing = 300, maxiters_em=50, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
model.parameters = optim_params_pfbs_em

# Define model for EM-EKS
parameters = [1333.0, 200.0, -2.30, -2.30, 1.0]
model2 = ForecastingModel{GaussianNonLinearStateSpaceSystem}(gnlss, init_state, parameters)

# Optimize with EM using approximate PFBS smoother
lb = [1e-2, 1e-2, -Inf, -Inf, 1e-2]
ub = [Inf, Inf, Inf, Inf, Inf]
@timed optim_params_eks_em, results_eks = EM_EKS(model2, y_train, E_train, U_train; lb=lb, ub=ub, maxiters_em=50, optim_method=Opt(:LD_LBFGS, 5), maxiters=100);
model2.parameters = optim_params_eks_em

######## PEM ########
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
################# Predictions#################
##############################################

# Prediction with EM-PFBS
n_filtering = 300; n_smoothing = 300
filter_output, _, _ = filter(model, y_total, E_total, u_total, filter=ParticleFilter(model, n_particles = n_filtering));
smoother_output = backward_smoothing(y_total, E_total, filter_output, model, model.parameters; n_smoothing=n_smoothing);
x_pred_em = filter_output.predicted_particles_swarm[Int((1440/dt_model)*(T_training)+1):end];
mean_em = hcat([mean(x_pred_em[i].particles_state, dims=2) for i in 1:x_pred_em.n_t]...)';
q_low_em = hcat([[quantile(x_pred_em[i].particles_state[j, :], 0.025) for j in 1:x_pred_em.n_state] for i in 1:x_pred_em.n_t]...)';
q_high_em = hcat([[quantile(x_pred_em[i].particles_state[j, :], 0.975) for j in 1:x_pred_em.n_state] for i in 1:x_pred_em.n_t]...)';

# Prediction with EM-EKS
filter_output_eks = filter(model2, y_total, E_total, u_total, filter=ExtendedKalmanFilter(model2));
smoother_output = smoother(model2, y_total, E_total, u_total, filter_output_eks, smoother_method=ExtendedKalmanSmoother(model2));
x_pred_em_eks = filter_output_eks.predicted_state[Int((1440/dt_model)*(T_training)+1):end];
mean_em_eks = hcat([x_pred_em_eks[i].μ_t for i in 1:x_pred_em_eks.n_t]...)';
q_low_em_eks = hcat([x_pred_em_eks[i].μ_t - 1.96*sqrt.(x_pred_em_eks[i].σ_t)  for i in 1:x_pred_em_eks.n_t]...)';
q_high_em_eks = hcat([x_pred_em_eks[i].μ_t + 1.96*sqrt.(x_pred_em_eks[i].σ_t)  for i in 1:x_pred_em_eks.n_t]...)';


# Prediction with PEM
x_pred_pem = zeros(Int(1440/dt_model+1));
x_pred_pem[1] = y_train[end, 1];
for i in 1:Int(1440/dt_model)
    x_pred_pem[i+1] = model_nh4(optim_params_pem, x_pred_pem[i], U_test[i, :], E_test[i, :])[1, 1]
end

# Get true state
x_true = x_total[10, Int((1440)*(T_training)):end];
x_true_down = x_true[1:dt_model:end];
index_true = [T_steady_state + T_training + (1/1440)*t for t in 0:Int((T_testing)*1440)];
index_pred = [x_pred_em.state[i].t for i in 1:x_pred_em.n_t];

# Add params to dicts
full_dict = copy(params_exp)
full_dict = convert(Dict{String, Any}, full_dict)
full_dict["x_pred_em"] = x_pred_em
full_dict["mean_em"] = mean_em
full_dict["q_low_em"] = q_low_em
full_dict["q_high_em"] = q_high_em
full_dict["x_pred_em_eks"] = x_pred_em_eks
full_dict["mean_em_eks"] = mean_em_eks
full_dict["q_low_em_eks"] = q_low_em_eks
full_dict["q_high_em_eks"] = q_high_em_eks
full_dict["x_pred_pem"] = x_pred_pem
full_dict["x_true"] = x_true
full_dict["x_true_down"] = x_true_down
full_dict["index_true"] = index_true
full_dict["index_pred"] = index_pred
full_dict["optim_params_pfbs_em"] = optim_params_pfbs_em
full_dict["optim_params_eks_em"] = optim_params_eks_em
full_dict["optim_params_pem.u"] = optim_params_pem.u

# Save results
wsave(datadir("simulations", savename(params_exp, "jld2")), full_dict)