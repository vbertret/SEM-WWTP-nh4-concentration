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

X_in_args = 7.8924
include("generate_data.jl")

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

    A = sparse([1 - Q_in/V*(dt_model/1440);;])
    B = sparse([- β*(dt_model/1440);;])

    return A*x + B*u.*(x./(x .+ K)) .+ sparse([(X_in*Q_in)/V*(dt_model/1440)])

end

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
gnlss = GaussianNonLinearStateSpaceSystem(M_t, H_t, R_t, Q_t, n_X, n_Y, dt_model/(1440))

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

println("--------------------------------------------------------------------")
println("----------------------- OPTIMIZATION RESULTS -----------------------")
println("--------------------------------------------------------------------\n")

println("   V    | Estimated PFBS = ", round(optim_params_pfbs_em[1], digits=3))
println("   β    | Estimated PFBS = ", round(optim_params_pfbs_em[2], digits=3))
println("   K    | Estimated PFBS = ", round(optim_params_pfbs_em[5], digits=3))
println("σ_model | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[3])), digits=3))
println("σ_obs   | Estimated PFBS = ", round(sqrt(exp(optim_params_pfbs_em[4])), digits=3))

println("\n--------------------------------------------------------------------")
println("--------------------------------------------------------------------")
println("--------------------------------------------------------------------")

#######################################################
################# Prediction with PEM #################
#######################################################

index_x = [T_steady_state + (1/1440)*t for t in 1:Int((T_training+T_testing)*1440)]
index_y = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training)*1440/dt_model)]
index_u = [T_steady_state + (1/1440)*t*dt_model for t in 1:Int((T_training + T_testing)*1440/dt_model)]

Q_in_t = [Q_in(T_steady_state + t*dt_model/(24*60)) for t in 1:(size(y_train, 1)+size(y_test, 1))]
X_in_t = [X_in for t in 1:(size(y_train, 1)+size(y_test, 1))]
E_graph = hcat([Q_in_t, X_in_t]...)

y_graph = vcat(y_train, similar(y_test).*NaN)
u_graph = vcat(U_train, U_test)
x_graph = hcat(x_train, x_test)

n_smoothing = 300
filter_output, _, _ = filter(model, y_graph, E_graph, u_graph, filter=ParticleFilter(model, n_particles = 300))
smoother_output_bs1 = backward_smoothing(y_graph, E_graph, filter_output, model, model.parameters; n_smoothing=n_smoothing)

plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)
scalefontsizes(0.9)

plot(index_u[Int((1440/dt_model)*(T_training-1)):end], u_graph[Int((1440/dt_model)*(T_training-1)):end, :], alpha=0.5, color=:grey, label=L"u(t)")
plot!(filter_output.predicted_particles_swarm[Int((1440/dt_model)*(T_training-1)+1):end], label= ["Model"])
plot!(size=(1000, 320), xlabel="Time (in days)", ylabel=L"S_{NH}"*" (mg/L)", margin=6mm)
plot!(index_x[Int((1440)*(T_training-1)):end], x_graph[10, Int((1440)*(T_training-1)):end], label="True NH4", linestyle=:dashdot)
scatter!(index_y[Int((1440/dt_model)*(T_training-1)+1):end], y_train[Int((1440/dt_model)*(T_training-1)+1):end], label="Observations", markersize=1.0)
plot!(legend=:topright)
vline!([25.0], color=:black)
annotate!(24.02,7.5,text("Past",plot_font,15))
annotate!(25.1,7.5,text("Future",plot_font,15))
plot!(legend=:bottomleft)
savefig(plotsdir("model_prediction.png"))