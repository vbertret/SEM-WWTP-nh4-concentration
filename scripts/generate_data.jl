using DrWatson
@quickactivate "SEM-WTTP-nh4-concentration"

include(srcdir("env.jl"))
include(srcdir("asm1.jl"))

using DifferentialEquations, Distributions
using LinearAlgebra:I
using Statistics:mean

# User-defined parameters
T_steady_state = 20 #(in days)
T_training = 5.0 #(in days)
T_testing = 1.0 #(in days)
σ_ϵ = 0.2
# WARNING : dt_obs must be a multiple of dt_model
dt_obs = 5 #(in minutes)
dt_model = 5 #(in minutes)

# Fixed parameters
nb_var = 14
index_obs_var = [10]
nb_obs_var = size(index_obs_var, 1)

####################################################################################################################
########################################## REFERENCE DATA FOR OPTIMIZATION #########################################
####################################################################################################################

# Set up environment
sim = ODECore()
if @isdefined(X_in_args)
    sim.params_vec.X_in[10] = X_in_args
end

# Let the system evolve during T_steady_state days in order to be stable and get new X_init
res_steady = multi_step!(sim, redox_control(), Day(T_steady_state))
X_init_real = sim.state

# Let the system evolve during T_training + T_testing days to get training and testing data
sol_real  = multi_step!(sim, redox_control(), Hour((T_training + T_testing)*24))

# Define H
H = zeros(nb_var, nb_var)
H[index_obs_var, index_obs_var] = Matrix(I, nb_obs_var, nb_obs_var)
H = H[index_obs_var, :]

# Get training and test set for the state of the system
x_train = hcat(sol_real.u...)[:, 1:Int(T_training*1440)]
x_test = hcat(sol_real.u...)[:, (Int(T_training*1440)+1):end]

# Get training and test set for the observation of the system
y_train = max.((H*x_train + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_train, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_train, 2)], (1,size(x_train, 2))) for i in 1:nb_obs_var]...))', 0)
y_test = max.((H*x_test + rand(Normal(0, σ_ϵ), (nb_obs_var, size(x_test, 2))) + vcat([reshape([(i-1)%Int(dt_obs) == 0 ? 0 : NaN for i in 1:size(x_test, 2)], (1,size(x_test, 2)))  for i in 1:nb_obs_var]...))', 0)

# Get control variables of the system
U_train = transpose(hcat(getindex.(sol_real.u, 14)...))[1:Int(T_training*1440), :]
U_test = transpose(hcat(getindex.(sol_real.u, 14)...))[(Int(T_training*1440)+1):end, :]

# Adapt y_train and U_train according to dt_assimilation
y_train = y_train[1:dt_model:end, 1:end]
y_test = y_test[1:dt_model:end, 1:end]

U_train = mean(reshape(U_train, dt_model, :), dims=1)'
U_test = mean(reshape(U_test, dt_model, :), dims=1)'