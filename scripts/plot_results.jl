using DrWatson
@quickactivate "SEM-WTTP-nh4-concentration"

using DataFrames, Plots

include(srcdir("env.jl"))
include(srcdir("asm1.jl"))
include(srcdir("fit.jl"))

df = collect_results(datadir("simulations"))
N = size(df)[1]
dt_model = 5

#############################################################################################
##################################### RMSE PLOT #############################################
#############################################################################################

rmse(H, true_val, pred; interval=H) = mean(sqrt.((true_val[Int((max(1, H-interval)/(dt_model/60))):Int(H/(dt_model/60))] - pred[Int(max(1, H-interval)/(dt_model/60)):Int(H/(dt_model/60))]).^2))

mean_em = [df.mean_em[i] for i in 1:N]
mean_em_eks = [df.mean_em_eks[i] for i in 1:N]
x_true_down = [df[!, "x_true_down"][i] for i in 1:N] 
x_pred_pem = [df[!, "x_pred_pem"][i] for i in 1:N]

rmse_tab_em = hcat([[rmse(j, x_true_down[i], mean_em[i]) for i in 1:N] for j in 1:24]...)'
q_rmse_low_em = [quantile(rmse_tab_em[i, :], 0.025) for i in 1:24]
q_rmse_high_em = [quantile(rmse_tab_em[i, :], 0.975) for i in 1:24]
mean_rmse_em = [mean(rmse_tab_em[i, :]) for i in 1:24]

rmse_tab_em_eks = hcat([[rmse(j, x_true_down[i], mean_em_eks[i]) for i in 1:N] for j in 1:24]...)'
q_rmse_low_em_eks = [quantile(rmse_tab_em_eks[i, :], 0.025) for i in 1:24]
q_rmse_high_em_eks = [quantile(rmse_tab_em_eks[i, :], 0.975) for i in 1:24]
mean_rmse_em_eks = [mean(rmse_tab_em_eks[i, :]) for i in 1:24]

rmse_tab_pem = hcat([[rmse(j, x_true_down[i], x_pred_pem[i]) for i in 1:N] for j in 1:24]...)'
q_rmse_low_pem = [quantile(rmse_tab_pem[i, :], 0.025) for i in 1:24]
q_rmse_high_pem = [quantile(rmse_tab_pem[i, :], 0.975) for i in 1:24]
mean_rmse_pem = [mean(rmse_tab_pem[i, :]) for i in 1:24]

Plots.backend(:gr)

using LaTeXStrings
using Measures
plot_font = "Computer Modern"
default(fontfamily=plot_font,
        linewidth=2, framestyle=:box, label=nothing, grid=false)

# plot(rmse_tab_pem)
plot(q_rmse_low_em, fillrange = q_rmse_high_em, alpha=0.9, label="CI 95% SEM", linestyle=:dashdot, color=:lightblue)
plot!(mean_rmse_em, label="Mean SEM", linestyle=:dashdot, color=:blue)
plot!(q_rmse_low_em_eks, fillrange = q_rmse_high_em_eks, alpha=0.9, label="CI 95% EM-EKS", linestyle=:dashdot, color=:red)
plot!(mean_rmse_em_eks, label="Mean EM-EKS", linestyle=:dashdot, color=:red)
plot!(q_rmse_low_pem, fillrange = q_rmse_high_pem, alpha=0.4, label="CI 95% PEM", color=:orange)
plot!(mean_rmse_pem, label="Mean PEM", color=:orange)
fig = plot!(xlabel="Hours of prediction", ylabel="RMSE")
safesave(plotsdir("comparison_sem_pem.png"), fig)

#############################################################################################
##################################### IC STATS  #############################################
#############################################################################################

ic_function(H, true_val, q_inf, q_sup) = mean(q_inf[1:Int(H/(dt_model/60))] .< true_val[1:Int(H/(dt_model/60))] .< q_sup[1:Int(H/(dt_model/60))])

q_low_em = [df[!, "q_low_em"][i] for i in 1:N]  
q_high_em = [df[!, "q_high_em"][i] for i in 1:N] 
x_true_down = [df[!, "x_true_down"][i] for i in 1:N]  
IC_res = [mean(q_low_em[i] .< x_true_down[i] .< q_high_em[i]) for i in 1:N] 
NW_res = [mean(q_high_em[i] - q_low_em[i]) for i in 1:N] 

q_low_em_eks = [df[!, "q_low_em_eks"][i] for i in 1:N]  
q_high_em_eks = [df[!, "q_high_em_eks"][i] for i in 1:N] 
IC_res_eks = [mean(q_low_em_eks[i] .< x_true_down[i] .< q_high_em_eks[i]) for i in 1:N] 
NW_res_eks = [mean(q_high_em_eks[i] - q_low_em_eks[i]) for i in 1:N] 

println("Global probability coverage | PFBS = ", mean(IC_res), " | EKS = ", mean(IC_res_eks))
println("Global width interval | PFBS = ", mean(NW_res), " | EKS = ", mean(NW_res_eks))


ic_tab = hcat([[ic_function(H, x_true_down[i], q_low_em[i], q_high_em[i]) for i in 1:N] for H in 1:24]...)'
q_ic_low = [quantile(ic_tab[i, :], 0.025) for i in 1:24]
q_ic_high = [quantile(ic_tab[i, :], 0.975) for i in 1:24]
mean_ic = [mean(ic_tab[i, :]) for i in 1:24]

ic_tab_eks = hcat([[ic_function(H, x_true_down[i], q_low_em_eks[i], q_high_em_eks[i]) for i in 1:N] for H in 1:24]...)'
q_ic_low_eks = [quantile(ic_tab_eks[i, :], 0.025) for i in 1:24]
q_ic_high_eks = [quantile(ic_tab_eks[i, :], 0.975) for i in 1:24]
mean_ic_eks = [mean(ic_tab_eks[i, :]) for i in 1:24]

plot(q_ic_low, fillrange = q_ic_high, alpha=0.3, color=:lightblue, label="CI 95% SEM")
plot!(mean_ic, color=:blue, label="SEM")
plot!(q_ic_low_eks, fillrange = q_ic_high_eks, alpha=0.3, color=:red, label="CI 95% EM-EKS")
fig = plot!(mean_ic_eks, color=:red, label="EM-EKS")
safesave(plotsdir("evolution_confidence_interval_horizon.png"), fig)

#############################################################################################
##################################### PARAMS STATS  #########################################
#############################################################################################

params_em = [df[!, "optim_params_pfbs_em"][i] for i in 1:N]
params_em_eks = [df[!, "optim_params_eks_em"][i] for i in 1:N]
params_pem = [df[!, "optim_params_pem.u"][i] for i in 1:N]

mean_params_em = mean(hcat([(hcat(params_em...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em...)'[:, 3:4]))), (hcat(params_em...)'[:, 5:5])]...), dims=1)'
var_params_em = sqrt.(var(hcat([(hcat(params_em...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em...)'[:, 3:4]))), (hcat(params_em...)'[:, 5:5])]...), dims=1)')

mean_params_em_eks = mean(hcat([(hcat(params_em_eks...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em_eks...)'[:, 3:4]))), (hcat(params_em_eks...)'[:, 5:5])]...), dims=1)'
var_params_em_eks = sqrt.(var(hcat([(hcat(params_em_eks...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em_eks...)'[:, 3:4]))), (hcat(params_em_eks...)'[:, 5:5])]...), dims=1)')

mean_params_em = mean(hcat([(hcat(params_pem...)'[:, 1:2]), (hcat(params_pem...)'[:, 3:3])]...), dims=1)'
var_pem = sqrt.(var(hcat([(hcat(params_pem...)'[:, 1:2]), (hcat(params_pem...)'[:, 3:3])]...), dims=1)')