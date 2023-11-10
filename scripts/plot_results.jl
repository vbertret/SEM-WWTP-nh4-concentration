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
x_true_down = [df[!, "x_true_down"][i] for i in 1:N] 
x_pred_pem = [df[!, "x_pred_pem"][i] for i in 1:N]

rmse_tab_em = hcat([[rmse(j, x_true_down[i], mean_em[i]) for i in 1:N] for j in 1:24]...)'
q_rmse_low_em = [quantile(rmse_tab_em[i, :], 0.025) for i in 1:24]
q_rmse_high_em = [quantile(rmse_tab_em[i, :], 0.975) for i in 1:24]
mean_rmse_em = [mean(rmse_tab_em[i, :]) for i in 1:24]

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

plot(rmse_tab_pem)
plot(q_rmse_low_em, fillrange = q_rmse_high_em, alpha=0.9, label="IC 95% SEM", linestyle=:dashdot, color=:lightblue)
plot!(mean_rmse_em, label="Mean SEM", linestyle=:dashdot, color=:blue)
plot!(q_rmse_low_pem, fillrange = q_rmse_high_pem, alpha=0.4, label="IC 95% PEM", color=:orange)
plot!(mean_rmse_pem, label="Mean PEM", color=:orange)
plot!(xlabel="Hours of prediction", ylabel="RMSE")
savefig(plotsdir("comparison_sem_pem.png"))

#############################################################################################
##################################### IC STATS  #############################################
#############################################################################################

ic_function(H, true_val, q_inf, q_sup) = mean(q_inf[1:Int(H/(dt_model/60))] .< true_val[1:Int(H/(dt_model/60))] .< q_sup[1:Int(H/(dt_model/60))])

q_low_em = [df[!, "q_low_em"][i] for i in 1:N]  
q_high_em = [df[!, "q_high_em"][i] for i in 1:N] 
x_true_down = [df[!, "x_true_down"][i] for i in 1:N]  
IC_res = [mean(q_low_em[i] .< x_true_down[i] .< q_high_em[i]) for i in 1:10] 
NW_res = [mean(q_high_em[i] - q_low_em[i]) for i in 1:10] 

println("Global probability coverage : ", mean(IC_res))
println("Global width interval : ", mean(NW_res))


ic_tab = hcat([[ic_function(H, x_true_down[i], q_low_em[i], q_high_em[i]) for i in 1:10] for H in 1:24]...)'
q_ic_low = [quantile(ic_tab[i, :], 0.025) for i in 1:24]
q_ic_high = [quantile(ic_tab[i, :], 0.975) for i in 1:24]
mean_ic = [mean(ic_tab[i, :]) for i in 1:24]
plot(q_ic_low, fillrange = q_ic_high, alpha=0.3)
plot!(mean_ic)
savefig(plotsdir("evolution_confidence_interval_horizon.png"))

#############################################################################################
##################################### PARAMS STATS  #########################################
#############################################################################################

params_em = [df[!, "optim_params_pfbs_em"][i] for i in 1:N]
params_pem = [df[!, "optim_params_pem.u"][i] for i in 1:N]

mean_params_em = mean(hcat([(hcat(params_em...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em...)'[:, 3:4]))), (hcat(params_em...)'[:, 5:5])]...), dims=1)'
var_params_em = sqrt.(var(hcat([(hcat(params_em...)'[:, 1:2]), (sqrt.(exp.(hcat(params_em...)'[:, 3:4]))), (hcat(params_em...)'[:, 5:5])]...), dims=1)')

var_pem = sqrt.(var(hcat([(hcat(params_pem...)'[:, 1:2]), (hcat(params_pem...)'[:, 3:3])]...), dims=1)')
mean_params_em = mean(hcat([(hcat(params_pem...)'[:, 1:2]), (hcat(params_pem...)'[:, 3:3])]...), dims=1)'