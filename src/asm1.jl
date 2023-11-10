using DifferentialEquations:DiscreteCallback, CallbackSet, PresetTimeCallback
using Interpolations:interpolate, Gridded, Linear
using DelimitedFiles:readdlm
using StaticArrays

"""
Returns the ASM1 Function.
"""
function asm1!(dX, X, p, t)

    # If Q_in is a function of the time t, then evaluate it
    if typeof(p[23]) <: Function
         Q_in = p[23](t)
    else
         Q_in = p[23]
    end

    Y_A = p[15] ; Y_H = p[16] ; f_P = p[17] ; i_XB = p[18] ; i_XP = p[19]
    R = @SMatrix [0              0                                 0                       0                   0                 0     0     0 ;
                  -1/Y_H         -1/Y_H                            0                       0                   0                 0     1     0 ;
                  0              0                                 0                       0                   0                 0     0     0 ;
                  0              0                                 0                       1-f_P               1-f_P             0    -1     0 ;
                  1              1                                 0                       -1                  0                 0     0     0 ;
                  0              0                                 1                       0                   -1                0     0     0 ;
                  0              0                                 0                       f_P                 f_P               0     0     0 ;
                  -(1-Y_H)/Y_H   0                                 -4.57/Y_A+1             0                   0                 0     0     0 ;
                  0              -(1-Y_H)/(2.86*Y_H)               1.0/Y_A                 0                   0                 0     0     0 ;
                  -i_XB          -i_XB                             -(i_XB+(1.0/Y_A))       0                   0                 1     0     0 ;
                  0              0                                 0                       0                   0                 -1    0     1 ;
                  0              0                                 0                       (i_XB-f_P*i_XP)     (i_XB-f_P*i_XP)   0     0     -1;
                  -i_XB/14       (1-Y_H)/(14*2.86*Y_H)-(i_XB/14)   -(i_XB/14)+1/(7*Y_A)    0                   0                 1/14  0     0 ]

    ### Calculate process rates ###
    K_OH = p[3]
    saturation_1 = p[1]*(X[2]/(p[2]+X[2]))
    saturation_2 = (X[8]/(K_OH+X[8]))
    saturation_3 = (K_OH/(K_OH+X[8]))
    saturation_4 = (X[9]/(p[4]+X[9]))
    penultimate_term = p[8]*((X[4]/X[5])/(p[9]+(X[4]/X[5])))*(saturation_2+p[7]*saturation_3*saturation_4)*X[5]
    process_rates = @SArray [saturation_1*saturation_2*X[5], # Aerobic growth of heterotrophs
                             saturation_1*saturation_3*saturation_4*p[6]*X[5], # Anoxic growth of heterotrophs
                             p[10]*(X[10]/(p[11]+X[10]))*(X[8]/(p[13]+X[8]))*X[6], # Aerobic growth of autotrophs
                             p[5]*X[5], # "Decay" of heterotrophs
                             p[12]*X[6], # "Decay" of autotrophs
                             p[14]*X[11]*X[5], # Ammonification of soluble organic nitrogen
                             penultimate_term, # "Hydrolysis" of entrapped organics
                             penultimate_term*X[12]/X[4]] # "Hydrolysis" of entrapped organics nitrogen

    ### Calculate differential equations ###
    # General expression
    dX[1:13] = (Q_in/p[20]) * (p[24] - X[1:13]) + R * process_rates
    dX[14] = 0.0

    # Control input for oxygen
    dX[8] += X[14] * p[22] * (p[21] - X[8])

end

"""
Get the default parameters of the ASM1 model.
"""
function get_default_parameters_asm1(; T = 15, influent_file_path = nothing)

    ### Define the function that adapts the parameters according to the temperature ###
    function T_var(T, ρ, a)
         return ρ * exp((log2(ρ/a)/5)*(T-15))
    end  

    ### Kinetic parameters ###
    μ_H = T_var(T, 4.0, 3) ; K_S = 10.0 ; K_OH = 0.2 ; K_NO = 0.5 ; b_H = T_var(T, 0.3, 0.2) ; η_g = 0.8 ; η_h = 0.8 ; k_h = T_var(T, 3.0, 2.5) ; K_X = 0.1 ; μ_A = T_var(T, 0.5, 0.3) ; K_NH = 1.0 ; b_A = T_var(T, 0.05, 0.03) ; K_OA =  0.4 ; k_a = T_var(T, 0.05, 0.04)
    kinetic_parameters = (μ_H=μ_H, K_S=K_S, K_OH=K_OH, K_NO=K_NO, b_H=b_H, η_g=η_g, η_h=η_h, k_h=k_h, K_X=K_X, μ_A=μ_A, K_NH=K_NH, b_A=b_A, K_OA=K_OA, k_a=k_a)

    ### Stoichiometric parameters ###
    Y_A = 0.24 ; Y_H = 0.67 ; f_P = 0.08 ; i_XB = 0.08 ; i_XP = 0.06
    stoichiometric_parameters = (Y_A=Y_A, Y_H=Y_H, f_P=f_P, i_XB=i_XB, i_XP=i_XP)

    ### Other parameters ###
    V = 1333.0 # volume
    SO_sat = (8/10.50237016)*6791.5*(56.12*exp(-66.7354 + 87.4755/((T+273.15)/100.0) + 24.4526*log((T+273.15)/100.0)))
    KLa = 200*(1.024^(T-15)) # KLa
    other_params = (V=V, SO_sat=SO_sat, KLa=KLa)

    if false#influent_file_path ≠ nothing
         inflow_generator = readdlm(influent_file_path)
         list_order = [7, 2, 5, 4, 3, 0.0, 0.0, 0.0, 0.0, 6, 8, 9, 7.0]
         constant_value = [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1]
         function X_in(t) 
              return [[(constant_value[i] ==  0) ? interpolate((inflow_generator[: ,1], ), inflow_generator[: ,Int(list_order[i])], Gridded(Linear())) :  interpolate((inflow_generator[: ,1], ), list_order[i] .* ones(size(inflow_generator, 1)), Gridded(Linear())) for i in 1:13][i](abs(t) .% maximum(inflow_generator[: ,1])) for i in 1:13]
         end
    else 
         # Xin_Si = 28.0643; Xin_Ss = 3.0503; Xin_Xi = 1532.3; Xin_Xs = 63.0433; Xin_Xbh = 2245.1; Xin_Xba = 166.6699; Xin_Xp = 964.8992; Xin_So = 0.0093; Xin_Sno = 3.9350; Xin_Snh = 6.8924; Xin_Snd = 0.9580; Xin_Xnd = 3.8453; Xin_Salk = 5.4213
         X_in =  [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 8.8924, 0.9580, 3.8453, 5.4213]
    end
    
    if influent_file_path ≠ nothing
         inflow_generator = readdlm(influent_file_path)
         itp = interpolate((inflow_generator[: ,1], ), inflow_generator[: ,10], Gridded(Linear()))
         T_max = maximum(inflow_generator[: ,1])
         function Q_in(t) 
              return itp(abs(t) % T_max)
         end
    else
         Q_in = 18061.0
    end
    exogenous_params = (Q_in=Q_in, X_in=X_in)
    # exogenous_params = (Q_in=Q_in, Xin_Si=Xin_Si, Xin_Ss=Xin_Ss, Xin_Xi=Xin_Xi, Xin_Xs=Xin_Xs, Xin_Xbh=Xin_Xbh, Xin_Xba=Xin_Xba, Xin_Xp=Xin_Xp, Xin_So=Xin_So, Xin_Sno=Xin_Sno, Xin_Snh=Xin_Snh, Xin_Snd=Xin_Snd, Xin_Xnd=Xin_Xnd, Xin_Salk=Xin_Salk)
    

    # Merge parameters
    p = merge(kinetic_parameters, stoichiometric_parameters, other_params, exogenous_params)

    ### X_init ###
    X_init =  [28.0643, 3.0503, 1532.3, 63.0433, 2245.1, 166.6699, 964.8992, 0.0093, 3.9350, 6.8924, 0.9580, 3.8453, 5.4213, 1.0]

    return (p, X_init)

end


"""
Returns a CallbackSet that simulates the redox control of a bioreactor.
"""
function redox_control(;index_no3 = 9, index_nh4 = 10, index_u = -1)

    # If the S_{NO} is near 0, then the aeration is turned on
    S_NO_low(u,t,integrator) = u[index_no3] < 1.0
    aeration_on!(integrator) = index_u == -1 ? integrator.u[end] = 1 : integrator.u[index_u] = 1
    redox_c1 = DiscreteCallback(S_NO_low, aeration_on!; save_positions=(false, false))

    # If the S_{NH} is near 0, then the aeration is turned off
    S_NH_low(u,t,integrator) = u[index_nh4] < 0.8
    aeration_off!(integrator) = index_u == -1 ? integrator.u[end] = 0 : integrator.u[index_u] = 0
    redox_c2 = DiscreteCallback(S_NH_low, aeration_off!; save_positions=(false, false))

    redox_callback = CallbackSet(redox_c1, redox_c2)

    return redox_callback

end


"""
Returns a ContinuousCallback that simulates the external control of a bioreactor given as a time array and a control array.
"""
function external_control(array_t, array_u; index_u = -1)

    # Create a function that returns the control value at a given time
    control = Interpolations.interpolate((array_t,), vcat(array_u[2:end], array_u[end]), Gridded(Linear()))

    # Search for the change in the control value in the array_u
    event_times = [0.0]
    for i in 1:length(array_u)-1
        if abs(array_u[i] - array_u[i+1]) > 0.5
            push!(event_times, array_t[i])
        end
    end

    # Create a callback that changes the control value at the given times
    external_control_callback = PresetTimeCallback(event_times,(integrator) -> index_u == -1 ? integrator.u[end] = abs(control(integrator.t)) : integrator.u[index_u] = abs(control(integrator.t)), save_positions=(false, false))

    return external_control_callback

end