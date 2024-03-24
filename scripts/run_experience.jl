nb_exp = 10
X_in_table = [6.8924, 7.8924, 8.8924, 9.8924]

for x_in_ite in X_in_table
    for num_exp in 1:nb_exp
        run(`julia --project=. scripts/estimation.jl $num_exp $x_in_ite`)
    end
end