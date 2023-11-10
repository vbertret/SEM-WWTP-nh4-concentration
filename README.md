# SEM-WTTP-nh4-concentration

Wastewater Treatment plants (WTTPs) need to curtail their energy consumption and adhere to environmental regulations due to the surge in energy prices and the imperative for environmental sustainability. Our primary goal is to minimize fossil energy consumption and decrease ammonium $(NH_4^+)$ and nitrate $(NO_3^-)$ concentrations.

Existing controllers struggle to adapt to fluctuating electricity prices and the variable conditions within WWTPs. While Model Predictive Control and Dynamic Programming offer promising control strategies, their effective deployment relies on the availability of a robust system dynamics model.

Our approach introduces a stochastic model and estimation method that combines a Monte Carlo Sequential smoothing algorithm with Stochastic Expectation Maximization. Further details are presented in the article ["Stochastic expectation maximization algorithm for the estimation of wastewater treatment plant ammonium concentration"](), authored by Victor Bertret, Valérie Monbet, and Roman Legoff Latimier submitted at the [ECC24](https://ecc24.euca-ecc.org/) conference.

The `SEM-WTTP-nh4-concentration` source code was built to generate the results presented in the article. In the next sections, the structure, installation, and usage of the project are explained.

# Organization

The project is organized as follows:

* `data`
   * `influent_files`: Influent files coming from [BSM repo](https://github.com/wwtmodels/Benchmark-Simulation-Models)
   * `simulations`: Simulations started by the `estimation.jl` scripts
* `plots`: Generated plots from all the scripts
* `scripts`
   * `estimation.jl`: Estimation with SEM and PEM methods according to parameters given in kwargs
   * `generate_data.jl`: Generation of data using the ASM1 model
   * `plot_results.jl`: Comparison of PEM and SEM methods according to simulations present in the `data/simulations` folder
   * `run_experience.jl`: Starts the `estimation.jl` script multiple times with different parameters
   * `simple_example.jl`: Estimation of the proposed model and estimation with SEM
* `src`: Content used by the different scripts

# Installation

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> SEM-WTTP-nh4-concentration

It is authored by Victor Bertret, Valérie Monbet and Roman Legoff Latimier.

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:
```julia
using DrWatson
@quickactivate "SEM-WTTP-nh4-concentration"
```

For more details, see the [DrWatson repository](https://juliadynamics.github.io/DrWatson.jl/stable/).

# Usage

If you have carefully followed the [Installation](#Installation) steps, you can launch all the scripts simply using:

```
julia scripts/estimation.jl
```

If you want to try different parameters for the `estimation.jl` scripts, you can specify them directly with command lines:

```
julia scripts/estimation.jl nb_exp X_in
```

Here, `nb_exp` is the number of experiments, and `X_in` is the fixed and constant inlet concentration of ammonium.