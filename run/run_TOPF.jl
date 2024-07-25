noise_label = length(ARGS) > 0 ? ARGS[1] : 1
noise_factor = parse(Float64, noise_label)

using TemperateOptimalPowerFlow
using Gurobi

compute("data/TOPF_run", "P_result_$(noise_label)", [52, 168], noise_factor)
# compute("data/TOPF_run", "P_result_$(noise_label)", [13, 28, 24], noise_factor)
