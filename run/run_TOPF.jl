using TemperateOptimalPowerFlow
using Gurobi

compute("data/europe_opf_test", "P_result", [52, 168])
# compute("data/europe_opf_test", "P_result", [13, 28, 24])
