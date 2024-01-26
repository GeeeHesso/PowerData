module TemperateOptimalPowerFlowGurobiExt

using TemperateOptimalPowerFlow
using Gurobi
using PowerModels

function __init__()
    global gurobi_env = Gurobi.Env()
end

function TemperateOptimalPowerFlow.get_optimizer()
    () -> Gurobi.Optimizer(gurobi_env)
end

function TemperateOptimalPowerFlow.get_silent_optimizer()
    optimizer_with_attributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0)
end

end # module TemperateOptimalPowerFlowGurobiExt
