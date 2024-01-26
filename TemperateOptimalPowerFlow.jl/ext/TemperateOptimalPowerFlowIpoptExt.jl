module TemperateOptimalPowerFlowIpoptExt

using TemperateOptimalPowerFlow
using Ipopt
using PowerModels

function TemperateOptimalPowerFlow.get_optimizer()
    Ipopt.Optimizer
end

function TemperateOptimalPowerFlow.get_silent_optimizer()
    optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
end

end # module TemperateOptimalPowerFlowIpoptExt