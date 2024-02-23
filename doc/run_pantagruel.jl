# This is a script intended to run the OPF for all time steps of a series
# with the model PanTaGruEl and the optimizer Ipopt

using PowerModels
using TemperateOptimalPowerFlow
using Ipopt

label = length(ARGS) > 0 ? "_$(ARGS[1])" : ""

network = parse_file("pantagruel.json")
add_line_costs!(network, 2000)

assign_ramp_max!(network, 1.0, ["Nuclear", "nuclear", "nuclear_cons"]) # max ramp up/down of 100 MW per hour

iterate_dc_opf(network,
	"pantagruel_load_series$(label).csv", 
	"pantagruel_gen_cost_series$(label).csv",
	"pantagruel_gen_series$(label).csv", 
	get_silent_optimizer())
