module TemperateOptimalPowerFlow


using PowerModels
using DataFrames
using CSV
using ProgressMeter


export create_list_of_loads, get_loads_info, create_list_of_gens, get_gens_info
export assign_loads!, assign_loads_from_file!, assign_costs!, assign_costs_from_file!
export add_line_costs!, iterate_dc_opf


"Adds a 'cost' field to all branches of a PowerModels network data"
function add_line_costs!(network::Dict{String,Any}, cost::Real)
    for (key, branch) in network["branch"]
        branch["cost"] = cost
    end
    nothing
end


"Creates a list of loads from a PowerModels network data"
function create_list_of_loads(network::Dict{String,Any}) :: Vector{String}
    return collect(keys(network["load"]))
end


"Creates a list of generators from a PowerModels network data"
function create_list_of_gens(network::Dict{String,Any}) :: Vector{String}
    return collect(keys(network["gen"]))
end


"Gets some info about a list of loads from a PowerModels network data and turns it into a DataFrame"
function get_loads_info(network::Dict{String,Any}, list_of_loads::Vector{String}, info::Vector{String}) :: DataFrame
    load_info = Dict(i => Vector{Any}() for i in info)
    for load_id in list_of_loads
        bus_id = string(network["load"][load_id]["load_bus"])
        for i in info
            value = i in keys(network["load"][load_id]) ? network["load"][load_id][i] : network["bus"][bus_id][i]
            push!(load_info[i], value)
        end
    end
    dataframe = DataFrame(id = list_of_loads)
    for i in info
        dataframe[!, i] = load_info[i]
    end
    return dataframe
end


"Gets some info about a list of generators from a PowerModels network data and turns it into a DataFrame"
function get_gens_info(network::Dict{String,Any}, list_of_gens::Vector{String}, info::Vector{String}) :: DataFrame
    gen_info = Dict(i => Vector{Any}() for i in info)
    for gen_id in list_of_gens
        bus_id = string(network["gen"][gen_id]["gen_bus"])
        for i in info
            value = i in keys(network["gen"][gen_id]) ? network["gen"][gen_id][i] : network["bus"][bus_id][i]
            push!(gen_info[i], value)
        end
    end
    dataframe = DataFrame(id = list_of_gens)
    for i in info
        dataframe[!, i] = gen_info[i]
    end
    return dataframe
end


"Assign the loads from a list."
function assign_loads!(network::Dict{String,Any}, load_ids::Vector{String}, loads::Vector{Float64})
    if network["per_unit"]
        loads /= network["baseMVA"]
    end
    for i = 1:length(loads)
        network["load"][load_ids[i]]["pd"] = loads[i]
    end
    nothing
end


"Assign the loads from a CSV file whose columns are timesteps and whose rows correspond to a list of loads."
function assign_loads_from_file!(network::Dict{String,Any}, file::String, timestep::Int)
    column = string(timestep)
    data = CSV.File(file, select=["id", column])
    assign_loads!(network, string.(data["id"]), data[column])
    nothing
end


"Assign the loads from a dataframe"
function assign_loads!(network::Dict{String,Any}, dataframe::DataFrame,  timestep::Int)
    assign_loads!(network, string.(dataframe[:, "id"]), dataframe[:, string(timestep)])
    nothing
end


"Assign the production costs from a list"
function assign_costs!(network::Dict{String,Any}, gen_ids::Vector{String}, costs::Vector{Float64})
    for i = 1:length(gen_ids)
        network["gen"][gen_ids[i]]["cost"][1] = costs[i]
    end
    nothing
end


"Assign the production costs from a CSV file whose columns are timesteps and whose rows correspond to a list of gens."
function assign_costs_from_file!(network::Dict{String,Any}, file::String, timestep::Int)
    column = string(timestep)
    data = CSV.File(file, select=["id", column])
    assign_costs!(network, string.(data["id"]), data[column])
    nothing
end


"Assign the production costs from a dataframe"
function assign_costs!(network::Dict{String,Any}, dataframe::DataFrame,  timestep::Int)
    assign_costs!(network, string.(dataframe[:, "id"]), dataframe[:, string(timestep)])
    nothing
end


"Raise the thermal limit for all lines by a given factor (by default 10%)"
function raise_thermal_limit!(network::Dict{String,Any}, factor::Float64 = 1.1)
    for (id, branch) in network["branch"]
		branch["rate_a"] *= factor
	end
    nothing
end


""""Perform the OPF computation for each time step, given time series for loads and production costs.
The production of a given list of generators is recorded into a file. Existing data is not overwritten."""
function iterate_dc_opf(network::Dict{String,Any}, loads_file::String, costs_file::String,
    gen_info_file::String, output_file::String, optimizer,
    timesteps_range::UnitRange{Int} = 1:0, save_interval::Int = 150, raise_thermal_limit::Int = 5)
    # load time series into memory
    loads = CSV.read(loads_file, DataFrame)
    costs = CSV.read(costs_file, DataFrame)
    list_of_gens = string.(CSV.read(gen_info_file, DataFrame).id)
    # if the output file exists already, load its data, otherwise create an empty dataframe
    output = isfile(output_file) ? CSV.read(output_file, DataFrame) : DataFrame(id=list_of_gens)
    # call the main function
    iterate_dc_opf!(network, loads, costs, list_of_gens, output, output_file,
        optimizer, timesteps_range, save_interval, raise_thermal_limit)
    nothing
end


"""Recursive logic of the function iterate_dc_opf"""
function iterate_dc_opf!(network::Dict{String,Any}, loads::DataFrame, costs::DataFrame,
    list_of_gens::Vector{String}, output::DataFrame, output_file::String, optimizer,
    timesteps_range::UnitRange{Int}, save_interval::Int, raise_thermal_limit)
    # if no range for the timesteps is defined, determine the range from the loads
    timesteps_count = length(timesteps_range)
    if timesteps_count == 0
        timesteps_count = ncol(loads) - 1
        timesteps_range = 0:timesteps_count - 1
    end
    # loop over all time steps
    failed_opf_count = 0
    @showprogress for timestep=timesteps_range
        # if the timestep is already present in the dataframe, then ignore it
        column = string(timestep)
        if column in names(output)
            continue
        end
        # prepare the network
        assign_loads!(network, loads, timestep)
        assign_costs!(network, costs, timestep)
        # perform OPF computation
        opf_solution = solve_dc_opf(network, optimizer)["solution"]
        # if the OPF did not converge, do nothing
        if length(opf_solution) == 0
            failed_opf_count += 1
            println("OPF did not converge at time step ", timestep)
            continue
        end
        # otherwise, save the production data into the dataframe
        output[!, column] = [opf_solution["gen"][gen_id]["pg"] for gen_id in list_of_gens]
        # save the dataframe
        if timestep % save_interval == 0
            CSV.write(output_file, output)
        end
    end
    # determine whether some OPF did not converge, and raise the thermal limit if needed
    if failed_opf_count > 0
        println("OPF did not converge for ", failed_opf_count, " time steps")
        if raise_thermal_limit > 0
            println(" -> raising the thermal limit")
            raise_thermal_limit!(network)
            iterate_dc_opf!(network, loads, costs, list_of_gens, output, output_file, optimizer,
                timesteps_range, save_interval, raise_thermal_limit - 1)
            return nothing
        end
    end
    println("OPF converged for ", timesteps_count, " time steps")
    # re-order the columns before saving the result
    columns = names(output)
    columns_int = (t -> parse(Int, t)).(columns[columns .!= "id"])
    columns_sorted = string.(sort(columns_int))
    select!(output, "id", columns_sorted)
    println("Saving time series with ", length(columns_sorted), " time steps")
    CSV.write(output_file, output)
    nothing
end


end # module TemperateOptimalPowerFlow