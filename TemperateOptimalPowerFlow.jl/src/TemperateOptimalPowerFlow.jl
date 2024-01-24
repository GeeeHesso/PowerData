module TemperateOptimalPowerFlow

using PowerModels
using DataFrames
using CSV

export add_line_costs!
export create_list_of_loads, get_loads_info, export_loads_info
export create_list_of_gens, get_gens_info, export_gens_info
export assign_loads!, assign_loads_from_file!


"Adds a 'cost' field to all branches of a PowerModels network data"
function add_line_costs!(network::Dict{String,Any}, cost::Float64)
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


"Gets some info about a list of loads from a PowerModels network data and export it to a CSV file"
function export_loads_info(network::Dict{String,Any}, list_of_loads::Vector{String}, info::Vector{String}, file::String)
    CSV.write(file, get_loads_info(network, list_of_loads, info))
    nothing
end


"Gets some info about a list of generators from a PowerModels network data and export it to a CSV file"
function export_gens_info(network::Dict{String,Any}, list_of_gens::Vector{String}, info::Vector{String}, file::String)
    CSV.write(file, get_gens_info(network, list_of_gens, info))
    nothing
end


"Assign the loads from a list."
function assign_loads!(network::Dict{String,Any}, load_ids::Vector{String}, loads::Vector{Float64})
    # read the desired column from the CSV file
    if network["per_unit"]
        loads /= network["baseMVA"]
    end
    load_count = length(loads)
    for i = 1:load_count
        network["load"][load_ids[i]]["pd"] = loads[i]
    end
    nothing
end


"Assign the loads from a CSV file whose columns are timesteps and whose rows correspond to a list of loads."
function assign_loads_from_file!(network::Dict{String,Any}, load_ids::Vector{String}, file::String, timestep::Int)
    column = string(timestep)
    loads = CSV.File(file, select=[column])[column]
    assign_loads!(network, load_ids, loads)
    nothing
end


end # module TemperateOptimalPowerFlow
