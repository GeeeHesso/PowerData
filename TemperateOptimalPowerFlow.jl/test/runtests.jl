using Test
using PowerModels
using TemperateOptimalPowerFlow
using DataFrames


@testset "TemperateOptimalPowerFlow.jl" begin

    @testset "test_network_import_and_preparation" begin
        # import a simple model with 3 buses (see PowerModels.jl)
        network = parse_file("case3.m")
        @test isa(network, Dict)
        # verify presence of branches
        @test "branch" in keys(network)
        branches = network["branch"]
        @test isa(branches, Dict)
        @test length(branches) > 0
        # verify that branches do not have a "cost" property
        for (key, branch) in network["branch"]
            @test "cost" ∉ keys(branches)
        end
        # add a branch cost
        cost = 123.4
        add_line_costs!(network, cost)
        # verify that all branches now have a "cost" property
        for (key, branch) in network["branch"]
            @test "cost" in keys(branch)
            @test branch["cost"] == cost
        end
    end

    @testset "test_create_list_of_loads" begin
        # import a simple model with 3 buses (see PowerModels.jl)
        network = parse_file("case3.m")
        @test isa(network, Dict)
        # create a list of loads
        @test issetequal(create_list_of_loads(network), ["1", "2", "3"])
    end

    @testset "test_create_list_of_gens" begin
        # import a simple model with 3 buses (see PowerModels.jl)
        network = parse_file("case3.m")
        @test isa(network, Dict)
        # create a list of loads
        @test issetequal(create_list_of_gens(network), ["1", "2", "3"])
    end

    @testset "test_get_loads_info" begin
        # import PanTaGruEl
        network = parse_file("../models/pantagruel.json")
        @test isa(network, Dict)
        # create a list of loads
        list_of_loads = create_list_of_loads(network)
        @test length(list_of_loads) > 0
        # get info about the load's country and population
        df = get_loads_info(network, list_of_loads, ["country", "load_prop"])
        @test isa(df, DataFrame)
        # for each country, check that the 'load_prop' adds up to one
        countries = unique(df[:, "country"])
        for country in countries
            @test abs(1.0 - sum(df[df[:, "country"] .== country, "load_prop"])) < 1.0e-6
        end
    end

    @testset "test_get_gens_info" begin
        # import a simple model with 3 buses (see PowerModels.jl)
        network = parse_file("case3.m")
        @test isa(network, Dict)
        # create a list of gens
        list_of_gens = create_list_of_gens(network)
        @test length(list_of_gens) == 3
        # get info about the gen's max capacity
        df = get_gens_info(network, list_of_gens, ["pmax"])
        @test isa(df, DataFrame)
        @test issetequal(df[:, "pmax"], [0, 15.0, 20.0])
    end

    @testset "test_assign_loads" begin
        # import a simple model with 3 buses (see PowerModels.jl)
        network = parse_file("case3.m")
        @test isa(network, Dict)
        # verify that the model uses per units
        @test network["per_unit"]
        @test network["baseMVA"] == 100.0
        # create a list of loads
        list_of_loads = create_list_of_loads(network)
        @test length(list_of_loads) == 3
        # create loads in MW
        loads = [100.0, 200.0, 300.0]
        assign_loads!(network, list_of_loads, loads)
        for i = 1:3
            @test network["load"][list_of_loads[i]]["pd"] == i
        end
    end

end
