label = length(ARGS) > 0 ? "_$(ARGS[1])" : ""


using JuMP, OrderedCollections, Statistics, DataDrop
using Ipopt

function get_optimizer()
    return optimizer_with_attributes(Ipopt.Optimizer, "print_level" => 0)
end

function get_model()
    model = Model(get_optimizer(), add_bridges = false)
    set_string_names_on_creation(model, false)
    return model
end

function total(array, dims)
    return dropdims(sum(array, dims=dims), dims=dims)
end

function average(array, dims)
    return dropdims(mean(array, dims=dims), dims=dims)
end

function opf(Q::AbstractArray{<:Real,2}, L::AbstractArray{<:Real,2}, 
        P_max::AbstractVector{<:Real}, P_total::AbstractVector{<:Real},
        A_constraints::AbstractArray{<:Real,2}, P_constraints::AbstractVector{<:Real},
        A_ramp::AbstractArray{<:Real,2} = Array{Real}(undef, 0, 0), ΔP_ramp_max::AbstractVector{<:Real} = Real[],
        P_ramp_first::AbstractVector{<:Real} = Real[], P_ramp_last::AbstractVector{<:Real} = Real[];
        indent::Int = 0)
    
    N = length(P_max)
    T = length(P_total)
    n_constraints = length(P_constraints)
    n_ramp = length(ΔP_ramp_max)

    # check dimensions of the input
    @assert size(Q) == (N, N)
    @assert size(L) == (N, T)
    @assert size(A_constraints) == (n_constraints, N)
    @assert (size(A_ramp) == (n_ramp, N)) || (n_ramp == 0)
    @assert length(P_ramp_first) ∈ [0, n_ramp]
    @assert length(P_ramp_last) == length(P_ramp_first)
    
    print(" "^indent)
    print("OPF with $T time steps, $N generators, $n_constraints annual constraints")
    if n_ramp > 0
        if length(P_ramp_first) == 0
            print(", and $n_ramp ramp constraints (cyclic)")
        else
            print(", and $n_ramp ramp constraints (fixed boundaries)")
        end
    end
    println()
    
    # check feasibility of the model
    @assert all(P_constraints .<= A_constraints * P_max)
    @assert all(ΔP_ramp_max .>= 0)
    @assert all(P_total .<= sum(P_max))
    
    print(" "^indent)
    println("Creating the model...")
    flush(stdout)
    model = get_model()
    # variables
    @variable(model, 0.0 <= P[i=1:N, t=1:T] <= P_max[i])
    # constraints 
    @constraint(model, total(P, 1) .== P_total)
    @constraint(model, average(A_constraints * P, 2) .== P_constraints)
    if n_ramp > 0
        @expression(model, P_ramp, A_ramp * P);
        if length(P_ramp_first) == 0
            @expression(model, ΔP_ramp[i=1:n_ramp, t=1:T], P_ramp[i, t] - P_ramp[i, t % T + 1])
            @constraint(model, ΔP_ramp .<= ΔP_ramp_max)
            @constraint(model, ΔP_ramp .>= -ΔP_ramp_max)
        else
            @expression(model, ΔP_ramp[i=1:n_ramp, t=1:T-1], P_ramp[i, t] - P_ramp[i, t+1])
            @constraint(model, ΔP_ramp .<= ΔP_ramp_max)
            @constraint(model, ΔP_ramp .>= -ΔP_ramp_max)
            @expression(model, ΔP_ramp_first[i=1:n_ramp], P_ramp[i, 1] - P_ramp_first[i])
            @constraint(model, ΔP_ramp_first .<= ΔP_ramp_max)
            @constraint(model, ΔP_ramp_first .>= -ΔP_ramp_max)
            @expression(model, ΔP_ramp_last[i=1:n_ramp], P_ramp[i, T] - P_ramp_last[i])
            @constraint(model, ΔP_ramp_last .<= ΔP_ramp_max)
            @constraint(model, ΔP_ramp_last .>= -ΔP_ramp_max)
        end
    end
    # cost function
    @objective(model, Min, QuadExpr(sum(L[i, t] * P[i, t] for i=1:N for t=1:T),
        OrderedDict(UnorderedPair(P[i,t], P[j,t]) => (i == j ? 1 : 2)  * Q[i,j]
        for i=1:N for j=i:N for t=1:T)))

    print(" "^indent)
    println("Optimizing...")
    flush(stdout)
    optimize!(model)

    if termination_status(model) ∈ [OPTIMAL, LOCALLY_SOLVED] 
        return value.(P)
    end
    return termination_status(model)
end

function partitioned_opf(partitions::Vector{Int},
        Q::AbstractArray{<:Real,2}, L::AbstractArray{<:Real,2}, 
        P_max::AbstractVector{<:Real}, P_total::AbstractVector{<:Real},
        A_constraints::AbstractArray{<:Real,2}, P_constraints::AbstractVector{<:Real},
        A_ramp::AbstractArray{<:Real,2} = Array{Real}(undef, 0, 0), ΔP_ramp_max::AbstractVector{<:Real} = Real[],
        P_ramp_first::AbstractVector{<:Real} = Real[], P_ramp_last::AbstractVector{<:Real} = Real[];
        indent::Int = 0)
    
    N = length(P_max)
    T = length(P_total)
    n_constraints = length(P_constraints)
    n_ramp = length(ΔP_ramp_max)

    # check that the number of partitions matches the total number of steps
    @assert prod(partitions) == T

    if length(partitions) == 1
        return opf(Q, L, P_max, P_total, A_constraints, P_constraints,
            A_ramp, ΔP_ramp_max, P_ramp_first, P_ramp_last, indent = indent + 4)
    end

    # check dimensions of the input that needs to be partitioned
    @assert size(L) == (N, T)
    @assert size(A_constraints) == (n_constraints, N)
    @assert (size(A_ramp) == (n_ramp, N)) || (n_ramp == 0)

    n_partitions = partitions[1]
    partition_length = T ÷ n_partitions
    
    println()
    print(" "^indent)
    println(">>> Partitioning a dataset of $T time steps into $n_partitions chunks of $partition_length time steps")
    println()

    partitioned_P_total = reshape(P_total, (partition_length, n_partitions))
    aggregated_P_total = average(partitioned_P_total, 1)

    partitioned_L = reshape(L, (N, partition_length, n_partitions))
    aggregated_L = average(partitioned_L, 2)
    
    aggregated_P = opf(Q, aggregated_L, P_max, aggregated_P_total, A_constraints, P_constraints,
        A_ramp, ΔP_ramp_max, P_ramp_first, P_ramp_last, indent = indent + 4)

    aggregated_P = min.(aggregated_P, P_max)
    partitioned_P_constraints = A_constraints * aggregated_P
    partitioned_P_ramp = n_ramp > 0 ? A_ramp * aggregated_P : Real[]

    result = Matrix{Float64}(undef, N, 0)
    timing = []
    for a=1:n_partitions
        println()
        print(" "^indent)
        print(">>> Step $a of $n_partitions")
        if length(timing) > 0
            print(" (estimated remaining time:")
            s = round(Int, (n_partitions - a + 1) * mean(timing))
            if s >= 60
                m = s ÷ 60
                s = s % 60
                if m >= 60
                    h = m ÷ 60
                    m = m % 60
                    print(" $h h")
                end
                print(" $m min")
            end
            print(" $s s)")
        end
        println()
        
        if n_ramp == 0
            partitioned_P_ramp_previous = Real[]
            partitioned_P_ramp_next = Real[]
        else
            if a == 1
                partitioned_P_ramp_previous = length(P_ramp_first) > 0 ? P_ramp_first : partitioned_P_ramp[:, end]
            else
                partitioned_P_ramp_previous = partitioned_P_ramp[:, a - 1]
            end
            if a == n_partitions
                partitioned_P_ramp_next = length(P_ramp_last) > 0 ? P_ramp_last : partitioned_P_ramp[:, 1]
            else
                partitioned_P_ramp_next = partitioned_P_ramp[:, a + 1]
            end
        end
        
        partition_result = @timed partitioned_opf(partitions[2:end], Q, partitioned_L[:,:,a],
            P_max, partitioned_P_total[:,a], A_constraints, partitioned_P_constraints[:,a],
            A_ramp, ΔP_ramp_max, partitioned_P_ramp_previous, partitioned_P_ramp_next, indent = indent + 4)
        push!(timing, partition_result.time)
        result = hcat(result, partition_result.value)
    end

    return result
end

Q = DataDrop.retrieve_matrix("quadratic_cost$label.h5")
L = DataDrop.retrieve_matrix("linear_cost$label.h5")
P_max = DataDrop.retrieve_matrix("P_max_gen.h5")
P_total = DataDrop.retrieve_matrix("P_total.h5")
A_constraints = DataDrop.retrieve_matrix("A_gen_total.h5")
P_constraints = DataDrop.retrieve_matrix("gen_total.h5")
A_ramp = DataDrop.retrieve_matrix("A_gen_ramp.h5");
ΔP_ramp = DataDrop.retrieve_matrix("gen_ramp.h5");

result = partitioned_opf([52, 168], Q, L, P_max, P_total, A_constraints, P_constraints, A_ramp, ΔP_ramp)

result = map(x -> isapprox(x, 0, atol=1e-6) ? 0.0 : x, result)

DataDrop.store_matrix("P_result$label.h5", result)

