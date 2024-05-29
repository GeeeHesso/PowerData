noise_factor = length(ARGS) > 0 ? parse(Float64, ARGS[1]) : 1.0

using MiniLoggers

logger = MiniLogger(format="[{timestamp:blue}] {group:red:bold} {message}")
global_logger(logger);

using DataDrop
import MathOptInterface as MOI
using Gurobi

const gurobi_env = Gurobi.Env()

optimizer = MOI.instantiate(MOI.OptimizerWithAttributes(() -> Gurobi.Optimizer(gurobi_env), "OutputFlag" => 0))

function opf(quadratic_cost::AbstractArray{<:Real,2}, linear_cost::AbstractArray{<:Real,2}, 
        P_max::AbstractVector{<:Real}, P_exp::AbstractVector{<:Real}, P_total::AbstractVector{<:Real},
        A_ramp::AbstractArray{<:Real,2} = Array{Real}(undef, 0, 0), ΔP_ramp::AbstractVector{<:Real} = Real[],
        P_ramp_first::AbstractVector{<:Real} = Real[], P_ramp_last::AbstractVector{<:Real} = Real[];
        log_group::String = "")
    
    N = length(P_max)
    T = length(P_total)
    n_ramp = length(ΔP_ramp)

    # check dimensions of the input
    @assert length(P_exp) == N
    @assert size(quadratic_cost) == (N, N)
    @assert size(linear_cost) == (N, T)
    @assert (size(A_ramp) == (n_ramp, N)) || (n_ramp == 0)
    @assert length(P_ramp_first) ∈ [0, n_ramp]
    @assert length(P_ramp_last) == length(P_ramp_first)

    ramp_constraint_type = length(P_ramp_first) == 0 ? "periodic" : "fixed boundaries"
    @info ("OPF with $T time steps, $N generators, " *
        "and $n_ramp ramp constraints ($ramp_constraint_type)") _group = log_group
    log_group = " "^length(log_group)
    
    # check feasibility of the model
    @info " -> checking model" _group = log_group
    @assert all(P_exp .<= P_max)
    @assert all(ΔP_ramp .>= 0)
    @assert all(P_total .<= sum(P_max))
    @assert abs(sum(P_total) / sum(P_exp) / T - 1) < 1e-3

    # variables
    @info " -> defining variables" _group = log_group
    P_vec = MOI.add_variables(optimizer, N * T)
    P = reshape(P_vec, (N, T))

    # constraints 
    @info " -> defining constraints" _group = log_group
    MOI.add_constraints(optimizer, P_vec, [MOI.Interval(0.0, P_max[i]) for t = 1:T for i = 1:N])

    MOI.add_constraints(optimizer,
        [MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, P[i,t]) for t = 1:T], 0.0) for i = 2:N],
        [MOI.EqualTo(T * P_exp[i]) for i = 2:N])
    
    MOI.add_constraints(optimizer,
        [MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, P[i,t]) for i = 1:N], 0.0) for t = 1:T],
        [MOI.EqualTo(P_total[t]) for t = 1:T])
    
    if n_ramp > 0
        P_ramp = A_ramp * P
        if length(P_ramp_first) == 0
            ΔP = [P_ramp[n, t] - P_ramp[n, t % T + 1] for t = 1:T for n = 1:n_ramp]
            MOI.add_constraints(optimizer, ΔP, [MOI.GreaterThan(-ΔP_ramp[n]) for t = 1:T for n = 1:n_ramp])
            MOI.add_constraints(optimizer, ΔP, [MOI.LessThan(ΔP_ramp[n]) for t = 1:T for n = 1:n_ramp])
        else
            ΔP = [P_ramp[n, t] - P_ramp[n, t + 1] for t = 1:T-1 for n = 1:n_ramp]
            MOI.add_constraints(optimizer, ΔP, [MOI.GreaterThan(-ΔP_ramp[n]) for t = 1:T-1 for n = 1:n_ramp])
            MOI.add_constraints(optimizer, ΔP, [MOI.LessThan(ΔP_ramp[n]) for t = 1:T-1 for n = 1:n_ramp])
            P_first = [P_ramp[n, 1] for n = 1:n_ramp]
            MOI.add_constraints(optimizer, P_first, [MOI.GreaterThan(P_ramp_first[n] - ΔP_ramp[n]) for n = 1:n_ramp])
            MOI.add_constraints(optimizer, P_first, [MOI.LessThan(P_ramp_first[n] + ΔP_ramp[n]) for n = 1:n_ramp])
            P_last = [P_ramp[n, T] for n = 1:n_ramp]
            MOI.add_constraints(optimizer, P_last, [MOI.GreaterThan(P_ramp_last[n] - ΔP_ramp[n]) for n = 1:n_ramp])
            MOI.add_constraints(optimizer, P_last, [MOI.LessThan(P_ramp_last[n] + ΔP_ramp[n]) for n = 1:n_ramp])
        end
    end
    
    @info " -> computing objective function" _group = log_group
    quadratic_terms = vcat(
        [MOI.ScalarQuadraticTerm(2.0 * quadratic_cost[i,i], P[i, t], P[i, t]) for i = 1:N for t = 1:T],
        [MOI.ScalarQuadraticTerm(quadratic_cost[i,j], P[i, t], P[j, t]) for i = 1:N for j = (i+1):N for t = 1:T]
    )
    affine_terms = [MOI.ScalarAffineTerm(linear_cost[i, t], P[i, t]) for i = 1:N for t = 1:T]
    objective = MOI.ScalarQuadraticFunction(quadratic_terms, affine_terms, 0.0)
    
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), objective)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    
    @info " -> optimizing" _group = log_group
    MOI.optimize!(optimizer)
    
    @info " -> exporting results" _group = log_group
    P_vec_solution = MOI.get(optimizer, MOI.VariablePrimal(), P_vec)
    P_vec_solution = map(x -> isapprox(x, 0, atol=1e-6) ? 0.0 : x, P_vec_solution)
    P_solution = reshape(P_vec_solution, (N, T))
    
    return P_solution
end

function partitioned_opf(partitions::Vector{Int},
        quadratic_cost::AbstractArray{<:Real,2}, linear_cost::AbstractArray{<:Real,2}, 
        P_max::AbstractVector{<:Real}, P_exp::AbstractVector{<:Real}, P_total::AbstractVector{<:Real},
        A_ramp::AbstractArray{<:Real,2} = Array{Real}(undef, 0, 0), ΔP_ramp::AbstractVector{<:Real} = Real[],
        P_ramp_first::AbstractVector{<:Real} = Real[], P_ramp_last::AbstractVector{<:Real} = Real[];
        log_group::String = "")

    if length(partitions) <= 1
        return opf(quadratic_cost, linear_cost, P_max, P_exp, P_total,
            A_ramp, ΔP_ramp, P_ramp_first, P_ramp_last, log_group = log_group)
    end
    
    N = length(P_max)
    T = length(P_total)
    n_ramp = length(ΔP_ramp)

    # check dimensions of the input that needs to be partitioned
    @assert length(P_exp) == N
    @assert size(quadratic_cost) == (N, N)
    @assert size(linear_cost) == (N, T)
    @assert (size(A_ramp) == (n_ramp, N)) || (n_ramp == 0)
    
    # check that the number of partitions matches the total number of steps
    @assert prod(partitions) == T
    
    n_partitions = partitions[1]
    partition_length = T ÷ n_partitions
    
    counter_width = length(string(n_partitions))    
    @info "Partitioning a dataset of $T time steps into $n_partitions chunks of $partition_length time steps" _group = log_group

    partitioned_P_total = reshape(P_total, (partition_length, n_partitions))
    aggregated_P_total = dropdims(sum(partitioned_P_total, dims=1), dims=1) / partition_length

    partitioned_linear_cost = reshape(linear_cost, (N, partition_length, n_partitions))
    aggregated_linear_cost = dropdims(sum(partitioned_linear_cost, dims=2), dims=2) / partition_length

    aggregated_P_max = (P_max + P_exp) / 2.0
    
    partitioned_P_exp = opf(quadratic_cost, aggregated_linear_cost, aggregated_P_max, P_exp, aggregated_P_total,
        A_ramp, ΔP_ramp, P_ramp_first, P_ramp_last,
        log_group = log_group * " $(lpad(0, counter_width))/$(n_partitions)")

    partitioned_P_ramp = n_ramp > 0 ? A_ramp * partitioned_P_exp : Real[]

    result = Matrix{Float64}(undef, N, 0)
    timing = []
    for a=1:n_partitions

        if length(timing) > 0
            estimated_remaining_time = "Estimated remaining time:"
            s = round(Int, (n_partitions - a + 1) * sum(timing) / length(timing))
            if s >= 60
                m = s ÷ 60
                s = s % 60
                if m >= 60
                    h = m ÷ 60
                    m = m % 60
                    estimated_remaining_time = estimated_remaining_time * " $h h"
                end
                estimated_remaining_time = estimated_remaining_time * " $m min"
            end
            estimated_remaining_time = estimated_remaining_time * " $s s"
            @info estimated_remaining_time _group = log_group
        end
        
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
        
        partition_result = @timed partitioned_opf(partitions[2:end], quadratic_cost, partitioned_linear_cost[:,:,a],
            P_max, partitioned_P_exp[:,a], partitioned_P_total[:,a],
            A_ramp, ΔP_ramp, partitioned_P_ramp_previous, partitioned_P_ramp_next,
            log_group = log_group * " $(lpad(a, counter_width))/$(n_partitions)")
        push!(timing, partition_result.time)
        result = hcat(result, partition_result.value)
    end

    return result
end

quadratic_cost = DataDrop.retrieve_matrix("quadratic_cost.h5")
linear_line_cost = DataDrop.retrieve_matrix("linear_line_cost.h5")
linear_gen_cost = DataDrop.retrieve_matrix("linear_gen_cost.h5")
P_max = DataDrop.retrieve_matrix("P_max_gen.h5")
P_exp = DataDrop.retrieve_matrix("P_exp_gen.h5")
P_total = DataDrop.retrieve_matrix("P_total.h5")
A_ramp = DataDrop.retrieve_matrix("A_gen_ramp.h5")
ΔP_ramp = DataDrop.retrieve_matrix("gen_ramp.h5")

result = partitioned_opf([52, 168], quadratic_cost, linear_line_cost + noise_factor * linear_gen_cost,
    P_max, P_exp, P_total, A_ramp, ΔP_ramp)

result = map(x -> isapprox(x, 0, atol=1e-6) ? 0.0 : x, result)

DataDrop.store_matrix("P_result_$(noise_factor).h5", result)

