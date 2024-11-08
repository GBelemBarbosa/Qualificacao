export PGModel

@with_kw mutable struct PGModel <: AbstractCompositeSolver
    method = :PG
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    L:: Number = NaN
end

function PG(ProblemModel:: AbstractCompositeModel, SolverModel:: PGModel)
    @unpack_PGModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.F, ProblemModel.∇f, ProblemModel.prox

    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    F_hist[1] = ProblemModel.F(x₀)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)

    T₀ = time()
    xₖ₋₁ = xₖ = x₀
    ∇fxₖ₋₁ = ∇fxₖ = ∇f(xₖ)
    if isnan(L)
        L = ProblemModel.L
        @info "L = $(@sprintf "%.3e" L)"
    end
    Lᵢₙᵥ = 1/L
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        xₖ = prox(Lᵢₙᵥ, xₖ, ∇fxₖ) 

        T₁ = time()
        F_hist[k+1] = F(xₖ)
        ⎷nψₖ = norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ).*L, p)+Inf*(k == 1)
        crit_hist[k] = ⎷nψₖ
        T₀ += time()-T₁ # Desconta o tempo entre T₁ e aqui

        if ⎷nψₖ < ϵ 
            status = :optimal
        elseif k == kₘₐₓ
            status = :max_iter
        elseif time()-T₀ >= Tₘₐₓ
            status = :max_time
        end
        if status != :running
            break
        end
        k += 1
        
        xₖ₋₁ = xₖ
        ∇fxₖ₋₁, ∇fxₖ = ∇fxₖ, ∇f(xₖ)
    end

    return CompositeExecutionStats(
                    status = status,
                    problem = ProblemModel,
                    solver = SolverModel,
                    solution = xₖ,
                    objective = F_hist[k+1],
                    criticality = crit_hist[k],
                    total_iter = k,
                    elapsed_time = time()-T₀,
                    nF_hist = zeros(k+1),
                    pr_hist = [i for i = 1:k],
                    gr_hist = [i for i = 1:k],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end

function PGopt(ProblemModel:: AbstractCompositeModel, SolverModel:: PGModel)
    @unpack_PGModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.Fopt, ProblemModel.∇fopt, ProblemModel.prox
    
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    F_hist[1], aux = F(x₀)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)

    T₀ = time()
    xₖ₋₁ = xₖ = x₀
    ∇fxₖ₋₁ = ∇fxₖ = ∇f(xₖ, aux)
    if isnan(L)
        L = ProblemModel.L
        @info "L = $(@sprintf "%.3e" L)"
    end
    Lᵢₙᵥ = 1/L
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        xₖ = prox(Lᵢₙᵥ, xₖ, ∇fxₖ) 

        T₁ = time()
        F_hist[k+1], aux = F(xₖ)
        ⎷nψₖ = norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ).*L, p)+Inf*(k == 1)
        crit_hist[k] = ⎷nψₖ
        T₀ += time()-T₁ # Desconta o tempo entre T₁ e aqui

        if ⎷nψₖ < ϵ 
            status = :optimal
        elseif k == kₘₐₓ
            status = :max_iter
        elseif time()-T₀ >= Tₘₐₓ
            status = :max_time
        end
        if status != :running
            break
        end
        k += 1
        
        xₖ₋₁ = xₖ
        ∇fxₖ₋₁, ∇fxₖ = ∇fxₖ, ∇f(xₖ, aux)
    end

    return CompositeExecutionStats(
                    status = status,
                    problem = ProblemModel,
                    solver = SolverModel,
                    solution = xₖ,
                    objective = F_hist[k+1],
                    criticality = crit_hist[k],
                    total_iter = k,
                    elapsed_time = time()-T₀,
                    nF_hist = zeros(k+1),
                    pr_hist = [i for i = 1:k],
                    gr_hist = [i for i = 1:k],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end