export FISTAModel

@with_kw mutable struct FISTAModel <: AbstractCompositeSolver 
    method = :FISTA
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}() 
    
    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    L:: Number = NaN
end

function FISTA(ProblemModel:: AbstractCompositeModel, SolverModel:: FISTAModel) 
    @unpack_FISTAModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.F, ProblemModel.∇f, ProblemModel.prox

    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    F_hist[1] = ProblemModel.F(x₀)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)

    T₀ = time()
    yₖ = xₖ₋₁ = xₖ = x₀
    if isnan(L)
        L = ProblemModel.L
        @info "L = $(@sprintf "%.3e" L)"
    end
    Lᵢₙᵥ = 1/L
    tₖ = 1.0
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        ∇fyₖ = ∇f(yₖ)
        xₖ = prox(Lᵢₙᵥ, yₖ, ∇fyₖ) 

        T₁ = time()
        F_hist[k+1] = F(xₖ)
        ⎷nψₖ = norm(∇f(xₖ).-∇fyₖ.+(yₖ.-xₖ).*L, p)
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

        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ = xₖ.+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁ = xₖ
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

function FISTAopt(ProblemModel:: AbstractCompositeModel, SolverModel:: FISTAModel) 
    @unpack_FISTAModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox, ∇fy = ProblemModel.Fopt, ProblemModel.∇fopt, ProblemModel.prox, ProblemModel.∇f
    
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    F_hist[1] = ProblemModel.F(x₀)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)

    T₀ = time()
    yₖ = xₖ₋₁ = xₖ = x₀
    if isnan(L)
        L = ProblemModel.L
        @info "L = $(@sprintf "%.3e" L)"
    end
    Lᵢₙᵥ = 1/L
    tₖ = 1.0
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        ∇fyₖ = ∇fy(yₖ)
        xₖ = prox(Lᵢₙᵥ, yₖ, ∇fyₖ) 

        T₁ = time()
        F_hist[k+1], aux = F(xₖ)
        ⎷nψₖ = norm(∇f(xₖ, aux).-∇fyₖ.+(yₖ.-xₖ).*L, p)
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

        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ = xₖ.+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁ = xₖ
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