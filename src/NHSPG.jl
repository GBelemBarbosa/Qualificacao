export NHSPGModel

@with_kw mutable struct NHSPGModel <: AbstractCompositeSolver
    method = :NHSPG
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()

    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    γ₀:: Number = NaN
    m:: Int64   = 5
    τ:: Float64 = 0.25
    δ:: Float64 = 0.01
        
    γₘᵢₙ:: Number = eps()
    γₘₐₓ:: Number = typemax(Int64)
end

function NHSPG(ProblemModel:: AbstractCompositeModel, SolverModel:: NHSPGModel) 
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)
    nF₌pr₊1_hist = Vector{Int64}(undef, kₘₐₓ+1)
    nF₌pr₊1_hist[1:2] .= 1
    gr₊1 = 0
    
    T₀ = time()
    @unpack_NHSPGModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.F, ProblemModel.∇f, ProblemModel.prox
    x_best = rₖ = sₖ = xₖ₋₁ = xₖ = x₀
    F_best = Fxₖ = F(x₀)
    ∇fxₖ₋₁ = ∇fxₖ = ∇f(xₖ)
    if isnan(γ₀)
        γ₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-∇f(x₀.+10^-5))
        gr₊1 = 1
        
        @info "γ₀ = $(@sprintf "%.3e" γ₀)"
    end
    nsₖ₋₁ = nsₖ = γₖ = γ₀
    lastₘ = [Fxₖ for i = 1:m]

    F_hist[1] = Fxₖ
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        i = 0
        while true
            xₖ = prox(γₖ, xₖ₋₁, ∇fxₖ)
            Fxₖ = F(xₖ)
            sₖ = xₖ.-xₖ₋₁
            nsₖ = sₖ'sₖ

            nF₌pr₊1_hist[k+1] += 1

            if Fxₖ+δ*nsₖ/(2*γₖ) <= Fxₗ₍ₖ₎ 
                break
            end
            i += 1

            i == 1 && k != 1 ? γₖ = nsₖ₋₁/(γₖ*rₖ'rₖ) : γₖ *= τ               

            if isnan(γₖ) || γₖ < γₘᵢₙ
                @warn "Busca linear resultou em isnan(γₖ) || γₖ < γₘᵢₙ" isnan(γₖ) γₖ < γₘᵢₙ
                status = :exception
                
                break
            end
        end
        ∇fxₖ₋₁, ∇fxₖ = ∇fxₖ, ∇f(xₖ)

        if F_best > Fxₖ
            F_best = Fxₖ
            x_best = xₖ
        end

        T₁ = time()
        F_hist[k+1] = Fxₖ
        ⎷nψₖ = norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ)./γₖ, p)+Inf*(k == 1)
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
        nF₌pr₊1_hist[k+1] = nF₌pr₊1_hist[k]

        popfirst!(lastₘ)
        push!(lastₘ, Fxₖ)
        xₖ₋₁ = xₖ
        rₖ = ∇fxₖ.-∇fxₖ₋₁
        γₖ = nsₖ/(sₖ'rₖ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsₖ/(rₖ'rₖ))
        end
        nsₖ₋₁ = nsₖ
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
                    nF_hist = nF₌pr₊1_hist[1:k+1],
                    pr_hist = nF₌pr₊1_hist[2:k+1].-1,
                    gr_hist = [i for i = 1+gr₊1:k+gr₊1],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end

function NHSPGopt(ProblemModel:: AbstractCompositeModel, SolverModel:: NHSPGModel) 
    @unpack_NHSPGModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.Fopt, ProblemModel.∇fopt, ProblemModel.prox
    
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)
    nF₌pr₊1_hist = Vector{Int64}(undef, kₘₐₓ+1)
    nF₌pr₊1_hist[1:2] .= 1
    gr₊1 = 0

    T₀ = time()
    x_best = rₖ = sₖ = xₖ₋₁ = xₖ = x₀
    Fxₖ, aux = F(xₖ)
    F_best = Fxₖ
    ∇fxₖ₋₁ = ∇fxₖ = ∇f(xₖ, aux)
    if isnan(γ₀)
        γ₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-ProblemModel.∇f(x₀.+10^-5))
        gr₊1 = 1
        
        @info "γ₀ = $(@sprintf "%.3e" γ₀)"
    end
    nsₖ₋₁ = nsₖ = γₖ = γ₀
    lastₘ = [Fxₖ for i = 1:m]

    F_hist[1] = Fxₖ
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        Fxₗ₍ₖ₎ = maximum(lastₘ)

        i = 0
        while true
            xₖ = prox(γₖ, xₖ₋₁, ∇fxₖ)
            Fxₖ, aux = F(xₖ)
            sₖ = xₖ.-xₖ₋₁
            nsₖ = sₖ'sₖ

            nF₌pr₊1_hist[k+1] += 1

            if Fxₖ+δ*nsₖ/(2*γₖ) <= Fxₗ₍ₖ₎ 
                break
            end
            i += 1

            i == 1 && k != 1 ? γₖ = nsₖ₋₁/(γₖ*rₖ'rₖ) : γₖ *= τ 

            if isnan(γₖ) || γₖ < γₘᵢₙ
                @warn "Busca linear resultou em isnan(γₖ) || γₖ < γₘᵢₙ" isnan(γₖ) γₖ < γₘᵢₙ
                status = :exception
                
                break
            end
        end
        ∇fxₖ₋₁, ∇fxₖ = ∇fxₖ, ∇f(xₖ, aux)

        if F_best > Fxₖ
            F_best = Fxₖ
            x_best = xₖ
        end

        T₁ = time()
        F_hist[k+1] = Fxₖ
        ⎷nψₖ = norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ)./γₖ, p)+Inf*(k == 1)
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
        nF₌pr₊1_hist[k+1] = nF₌pr₊1_hist[k]

        popfirst!(lastₘ)
        push!(lastₘ, Fxₖ)
        xₖ₋₁ = xₖ
        rₖ = ∇fxₖ.-∇fxₖ₋₁
        γₖ = nsₖ/(sₖ'rₖ)
        if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
            γₖ = sqrt(nsₖ/(rₖ'rₖ))
        end
        nsₖ₋₁ = nsₖ
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
                    nF_hist = nF₌pr₊1_hist[1:k+1],
                    pr_hist = nF₌pr₊1_hist[2:k+1].-1,
                    gr_hist = [i for i = 1+gr₊1:k+gr₊1],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end