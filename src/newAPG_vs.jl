export newAPG_vsModel

@with_kw mutable struct newAPG_vsModel <: AbstractCompositeSolver
    method = :newAPG_vs
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    Q:: Function = k:: Int64 -> 0.99^k
    E:: Function = k:: Int64 -> k^-1.1
    λ₁:: Number  = NaN
    μ₀:: Float64 = 0.99 
    μ₁:: Float64 = 0.95 
    c:: Number   = 10^4
    δ:: Float64  = 0.8
end

function newAPG_vs(ProblemModel:: AbstractCompositeModel, SolverModel:: newAPG_vsModel) 
    @unpack_newAPG_vsModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, f, h, ∇f, prox = ProblemModel.F, ProblemModel.f, ProblemModel.h, ProblemModel.∇f, ProblemModel.prox

    F_hist = Vector{Float64}(undef, kₘₐₓ+2)
    Fxₖ = F(x₀)
    F_hist[1] = Fxₖ
    crit_hist = Vector{Float64}(undef, kₘₐₓ+1)
    nF_hist = Vector{Float64}(undef, kₘₐₓ+1)
    nF_hist[1] .= 2
    pr₌gr₊1_hist = Vector{Int64}(undef, kₘₐₓ+1)
    pr₌gr₊1_hist[1:2] .= 1
    gr₊2 = 0
    
    T₀ = time()
    if isnan(λ₁)
        λ₁ = (sqrt(length(x₀))*10^-5)/norm(∇f(x₀).-∇f(x₀.+10^-5))
        gr₊2 = 2

        @info "λ₁ = $(@sprintf "%.3e" λ₁)"
    end
    λₖ₊₁ = λ₁
    xₖ, xₖ₊₁ = x₀, prox(λₖ₊₁, x₀, zeros(length(x₀[1]))) # prox_λ₁h(x₀)
    tₖ₊₁ = 1.0
    
    T₁ = time()
    ∇fxₖ₊₁ = ∇f(xₖ₊₁)
    T∇ = T₁-time() # Tempo descontado quando ∇f(xₖ₊₁) é calculado para definir convergência, mas não é usado pelo método

    crit_hist[1] = norm(∇fxₖ₊₁.+(xₖ.-xₖ₊₁)./λₖ₊₁, p)
    T₀ += time()-T₁ # Desconta o tempo entre T₁ e aqui

    fxₖ₊₁ = f(xₖ₊₁)
    Fxₖ₊₁ = fxₖ₊₁+h(xₖ₊₁)
    if Fxₖ > Fxₖ₊₁
        F_best = Fxₖ₊₁
    else
        F_best = Fxₖ
    end
    sₖ₊₁ = xₖ₊₁.-xₖ
    nsₖ₊₁ = sₖ₊₁'sₖ₊₁

    F_hist[2] = Fxₖ₊₁
    
    k = 2 # Considera o passo prox extra acima
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        if nsₖ₊₁ <= c*(Fxₖ-Fxₖ₊₁) 
            tₖ, tₖ₊₁ = tₖ₊₁, (1+sqrt(1+4*tₖ₊₁^2))/2
            yₖ₊₁ = xₖ.+((tₖ-1)/tₖ₊₁).*sₖ₊₁
            ∇fyₖ₊₁ = ∇f(yₖ₊₁)
            x̂ = prox(λₖ₊₁, yₖ₊₁, ∇fyₖ₊₁)
            fx̂ = f(x̂)
            Fx̂ = fx̂+h(x̂)

            nF_hist[k] += 1
            pr₌gr₊1_hist[k] += 1

            if Fx̂ <= Fxₖ₊₁+min(Q(k), δ*(Fxₖ-Fxₖ₊₁))
                xₖ, xₖ₊₁ = xₖ₊₁, x̂
                x₋yₖ₊₁ = xₖ₊₁.-yₖ₊₁
                nx₋yₖ₊₁ = x₋yₖ₊₁'x₋yₖ₊₁
                sₖ₊₁ = xₖ₊₁.-xₖ
                fyₖ₊₁ = f(yₖ₊₁)
                fxₖ₊₁ = fx̂
                Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, Fx̂

                nF_hist[k] += 1
            else
                xₖ, xₖ₊₁ = xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
                fyₖ₊₁, fxₖ₊₁ = fxₖ₊₁, f(xₖ₊₁)
                Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)

                nF_hist[k] += 1
                pr₌gr₊1_hist[k] += 1
                T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ₊₁ que foi usado

                if Fxₖ₊₁ > Fx̂
                    xₖ₊₁ = x̂
                    x₋yₖ₊₁ = xₖ₊₁.-yₖ₊₁
                    nx₋yₖ₊₁ = x₋yₖ₊₁'x₋yₖ₊₁
                    sₖ₊₁ = xₖ₊₁.-xₖ
                    fyₖ₊₁ = f(yₖ₊₁)
                    fxₖ₊₁ = fx̂
                    Fxₖ₊₁ = Fx̂

                    nF_hist[k] += 1
                else
                    x₋yₖ₊₁ = sₖ₊₁ = xₖ₊₁.-xₖ
                    nx₋yₖ₊₁ = sₖ₊₁'sₖ₊₁
                    ∇fyₖ₊₁ = ∇fxₖ₊₁ 
                end                    
            end
        else
            xₖ, xₖ₊₁ = xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
            x₋yₖ₊₁ = sₖ₊₁ = xₖ₊₁.-xₖ
            nx₋yₖ₊₁ = sₖ₊₁'sₖ₊₁
            fyₖ₊₁, fxₖ₊₁ = fxₖ₊₁, f(xₖ₊₁)
            Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)
            ∇fyₖ₊₁ = ∇fxₖ₊₁

            pr₌gr₊1_hist[k] += 1
            T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ₊₁ que foi usado
        end

        if F_best > Fxₖ₊₁
            F_best = Fxₖ₊₁
        end

        T₁ = time()
        ∇fxₖ₊₁ = ∇f(xₖ₊₁)
        T∇ = T₁-time()

        F_hist[k+1] = Fxₖ₊₁
        ⎷nψₖ = norm(∇fyₖ₊₁.-∇fxₖ₊₁.+x₋yₖ₊₁./λₖ₊₁, p)
        crit_hist[k+1] = ⎷nψₖ
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
        nF_hist[k] = nF_hist[k-1]
        pr₌gr₊1_hist[k] = pr₌gr₊1_hist[k]
        
        nsₖ₊₁ = sₖ₊₁'sₖ₊₁
        aux = nx₋yₖ₊₁/(2*abs(fyₖ₊₁-fxₖ₊₁+∇fxₖ₊₁'x₋yₖ₊₁))
        if 1.0 > μ₀*aux/λₖ₊₁
            λₖ₊₁ = μ₁*aux
        else
            λₖ₊₁ += min(1, λₖ₊₁)*E(k)
        end
    end 

    return CompositeExecutionStats(
                    status = status,
                    problem = ProblemModel,
                    solver = SolverModel,
                    solution = xₖ,
                    objective = F_hist[k+1],
                    criticality = crit_hist[k+1],
                    total_iter = k,
                    elapsed_time = time()-T₀,
                    nF_hist = nF_hist[1:k+1],
                    pr_hist = pr₌gr₊1_hist[1:k+1],
                    gr_hist = pr₌gr₊1_hist[1:k+1].+(gr₊2-1),
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k+1]
                    )
end

function newAPG_vsopt(ProblemModel:: AbstractCompositeModel, SolverModel:: newAPG_vsModel) 
    @unpack_newAPG_vsModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, f, h, ∇f, prox = ProblemModel.Fopt, ProblemModel.fopt, ProblemModel.h, ProblemModel.∇fopt, ProblemModel.prox
    
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    Fxₖ, aux = F(x₀)
    F_hist[1] = Fxₖ
    crit_hist = Vector{Float64}(undef, kₘₐₓ+1)
    nF_hist = Vector{Int64}(undef, kₘₐₓ+1)
    nF_hist[1] = 2
    pr₌gr₊1_hist = Vector{Int64}(undef, kₘₐₓ+1)
    pr₌gr₊1_hist[1:2] .= 1
    gr₊2 = 0
    
    T₀ = time()
    if isnan(λ₁)
        λ₁ = (sqrt(length(x₀))*10^-5)/norm(∇f(x₀, aux).-ProblemModel.∇f(x₀.+10^-5))
        gr₊2 = 2

        @info "λ₁ = $(@sprintf "%.3e" λ₁)"
    end
    λₖ₊₁ = λ₁
    xₖ, xₖ₊₁ = x₀, prox(λₖ₊₁, x₀, zeros(length(x₀[1]))) # prox_λ₁h(x₀)
    tₖ₊₁ = 1.0
    fxₖ₊₁, aux = f(xₖ₊₁)
    
    T₁ = time()
    ∇fxₖ₊₁ = ∇f(xₖ₊₁, aux)
    T∇ = T₁-time() # Tempo descontado quando ∇f(xₖ₊₁) é calculado para definir convergência, mas não é usado pelo método

    crit_hist[1] = norm(∇fxₖ₊₁.+(xₖ.-xₖ₊₁)./λₖ₊₁, p)
    T₀ += time()-T₁ # Desconta o tempo entre T₁ e aqui

    Fxₖ₊₁ = fxₖ₊₁+h(xₖ₊₁)
    if Fxₖ > Fxₖ₊₁
        F_best = Fxₖ₊₁
    else
        F_best = Fxₖ
    end
    sₖ₊₁ = xₖ₊₁.-xₖ
    nsₖ₊₁ = sₖ₊₁'sₖ₊₁

    F_hist[2] = Fxₖ₊₁
    
    k = 2 # Considera o passo prox extra acima
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        if nsₖ₊₁ <= c*(Fxₖ-Fxₖ₊₁) 
            tₖ, tₖ₊₁ = tₖ₊₁, (1+sqrt(1+4*tₖ₊₁^2))/2
            yₖ₊₁ = xₖ.+((tₖ-1)/tₖ₊₁).*sₖ₊₁
            fyₖ₊₁, aux_y = f(yₖ₊₁)
            ∇fyₖ₊₁ = ∇f(yₖ₊₁, aux_y)
            x̂ = prox(λₖ₊₁, yₖ₊₁, ∇fyₖ₊₁)
            fx̂, aux_x = f(x̂)
            Fx̂ = fx̂+h(x̂)

            nF_hist[k] += 2
            pr₌gr₊1_hist[k] += 1

            if Fx̂ <= Fxₖ₊₁+min(Q(k), δ*(Fxₖ-Fxₖ₊₁))
                xₖ, xₖ₊₁ = xₖ₊₁, x̂
                x₋yₖ₊₁ = xₖ₊₁.-yₖ₊₁
                nx₋yₖ₊₁ = x₋yₖ₊₁'x₋yₖ₊₁
                sₖ₊₁ = xₖ₊₁.-xₖ
                fxₖ₊₁ = fx̂
                aux = aux_x
                Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, Fx̂
            else
                xₖ, xₖ₊₁ = xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
                fyₖ₊₁, (fxₖ₊₁, aux) = fxₖ₊₁, f(xₖ₊₁)
                Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)

                nF_hist[k] += 1
                pr₌gr₊1_hist[k] += 1
                T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ₊₁ que foi usado

                if Fxₖ₊₁ > Fx̂
                    xₖ₊₁ = x̂
                    x₋yₖ₊₁ = xₖ₊₁.-yₖ₊₁
                    nx₋yₖ₊₁ = x₋yₖ₊₁'x₋yₖ₊₁
                    sₖ₊₁ = xₖ₊₁.-xₖ
                    fxₖ₊₁ = fx̂
                    aux = aux_x
                    Fxₖ₊₁ = Fx̂
                else
                    x₋yₖ₊₁ = sₖ₊₁ = xₖ₊₁.-xₖ
                    nx₋yₖ₊₁ = sₖ₊₁'sₖ₊₁
                    ∇fyₖ₊₁ = ∇fxₖ₊₁ 
                end                    
            end
        else
            xₖ, xₖ₊₁ = xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
            x₋yₖ₊₁ = sₖ₊₁ = xₖ₊₁.-xₖ
            nx₋yₖ₊₁ = sₖ₊₁'sₖ₊₁
            fyₖ₊₁, (fxₖ₊₁, aux) = fxₖ₊₁, f(xₖ₊₁)
            Fxₖ, Fxₖ₊₁ = Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)
            ∇fyₖ₊₁ = ∇fxₖ₊₁

            pr₌gr₊1_hist[k] += 1
            T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ₊₁ que foi usado
        end

        if F_best > Fxₖ₊₁
            F_best = Fxₖ₊₁
        end

        T₁ = time()
        ∇fxₖ₊₁ = ∇f(xₖ₊₁, aux)
        T∇ = T₁-time()

        F_hist[k+1] = Fxₖ₊₁
        ⎷nψₖ = norm(∇fyₖ₊₁.-∇fxₖ₊₁.+x₋yₖ₊₁./λₖ₊₁, p)
        crit_hist[k+1] = ⎷nψₖ
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
        nF_hist[k] = nF_hist[k-1]
        pr₌gr₊1_hist[k] = pr₌gr₊1_hist[k]
        
        nsₖ₊₁ = sₖ₊₁'sₖ₊₁
        aux = nx₋yₖ₊₁/(2*abs(fyₖ₊₁-fxₖ₊₁+∇fxₖ₊₁'x₋yₖ₊₁))
        if 1.0 > μ₀*aux/λₖ₊₁
            λₖ₊₁ = μ₁*aux
        else
            λₖ₊₁ += min(1, λₖ₊₁)*E(k)
        end
    end 

    k = k-(kₘₐₓ == 1)
    return CompositeExecutionStats(
                    status = status,
                    problem = ProblemModel,
                    solver = SolverModel,
                    solution = xₖ,
                    objective = F_hist[k+1],
                    criticality = crit_hist[k+1],
                    total_iter = k,
                    elapsed_time = time()-T₀,
                    nF_hist = nF_hist[1:k+1],
                    pr_hist = pr₌gr₊1_hist[1:k+1],
                    gr_hist = pr₌gr₊1_hist[1:k+1].+(gr₊2-1),
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k+1]
                    )
end