export nmAPGLSModel

@with_kw mutable struct nmAPGLSModel <: AbstractCompositeSolver
    method = :nmAPGLS
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()

    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    α₀:: Number = NaN
    ρ:: Float64 = 2/5 
    η:: Float64 = 0.8 
    δ:: Float64 = 10^-4
end

function nmAPGLS(ProblemModel:: AbstractCompositeModel, SolverModel:: nmAPGLSModel)
    @unpack_nmAPGLSModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.F, ProblemModel.∇f, ProblemModel.prox

    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)
    nF_hist = Vector{Int64}(undef, kₘₐₓ+1)
    nF_hist[1:2] .= 1
    pr_hist = Vector{Int64}(undef, kₘₐₓ)
    pr_hist[1] = 0
    gr_hist = Vector{Int64}(undef, kₘₐₓ)
    gr_hist[1] = 1
    T∇ = 0.0 # Tempo descontado quando ∇f(xₖ) é calculado para definir convergência, mas não é usado pelo método 
    
    T₀ = time()
    x_best = vₖ = yₖ₋₁ = yₖ = zₖ = vₗₐₛₜ = xₖ₋₁ = xₖ = x₀
    F_best = Fvₖ = Fzₖ = Fxₖ = cₖ = F(xₖ)
    ∇fyₖ₋₁ = ∇fyₖ = ∇fvₗₐₛₜ = ∇fxₖ = ∇f(xₖ)
    if isnan(α₀)
        α₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-∇f(x₀.+10^-5))
        gr_hist[1] += 1
        
        @info "α₀ = $(@sprintf "%.3e" α₀)"
    end
    αₗₐₛₜ = αyₖ = α₀
    qₖ = tₖ = 1.0 
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        Fyₖ = F(yₖ)

        while true
            zₖ = prox(αyₖ, yₖ, ∇fyₖ)
            Fzₖ = F(zₖ)
            nzₖ₋yₖ = norm(zₖ.-yₖ)^2

            pr_hist[k] += 1
            nF_hist[k+1] += 1

            if Fzₖ+δ*nzₖ₋yₖ <= cₖ
                vₗₐₛₜ = yₖ
                ∇fvₗₐₛₜ = ∇fyₖ
                αₗₐₛₜ = αyₖ
                xₖ = zₖ
                Fxₖ = Fzₖ

                break
            elseif Fzₖ+δ*nzₖ₋yₖ <= Fyₖ
                sxₖ = xₖ.-yₖ₋₁
                αxₖ = sxₖ'sxₖ/(sxₖ'*(∇fxₖ.-∇fyₖ₋₁))
                
                gr_hist[k] += 1
                T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ que foi usado

                while true
                    vₖ = prox(αxₖ, xₖ, ∇fxₖ)
                    Fvₖ = F(vₖ)

                    pr_hist[k] += 1
                    nF_hist[k+1] += 1

                    if Fvₖ+δ*norm(vₖ.-xₖ)^2 <= cₖ
                        break
                    end

                    αxₖ *= ρ

                    if isnan(αxₖ) || αxₖ < 10^-17
                        status = :exception
                        break
                    end
                end
                if Fzₖ > Fvₖ
                    vₗₐₛₜ, xₖ = xₖ, vₖ
                    ∇fvₗₐₛₜ = ∇fxₖ
                    αₗₐₛₜ = αxₖ
                    Fxₖ = Fvₖ
                else
                    vₗₐₛₜ = yₖ
                    ∇fvₗₐₛₜ = ∇fyₖ
                    αₗₐₛₜ = αyₖ
                    xₖ = zₖ
                    Fxₖ = Fzₖ
                end

                break
            end

            αyₖ *= ρ

            if isnan(αyₖ) || αyₖ < 10^-17
                status = :exception
                
                break
            end
        end

        if F_best > Fxₖ
            F_best = Fxₖ
            x_best = xₖ
        end

        T₁ = time()
        ∇fxₖ = ∇f(xₖ)
        T∇ = T₁-time()

        F_hist[k+1] = Fxₖ
        ⎷nψₖ = norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₗₐₛₜ, p)
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
        nF_hist[k+1] = nF_hist[k]+1
        pr_hist[k] = pr_hist[k-1]
        gr_hist[k] = gr_hist[k-1]+1
        
        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        qₖ₋₁, qₖ = qₖ, η*qₖ+1
        cₖ = (η*qₖ₋₁*cₖ+Fxₖ)/qₖ
        yₖ₋₁, yₖ = yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁ = xₖ
        syₖ = yₖ.-yₖ₋₁
        ∇fyₖ₋₁, ∇fyₖ = ∇fyₖ, ∇f(yₖ)
        αyₖ = syₖ'syₖ/(syₖ'*(∇fyₖ.-∇fyₖ₋₁))
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
                    nF_hist = nF_hist[1:k+1],
                    pr_hist = pr_hist[1:k],
                    gr_hist = gr_hist[1:k],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end

function nmAPGLSopt(ProblemModel:: AbstractCompositeModel, SolverModel:: nmAPGLSModel) 
    @unpack_nmAPGLSModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
    F, ∇f, prox = ProblemModel.Fopt, ProblemModel.∇fopt, ProblemModel.prox
    
    F_hist = Vector{Float64}(undef, kₘₐₓ+1)
    crit_hist = Vector{Float64}(undef, kₘₐₓ)
    nF_hist = Vector{Int64}(undef, kₘₐₓ+1)
    nF_hist[1:2] .= 1
    pr_hist = Vector{Int64}(undef, kₘₐₓ)
    pr_hist[1] = 0
    gr_hist = Vector{Int64}(undef, kₘₐₓ)
    gr_hist[1] = 1
    T∇ = 0.0 # Tempo descontado quando ∇f(xₖ) é calculado para definir convergência, mas não é usado pelo método 
    
    T₀ = time()
    x_best = vₖ = yₖ₋₁ = yₖ = zₖ = vₗₐₛₜ = xₖ₋₁ = xₖ = x₀
    Fxₖ, aux = F(xₖ)
    F_best = Fyₖ = Fvₖ = Fzₖ = cₖ = Fxₖ
    aux_z = aux_y = aux_v = aux
    ∇fyₖ₋₁ = ∇fyₖ = ∇fvₗₐₛₜ = ∇fxₖ = ∇f(xₖ, aux)
    if isnan(α₀)
        α₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-ProblemModel.∇f(x₀.+10^-5))
        gr_hist[1] += 1
        
        @info "α₀ = $(@sprintf "%.3e" α₀)"
    end
    αₗₐₛₜ = αyₖ = α₀
    qₖ = tₖ = 1.0 
    
    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        while true
            zₖ = prox(αyₖ, yₖ, ∇fyₖ)
            Fzₖ, aux_z = F(zₖ)
            nzₖ₋yₖ = norm(zₖ.-yₖ)^2

            pr_hist[k] += 1
            nF_hist[k+1] += 1

            if Fzₖ+δ*nzₖ₋yₖ <= cₖ
                vₗₐₛₜ = yₖ
                ∇fvₗₐₛₜ = ∇fyₖ
                αₗₐₛₜ = αyₖ
                xₖ = zₖ
                Fxₖ = Fzₖ
                aux = aux_z

                break
            elseif Fzₖ+δ*nzₖ₋yₖ <= Fyₖ
                sxₖ = xₖ.-yₖ₋₁
                αxₖ = sxₖ'sxₖ/(sxₖ'*(∇fxₖ.-∇fyₖ₋₁))
                
                gr_hist[k] += 1
                T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ que foi usado

                while true
                    vₖ = prox(αxₖ, xₖ, ∇fxₖ)
                    Fvₖ, aux_v = F(vₖ)

                    pr_hist[k] += 1
                    nF_hist[k+1] += 1

                    if Fvₖ+δ*norm(vₖ.-xₖ)^2 <= cₖ
                        break
                    end

                    αxₖ *= ρ

                    if isnan(αxₖ) || αxₖ < 10^-17
                        status = :exception
                        break
                    end
                end
                if Fzₖ > Fvₖ
                    vₗₐₛₜ, xₖ = xₖ, vₖ
                    ∇fvₗₐₛₜ = ∇fxₖ
                    αₗₐₛₜ = αxₖ
                    Fxₖ = Fvₖ
                    aux = aux_v
                else
                    vₗₐₛₜ = yₖ
                    ∇fvₗₐₛₜ = ∇fyₖ
                    αₗₐₛₜ = αyₖ
                    xₖ = zₖ
                    Fxₖ = Fzₖ
                    aux = aux_z
                end

                break
            end

            αyₖ *= ρ

            if isnan(αyₖ) || αyₖ < 10^-17
                status = :exception
                
                break
            end
        end

        if F_best > Fxₖ
            F_best = Fxₖ
            x_best = xₖ
        end

        T₁ = time()
        ∇fxₖ = ∇f(xₖ, aux)
        T∇ = T₁-time()

        F_hist[k+1] = Fxₖ
        ⎷nψₖ = norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₗₐₛₜ, p)
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
        nF_hist[k+1] = nF_hist[k]+1
        pr_hist[k] = pr_hist[k-1]
        gr_hist[k] = gr_hist[k-1]+1
        
        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        qₖ₋₁, qₖ = qₖ, η*qₖ+1
        cₖ = (η*qₖ₋₁*cₖ+Fxₖ)/qₖ
        yₖ₋₁, yₖ = yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁ = xₖ
        Fyₖ, aux_y = F(yₖ)
        ∇fyₖ₋₁, ∇fyₖ = ∇fyₖ, ∇f(yₖ, aux_y)
        syₖ = yₖ.-yₖ₋₁
        αyₖ = syₖ'syₖ/(syₖ'*(∇fyₖ.-∇fyₖ₋₁))
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
                    nF_hist = nF_hist[1:k+1],
                    pr_hist = pr_hist[1:k],
                    gr_hist = gr_hist[1:k],
                    F_hist = F_hist[1:k+1],
                    crit_hist = crit_hist[1:k]
                    )
end