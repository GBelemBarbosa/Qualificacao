export ANSPGModel

@with_kw mutable struct ANSPGModel <: AbstractCompositeSolver
    method = :ANSPG
    
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()

    ϵ:: Number     = eps()
    p:: Number     = Inf
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing

    α₀:: Number  = NaN
    n:: Int64    = 5
    ρ:: Float64  = 0.25
    β:: Float64  = 0.01

    αₘᵢₙ:: Number = eps()
    αₘₐₓ:: Number = typemax(Int64)

    m:: Int64   = 5
    τ:: Number  = 0.25
    δ:: Number  = 0.01

    γₘᵢₙ:: Number = eps()
    γₘₐₓ:: Number = typemax(Int64)
end

function ANSPG(ProblemModel:: AbstractCompositeModel, SolverModel:: ANSPGModel)
    @unpack_ANSPGModel SolverModel
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
    F_best = Fvₖ = Fzₖ = Fxₖ = F(xₖ)
    ∇fyₖ₋₁ = ∇fyₖ = ∇fvₗₐₛₜ = ∇fxₖ = ∇f(xₖ)
    if isnan(α₀)
        α₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-∇f(x₀.+10^-5))
        gr_hist[1] += 1
        
        @info "α₀ = $(@sprintf "%.3e" α₀)"
    end
    αₒᵣγₗₐₛₜ = αₖ = α₀
    nzₖᵢ₋yₖ = tₖ = 1.0  
    last_yₙ = [Fxₖ for i = 1:n]
    last_xₘ = [Fxₖ for i = 1:m]
    
    F_hist[1] = Fxₖ

    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        Fyₗ₍ₖ₎ = maximum(last_yₙ)

        while true
            zₖ = prox(αₖ, yₖ, ∇fyₖ)
            Fzₖ = F(zₖ)
            nzₖᵢ₋yₖ = norm(zₖ.-yₖ)^2

            pr_hist[k] += 1
            nF_hist[k+1] += 1

            if Fzₖ+β*nzₖᵢ₋yₖ/(2*αₖ) <= Fyₗ₍ₖ₎ 
                break
            end

            αₖ *= ρ

            if isnan(αₖ) || αₖ < αₘᵢₙ
                @warn "Busca linear resultou em isnan(αₖ) || αₖ < αₘᵢₙ" isnan(αₖ) αₖ < αₘᵢₙ
                status = :exception
                
                break
            end
        end

        Fxₗ₍ₖ₎ = maximum(last_xₘ)

        if Fzₖ+β*nzₖᵢ₋yₖ/(2*αₖ) <= Fxₗ₍ₖ₎
            vₗₐₛₜ = yₖ
            xₖ = zₖ
            Fxₖ = Fzₖ
            ∇fvₗₐₛₜ = ∇fyₖ
            αₒᵣγₗₐₛₜ = αₖ
        else
            sxₖ = xₖ.-yₖ₋₁
            nsxₖ = sxₖ'sxₖ
            rxₖ = ∇fxₖ.-∇fyₖ₋₁
            γₖ = nsxₖ/(sxₖ'rxₖ)
            if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
                γₖ = sqrt(nsxₖ/(rxₖ'rxₖ))
            end

            gr_hist[k] += 1
            T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ que foi usado

            while true
                vₖ = prox(γₖ, xₖ, ∇fxₖ)
                Fvₖ = F(vₖ)

                pr_hist[k] += 1
                nF_hist[k+1] += 1

                if Fvₖ+δ*norm(vₖ.-xₖ)^2/(2*γₖ) <= Fxₗ₍ₖ₎
                    break
                end
            
                γₖ *= τ

                if isnan(γₖ) || γₖ < γₘᵢₙ
                @warn "Busca linear resultou em isnan(γₖ) || γₖ < γₘᵢₙ" isnan(γₖ) γₖ < γₘᵢₙ
                status = :exception
                
                break
            end
            end
            
            if Fvₖ < Fzₖ
                vₗₐₛₜ, xₖ = xₖ, vₖ 
                Fxₖ = Fvₖ
                ∇fvₗₐₛₜ = ∇fxₖ
                αₒᵣγₗₐₛₜ = γₖ
            else
                vₗₐₛₜ = yₖ
                xₖ = zₖ
                Fxₖ = Fzₖ 
                ∇fvₗₐₛₜ = ∇fyₖ
                αₒᵣγₗₐₛₜ = αₖ
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
        ⎷nψₖ = norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₒᵣγₗₐₛₜ, p)
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
        
        popfirst!(last_xₘ)
        push!(last_xₘ, Fxₖ)
        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ₋₁, yₖ = yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        popfirst!(last_yₙ)
        push!(last_yₙ, F(yₖ))
        xₖ₋₁ = xₖ
        syₖ = yₖ.-yₖ₋₁
        nsyₖ = syₖ'syₖ
        ∇fyₖ₋₁, ∇fyₖ = ∇fyₖ, ∇f(yₖ)
        ryₖ = ∇fyₖ.-∇fyₖ₋₁
        αₖ = nsyₖ/(syₖ'ryₖ)
        if αₖ > αₘₐₓ || αₖ < αₘᵢₙ
            αₖ = sqrt(nsyₖ/(ryₖ'ryₖ))
        end
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

function ANSPGopt(ProblemModel:: AbstractCompositeModel, SolverModel:: ANSPGModel)
    @unpack_ANSPGModel SolverModel
    if isnothing(x₀)
        x₀ = ProblemModel.x₀
    end
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
    F_best = Fvₖ = Fzₖ = Fxₖ
    aux_z = aux_y = aux_v = aux
    ∇fyₖ₋₁ = ∇fyₖ = ∇fvₗₐₛₜ = ∇fxₖ = ∇f(xₖ, aux)
    if isnan(α₀)
        α₀ = (sqrt(length(x₀))*10^-5)/norm(∇fxₖ.-ProblemModel.∇f(x₀.+10^-5))
        gr_hist[1] += 1
        
        @info "α₀ = $(@sprintf "%.3e" α₀)"
    end
    αₒᵣγₗₐₛₜ = αₖ = α₀
    nzₖᵢ₋yₖ = tₖ = 1.0  
    last_yₙ = [Fxₖ for i = 1:n]
    last_xₘ = [Fxₖ for i = 1:m]
    
    F_hist[1] = Fxₖ

    k = 1
    status = k > kₘₐₓ ? :max_iter : time()-T₀ >= Tₘₐₓ ? :max_time : :running 
    while status == :running
        Fyₗ₍ₖ₎ = maximum(last_yₙ)

        while true
            zₖ = prox(αₖ, yₖ, ∇fyₖ)
            Fzₖ, aux_z = F(zₖ)
            nzₖᵢ₋yₖ = norm(zₖ.-yₖ)^2

            pr_hist[k] += 1
            nF_hist[k+1] += 1

            if Fzₖ+β*nzₖᵢ₋yₖ/(2*αₖ) <= Fyₗ₍ₖ₎ 
                break
            end

            αₖ *= ρ

            if isnan(αₖ) || αₖ < αₘᵢₙ
                @warn "Busca linear resultou em isnan(αₖ) || αₖ < αₘᵢₙ" isnan(αₖ) αₖ < αₘᵢₙ
                status = :exception
                
                break
            end
        end

        Fxₗ₍ₖ₎ = maximum(last_xₘ)

        if Fzₖ+β*nzₖᵢ₋yₖ/(2*αₖ) <= Fxₗ₍ₖ₎
            vₗₐₛₜ = yₖ
            xₖ = zₖ
            aux = aux_z
            Fxₖ = Fzₖ
            ∇fvₗₐₛₜ = ∇fyₖ
            αₒᵣγₗₐₛₜ = αₖ
        else
            sxₖ = xₖ.-yₖ₋₁
            nsxₖ = sxₖ'sxₖ
            rxₖ = ∇fxₖ.-∇fyₖ₋₁
            γₖ = nsxₖ/(sxₖ'rxₖ)
            if γₖ > γₘₐₓ || γₖ < γₘᵢₙ
                γₖ = sqrt(nsxₖ/(rxₖ'rxₖ))
            end

            gr_hist[k] += 1
            T₀ += T∇ # Acrescenta o tempo do cálculo ∇fxₖ que foi usado

            while true
                vₖ = prox(γₖ, xₖ, ∇fxₖ)
                Fvₖ, aux_v = F(vₖ)

                pr_hist[k] += 1
                nF_hist[k+1] += 1

                if Fvₖ+δ*norm(vₖ.-xₖ)^2/(2*γₖ) <= Fxₗ₍ₖ₎
                    break
                end
            
                γₖ *= τ

                if isnan(γₖ) || γₖ < γₘᵢₙ
                @warn "Busca linear resultou em isnan(γₖ) || γₖ < γₘᵢₙ" isnan(γₖ) γₖ < γₘᵢₙ
                status = :exception
                
                break
            end
            end
            
            if Fvₖ < Fzₖ
                vₗₐₛₜ, xₖ = xₖ, vₖ 
                aux = aux_v
                Fxₖ = Fvₖ
                ∇fvₗₐₛₜ = ∇fxₖ
                αₒᵣγₗₐₛₜ = γₖ
            else
                vₗₐₛₜ = yₖ
                xₖ = zₖ
                aux = aux_z
                Fxₖ = Fzₖ 
                ∇fvₗₐₛₜ = ∇fyₖ
                αₒᵣγₗₐₛₜ = αₖ
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
        ⎷nψₖ = norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₒᵣγₗₐₛₜ, p)
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
        
        popfirst!(last_xₘ)
        push!(last_xₘ, Fxₖ)
        tₖ₋₁, tₖ = tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ₋₁, yₖ = yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        popfirst!(last_yₙ)
        Fyₖ, aux_y = F(yₖ)
        push!(last_yₙ, Fyₖ)
        xₖ₋₁ = xₖ
        syₖ = yₖ.-yₖ₋₁
        nsyₖ = syₖ'syₖ
        ∇fyₖ₋₁, ∇fyₖ = ∇fyₖ, ∇f(yₖ, aux_y)
        ryₖ = ∇fyₖ.-∇fyₖ₋₁
        αₖ = nsyₖ/(syₖ'ryₖ)
        if αₖ > αₘₐₓ || αₖ < αₘᵢₙ
            αₖ = sqrt(nsyₖ/(ryₖ'ryₖ))
        end
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