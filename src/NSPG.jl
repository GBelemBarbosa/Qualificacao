export Solver_NSPG

@with_kw struct Solver_NSPG
    method=:NSPG
    params_user

    params_default=[
        :x₀ => nothing,

        :γ₀ => nothing,
        :m  => 5,
        :τ  => 0.5,
        :δ  => 0.01,
        
        :γₘᵢₙ => 10^-30,
        :γₘₐₓ => 10^30,
    ]

    params=append_params(params_default, params_user)
end

function NSPG(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, γ₀:: Number, τ:: Number, δ:: Number, γₘᵢₙ:: Number, γₘₐₓ:: Number, m:: Int64, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr=0
    gr=1
    
    Tₜ=T₀=time()
    sₖ=xₖ₋₁=xₖ=x₀
    F_best=Fxₖ=F(x₀)
    ∇fxₖ₋₁=∇fxₖ=∇f(xₖ)
    nsₖ=γₖ=γ₀
    
    lastₘ=[Fxₖ for i=1:m]
    
    k=1
    while true
        Fxₗ₍ₖ₎=maximum(lastₘ)

        while true
            xₖ=prox(γₖ, xₖ₋₁, ∇fxₖ)

            pr+=1

            Fxₖ=F(xₖ)
            sₖ=xₖ.-xₖ₋₁
            nsₖ=sₖ'sₖ

            if Fxₖ+δ*nsₖ/(2*γₖ)<=Fxₗ₍ₖ₎ 
                break
            end

            γₖ*=τ

            if γₖ<γₘᵢₙ || isnan(γₖ)
                return F_best, Inf, Inf, Inf
            end
        end
        ∇fxₖ₋₁, ∇fxₖ=∇fxₖ, ∇f(xₖ)

        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        Tₜ=T₁-T₀
        ⎷nψₖ=norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ)./γₖ, p)+Inf*(k==1)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ
            return F_best, time()-T₀, pr, gr
        elseif k==kₘₐₓ
            return F_best, Inf, Inf, Inf
        end
        k+=1
        gr+=1

        popfirst!(lastₘ)
        push!(lastₘ, Fxₖ)
        xₖ₋₁=xₖ
        rₖ=∇fxₖ.-∇fxₖ₋₁
        γₖ=nsₖ/(sₖ'rₖ)
        if γₖ>γₘₐₓ || γₖ<γₘᵢₙ
            γₖ=sqrt(nsₖ/(rₖ'rₖ))
        end
    end 
end

function NSPGopt(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, γ₀:: Number, τ:: Number, δ:: Number, γₘᵢₙ:: Number, γₘₐₓ:: Number, m:: Int64, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr=0
    gr=1

    T₀=time()
    xₖ₋₁=sₖ=xₖ=x₀
    Fxₖ, aux=F(xₖ)
    F_best=Fxₖ
    ∇fxₖ₋₁=∇fxₖ=∇f(xₖ, aux)
    nsₖ=γₖ=γ₀

    lastₘ=[Fxₖ for i=1:m]
    
    k=1
    while true
        Fxₗ₍ₖ₎=maximum(lastₘ)

        while true
            xₖ=prox(γₖ, xₖ₋₁, ∇fxₖ)

            pr+=1

            Fxₖ, aux=F(xₖ)
            sₖ=xₖ.-xₖ₋₁
            nsₖ=sₖ'sₖ

            if Fxₖ+δ*nsₖ/(2*γₖ)<=Fxₗ₍ₖ₎ 
                break
            end

            γₖ*=τ

            if γₖ<γₘᵢₙ || isnan(γₖ)
                return F_best, Inf, Inf, Inf
            end
        end
        ∇fxₖ₋₁, ∇fxₖ=∇fxₖ, ∇f(xₖ, aux)

        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        ⎷nψₖ=norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ)./γₖ, p)+Inf*(k==1)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ
            return F_best, time()-T₀, pr, gr
        elseif k==kₘₐₓ
            return F_best, Inf, Inf, Inf
        end
        k+=1
        gr+=1

        popfirst!(lastₘ)
        push!(lastₘ, Fxₖ)
        xₖ₋₁=xₖ
        rₖ=∇fxₖ.-∇fxₖ₋₁
        γₖ=nsₖ/(sₖ'rₖ)
        if γₖ>γₘₐₓ || γₖ<γₘᵢₙ
            γₖ=sqrt(nsₖ/(rₖ'rₖ))
        end
    end 
end