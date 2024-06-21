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
        :γₘₐₓ => 10^-30,
    ]

    params=append_params(params_default, params_user)
end

function NSPG(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, γ₀:: Number, τ:: Number, δ:: Number, γₘᵢₙ:: Number, γₘₐₓ:: Number, m:: Int64, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr=0
    gr=1
    flag=false
    Tₜ=T₀=time()
    xₖ₋₁=xₖ=x₀
    F_best=Fxₖ=F(x₀)
    ∇fxₖ₋₁=∇fxₖ=∇f(xₖ)
    nsₖ=γₖ=γ₀
    sₖ=zeros(Float64, length(xₖ))
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

            if Fxₖ+γₖ*δ*nsₖ/2<=Fxₗ₍ₖ₎ 
                break
            end

            γₖ*=τ

            if γₖ<γₘᵢₙ || isnan(γₖ)
                return F_best, Tₜ, pr, gr
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

        if ⎷nψₖ<ϵ || k==kₘₐₓ
            if ⎷nψₖ<ϵ
                flag=true
            end

            break
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

    converg=!flag*Inf
    return F_best, Tₜ+converg, pr+converg, gr+converg
end