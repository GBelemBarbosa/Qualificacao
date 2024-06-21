@with_kw struct Solver_PG
    method=:PG
    params_user
    
    params_default=[
        :x₀ => nothing,
        :L  => nothing
    ]

    params=append_params(params_default, params_user)
end

function PG(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, L:: Number, kₘₐₓ:: Int64; ϵ=eps(), p=Inf)
    flag=false
    Tₜ=T₀=time()
    xₖ₋₁=xₖ=x₀
    ∇fxₖ₋₁=∇fxₖ=∇f(xₖ)
    Lᵢₙᵥ=1/L
    
    k=1
    while true
        xₖ=prox(Lᵢₙᵥ, xₖ, ∇fxₖ) 

        T₁=time()
        Tₜ=T₁-T₀
        ⎷nψₖ=norm(∇fxₖ.-∇fxₖ₋₁.+(xₖ₋₁.-xₖ).*L, p)+Inf*(k==1)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ || k==kₘₐₓ
            if ⎷nψₖ<ϵ
                flag=true
            end

            break
        end
        k+=1
        
        xₖ₋₁=xₖ
        ∇fxₖ₋₁, ∇fxₖ=∇fxₖ, ∇f(xₖ)
    end 

    converg=!flag*Inf
    return F(xₖ), Tₜ+converg, k+converg, k+converg
end