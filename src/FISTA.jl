export Solver_FISTA

@with_kw struct Solver_FISTA
    method=:FISTA
    params_user
    
    params_default=[
        :x₀ => nothing,
        :L  => nothing
    ]

    params=append_params(params_default, params_user)
end

function FISTA(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, L:: Number, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    T₀=time()
    yₖ=xₖ₋₁=xₖ=x₀
    Lᵢₙᵥ=1/L
    tₖ=1.0
    
    k=1
    while true
        ∇fyₖ=∇f(yₖ)
        xₖ=prox(Lᵢₙᵥ, yₖ, ∇fyₖ) 

        T₁=time()
        ⎷nψₖ=norm(∇f(xₖ).-∇fyₖ.+(yₖ.-xₖ).*L, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ 
            return F(xₖ), time()-T₀, k, k
        elseif k==kₘₐₓ
            return F(xₖ), Inf, Inf, Inf
        end
        k+=1

        tₖ₋₁, tₖ=tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ=xₖ.+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁=xₖ
    end
end