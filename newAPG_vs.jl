@with_kw struct Solver_newAPG_vs
    method=:newAPG_vs
    params_user

    Q:: Function = k:: Int64 -> 0.99^k
    E:: Function = k:: Int64 -> k^-1.1
    params_default=[
        :x₀ => nothing,

        :Q  => Q,
        :E  => E,
        :λ₁ => nothing,
        :μ₀ => 0.99, 
        :μ₁ => 0.95, 
        :c  => 10^4,
        :δ  => 0.8
    ]

    params=append_params(params_default, params_user)
end

function newAPG_vs(f:: Function, h:: Function, ∇f:: Function, prox:: Function, Q:: Function, E:: Function, x₀:: Array{<:Number}, λ₁:: Number, μ₀:: Number, μ₁:: Number, c:: Number, δ:: Number, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr₌gr=0
    Tᵥ=0.0 # Possible time spent on ∇f(xₖ₊₁) (doesn't happen each iteration)
    flag=false
    Fxₖ=f(x₀)
    Tₜ=T₀=time()
    λₖ₊₁=λ₁
    tₖ₊₁=1.0
    xₖ, xₖ₊₁=x₀, prox(λ₁, x₀, 0 .*x₀) # prox_λ₁h(x₀)
    ∇fxₖ₊₁=∇f(xₖ₊₁)
    fxₖ₊₁=f(xₖ₊₁)
    Fxₖ₊₁=fxₖ₊₁+h(xₖ₊₁)
    if Fxₖ>Fxₖ₊₁
        F_best=Fxₖ₊₁
    else
        F_best=Fxₖ
    end
    sₖ₊₁=xₖ₊₁.-xₖ
    nsₖ₊₁=sₖ₊₁'sₖ₊₁
    
    k=2 # Account for the extra prox step above
    while true
        if nsₖ₊₁<=c*(Fxₖ-Fxₖ₊₁) 
            tₖ, tₖ₊₁=tₖ₊₁, (1+sqrt(1+4*tₖ₊₁^2))/2
            yₖ₊₁=xₖ.+((tₖ-1)/tₖ₊₁).*sₖ₊₁
            ∇fyₖ₊₁=∇f(yₖ₊₁)
            x̂=prox(λₖ₊₁, yₖ₊₁, ∇fyₖ₊₁)
            fx̂=f(x̂)
            Fx̂=fx̂+h(x̂)

            pr₌gr+=1

            if Fx̂<=Fxₖ₊₁+min(Q(k), δ*(Fxₖ-Fxₖ₊₁))
                xₖ, xₖ₊₁=xₖ₊₁, x̂
                x₋yₖ₊₁=xₖ₊₁.-yₖ₊₁
                nx₋yₖ₊₁=x₋yₖ₊₁'x₋yₖ₊₁
                sₖ₊₁=xₖ₊₁.-xₖ
                fyₖ₊₁=f(yₖ₊₁)
                fxₖ₊₁=fx̂
                Fxₖ, Fxₖ₊₁=Fxₖ₊₁, Fx̂
            else
                xₖ, xₖ₊₁=xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
                fyₖ₊₁, fxₖ₊₁=fxₖ₊₁, f(xₖ₊₁)
                Fxₖ, Fxₖ₊₁=Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)

                pr₌gr+=1
                T₀+=Tᵥ

                if Fxₖ₊₁>Fx̂
                    fyₖ₊₁=f(yₖ₊₁)
                    xₖ₊₁=x̂
                    x₋yₖ₊₁=xₖ₊₁.-yₖ₊₁
                    nx₋yₖ₊₁=x₋yₖ₊₁'x₋yₖ₊₁
                    sₖ₊₁=xₖ₊₁.-xₖ
                    fxₖ₊₁=fx̂
                    Fxₖ₊₁=Fx̂
                else
                    x₋yₖ₊₁=sₖ₊₁=xₖ₊₁.-xₖ
                    nx₋yₖ₊₁=sₖ₊₁'sₖ₊₁
                    ∇fyₖ₊₁=∇fxₖ₊₁ 
                end                    
            end
        else
            xₖ, xₖ₊₁=xₖ₊₁, prox(λₖ₊₁, xₖ₊₁, ∇fxₖ₊₁)
            x₋yₖ₊₁=sₖ₊₁=xₖ₊₁.-xₖ
            nx₋yₖ₊₁=sₖ₊₁'sₖ₊₁
            fyₖ₊₁, fxₖ₊₁=fxₖ₊₁, f(xₖ₊₁)
            Fxₖ, Fxₖ₊₁=Fxₖ₊₁, fxₖ₊₁+h(xₖ₊₁)
            ∇fyₖ₊₁=∇fxₖ₊₁

            pr₌gr+=1
            T₀+=Tᵥ
        end

        if F_best>Fxₖ₊₁
            F_best=Fxₖ₊₁
        end

        T₁=time()
        ∇fxₖ₊₁=∇f(xₖ₊₁)
        Tᵥ=T₁-time()

        Tₜ=T₁-T₀
        ⎷nψₖ=norm(∇fyₖ₊₁.-∇fxₖ₊₁.+x₋yₖ₊₁./λₖ₊₁, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ || k==kₘₐₓ
            if ⎷nψₖ<ϵ
                flag=true
            end

            break
        end
        k+=1
        
        nsₖ₊₁=sₖ₊₁'sₖ₊₁
        aux=nx₋yₖ₊₁/(2*abs(fyₖ₊₁-fxₖ₊₁+∇fxₖ₊₁'x₋yₖ₊₁))
        if 1.0>μ₀*aux/λₖ₊₁
            λₖ₊₁=μ₁*aux
        else
            λₖ₊₁+=min(1, λₖ₊₁)*E(k)
        end
    end 

    converg=!flag*Inf
    return F_best, Tₜ+converg, pr₌gr+1+converg, pr₌gr+converg # Account for the extra first prox step
end