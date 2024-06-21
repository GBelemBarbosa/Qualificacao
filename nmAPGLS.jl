@with_kw struct Solver_nmAPGLS
    method=:nmAPGLS
    params_user

    params_default=[
        :x₀ => nothing,

        :α₀ => nothing,
        :ρ  => 2/5, 
        :η  => 0.8, 
        :δ  => 10^-4
    ]
    
    params=append_params(params_default, params_user)
end

function nmAPGLS(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, α₀:: Number, ρ:: Number, η:: Number, δ:: Number, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr=0
    gr=1
    Tᵥ=0.0 # Possible time spent on ∇f(xₖ) (doesn't happen each iteration)
    flag=false
    Tₜ=T₀=time()
    vₖ=yₖ₋₁=yₖ=zₖ=vₗₐₛₜ=xₖ₋₁=xₖ=x₀
    αₗₐₛₜ=αyₖ=α₀
    F_best=Fvₖ=Fzₖ=Fxₖ=cₖ=F(xₖ)
    ∇fyₖ₋₁=∇fyₖ=∇fvₗₐₛₜ=∇fxₖ=∇f(xₖ)
    qₖ=tₖ=1.0 
    
    k=1
    while true
        Fyₖ=F(yₖ)

        while true
            zₖ=prox(αyₖ, yₖ, ∇fyₖ)

            pr+=1

            Fzₖ=F(zₖ)
            nzₖ₋yₖ=norm(zₖ.-yₖ)^2

            if Fzₖ+δ*nzₖ₋yₖ<=cₖ
                vₗₐₛₜ=yₖ
                ∇fvₗₐₛₜ=∇fyₖ
                αₗₐₛₜ=αyₖ
                xₖ=zₖ
                Fxₖ=Fzₖ

                break
            elseif Fzₖ+δ*nzₖ₋yₖ<=Fyₖ
                if k>1
                    sxₖ=xₖ.-yₖ₋₁
                    αxₖ=sxₖ'sxₖ/(sxₖ'*(∇fxₖ.-∇fyₖ₋₁))
                    
                    gr+=1
                    T₀+=Tᵥ
                else
                    αxₖ=α₀
                end

                while true
                    vₖ=prox(αxₖ, xₖ, ∇fxₖ)

                    pr+=1

                    Fvₖ=F(vₖ)

                    if Fvₖ+δ*norm(vₖ.-xₖ)^2<=cₖ
                        break
                    end

                    αxₖ*=ρ

                    if isnan(αxₖ) || αxₖ<10^-17
                        return F_best, Tₜ, pr, gr
                    end
                end
                if Fzₖ>Fvₖ
                    vₗₐₛₜ, xₖ=xₖ, vₖ
                    ∇fvₗₐₛₜ=∇fxₖ
                    αₗₐₛₜ=αxₖ
                    xₖ=vₖ
                    Fxₖ=Fvₖ
                else
                    vₗₐₛₜ=yₖ
                    ∇fvₗₐₛₜ=∇fyₖ
                    αₗₐₛₜ=αyₖ
                    xₖ=zₖ
                    Fxₖ=Fzₖ
                end

                break
            end

            αyₖ*=ρ

            if isnan(αyₖ) || αyₖ<10^-17
                return F_best, Tₜ, pr, gr
            end
        end

        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        ∇fxₖ=∇f(xₖ)
        Tᵥ=T₁-time()

        Tₜ=T₁-T₀
        ⎷nψₖ=norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₗₐₛₜ, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ || k==kₘₐₓ
            if ⎷nψₖ<ϵ
                flag=true
            end

            break
        end
        k+=1
        gr+=1
        
        tₖ₋₁, tₖ=tₖ, (1+sqrt(1+4*tₖ^2))/2
        qₖ₋₁, qₖ=qₖ, η*qₖ+1
        cₖ=(η*qₖ₋₁*cₖ+Fxₖ)/qₖ
        yₖ₋₁, yₖ=yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁=xₖ
        syₖ=yₖ.-yₖ₋₁
        ∇fyₖ₋₁, ∇fyₖ=∇fyₖ, ∇f(yₖ)
        αyₖ=syₖ'syₖ/(syₖ'*(∇fyₖ.-∇fyₖ₋₁))
    end 

    converg=!flag*Inf
    return F_best, Tₜ+converg, pr+converg, gr+converg
end