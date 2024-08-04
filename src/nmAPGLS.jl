export Solver_nmAPGLS

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
    
    T₀=time()
    vₖ=yₖ₋₁=yₖ=zₖ=vₗₐₛₜ=xₖ₋₁=xₖ=x₀
    F_best=Fvₖ=Fzₖ=Fxₖ=cₖ=F(xₖ)
    ∇fyₖ₋₁=∇fyₖ=∇fvₗₐₛₜ=∇fxₖ=∇f(xₖ)
    αₗₐₛₜ=αyₖ=α₀
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
                        return F_best, Inf, Inf, Inf
                    end
                end
                if Fzₖ>Fvₖ
                    vₗₐₛₜ, xₖ=xₖ, vₖ
                    ∇fvₗₐₛₜ=∇fxₖ
                    αₗₐₛₜ=αxₖ
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
                return F_best, Inf, Inf, Inf
            end
        end

        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        ∇fxₖ=∇f(xₖ)
        Tᵥ=T₁-time()

        ⎷nψₖ=norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₗₐₛₜ, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ
            return F_best, time()-T₀, pr, gr
        elseif k==kₘₐₓ
            return F_best, Inf, Inf, Inf
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
end

function nmAPGLSopt(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, α₀:: Number, ρ:: Number, η:: Number, δ:: Number, kₘₐₓ:: Int64; ϵ=eps(), p=Inf) 
    pr=0
    gr=1
    Tᵥ=0.0 # Possible time spent on ∇f(xₖ) (doesn't happen each iteration)
    
    T₀=time()
    vₖ=yₖ₋₁=yₖ=zₖ=vₗₐₛₜ=xₖ₋₁=xₖ=x₀
    Fxₖ, aux=F(xₖ)
    F_best=Fyₖ=Fvₖ=Fzₖ=cₖ=Fxₖ
    aux_z=aux_y=aux_v=aux
    ∇fyₖ₋₁=∇fyₖ=∇fvₗₐₛₜ=∇fxₖ=∇f(xₖ, aux)
    αₗₐₛₜ=αyₖ=α₀
    qₖ=tₖ=1.0 
    
    k=1
    while true
        while true
            zₖ=prox(αyₖ, yₖ, ∇fyₖ)

            pr+=1

            Fzₖ, aux_z=F(zₖ)
            nzₖ₋yₖ=norm(zₖ.-yₖ)^2

            if Fzₖ+δ*nzₖ₋yₖ<=cₖ
                vₗₐₛₜ=yₖ
                ∇fvₗₐₛₜ=∇fyₖ
                αₗₐₛₜ=αyₖ
                xₖ=zₖ
                Fxₖ=Fzₖ
                aux=aux_z

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

                    Fvₖ, aux_v=F(vₖ)

                    if Fvₖ+δ*norm(vₖ.-xₖ)^2<=cₖ
                        break
                    end

                    αxₖ*=ρ

                    if isnan(αxₖ) || αxₖ<10^-17
                        return F_best, Inf, Inf, Inf
                    end
                end
                if Fzₖ>Fvₖ
                    vₗₐₛₜ, xₖ=xₖ, vₖ
                    ∇fvₗₐₛₜ=∇fxₖ
                    αₗₐₛₜ=αxₖ
                    Fxₖ=Fvₖ
                    aux=aux_v
                else
                    vₗₐₛₜ=yₖ
                    ∇fvₗₐₛₜ=∇fyₖ
                    αₗₐₛₜ=αyₖ
                    xₖ=zₖ
                    Fxₖ=Fzₖ
                    aux=aux_z
                end

                break
            end

            αyₖ*=ρ

            if isnan(αyₖ) || αyₖ<10^-17
                return F_best, Inf, Inf, Inf
            end
        end

        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        ∇fxₖ=∇f(xₖ, aux)
        Tᵥ=T₁-time()

        ⎷nψₖ=norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₗₐₛₜ, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ
            return F_best, time()-T₀, pr, gr
        elseif k==kₘₐₓ
            return F_best, Inf, Inf, Inf
        end
        k+=1
        gr+=1
        
        tₖ₋₁, tₖ=tₖ, (1+sqrt(1+4*tₖ^2))/2
        qₖ₋₁, qₖ=qₖ, η*qₖ+1
        cₖ=(η*qₖ₋₁*cₖ+Fxₖ)/qₖ
        yₖ₋₁, yₖ=yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        xₖ₋₁=xₖ
        Fyₖ, aux_y=F(yₖ)
        ∇fyₖ₋₁, ∇fyₖ=∇fyₖ, ∇f(yₖ, aux_y)
        syₖ=yₖ.-yₖ₋₁
        αyₖ=syₖ'syₖ/(syₖ'*(∇fyₖ.-∇fyₖ₋₁))
    end 
end