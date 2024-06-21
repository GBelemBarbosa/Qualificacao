@with_kw struct Solver_ANSPG
    method=:ANSPG
    params_user

    params_default=[
        :x₀ => nothing,

        :α₀ => nothing,
        :n  => 2,
        :ρ  => 0.5,
        :β  => 0.01,

        :αₘᵢₙ => 10^-30,
        :αₘₐₓ => 10^-30,

        :γ₀ => nothing,
        :m  => 5,
        :τ  => 0.5,
        :δ  => 0.01,

        :γₘᵢₙ => 10^-30,
        :γₘₐₓ => 10^-30
    ]

    params=append_params(params_default, params_user)
end

function ANSPG(F:: Function, ∇f:: Function, prox:: Function, x₀:: Array{<:Number}, α₀:: Number, ρ:: Number, β:: Number, αₘᵢₙ:: Number, αₘₐₓ:: Number, n:: Int64, γ₀:: Number, τ:: Number, δ:: Number, γₘᵢₙ:: Number, γₘₐₓ:: Number, m:: Int64, kₘₐₓ:: Int64; ϵ=eps(), p=Inf)
    pr=0
    gr=1
    Tᵥ=0.0 # Possible time spent on ∇f(xₖ) (doesn't happen each iteration)
    flag=false
    Tₜ=T₀=time()
    vₖ=yₖ₋₁=yₖ=zₖ=vₗₐₛₜ=xₖ₋₁=xₖ=x₀
    αₒᵣγₗₐₛₜ=αₖ=α₀
    F_best=Fvₖ=Fzₖ=Fxₖ=F(xₖ)
    ∇fyₖ₋₁=∇fyₖ=∇fvₗₐₛₜ=∇fxₖ=∇f(xₖ)
    last_yₙ=[Fxₖ for i=1:n]
    last_xₘ=[Fxₖ for i=1:m]
    nzₖᵢ₋yₖ=tₖ=1.0 
    
    k=1
    while true
        Fyₗ₍ₖ₎=maximum(last_yₙ)

        while true
            zₖ=prox(αₖ, yₖ, ∇fyₖ)

            pr+=1

            Fzₖ=F(zₖ)
            nzₖᵢ₋yₖ=norm(zₖ.-yₖ)^2

            if Fzₖ+β*αₖ*nzₖᵢ₋yₖ<=Fyₗ₍ₖ₎ 
                break
            end

            αₖ*=ρ

            if isnan(αₖ) || αₖ<αₘᵢₙ
                return F_best, Tₜ+Inf, pr+Inf, gr+Inf
            end
        end

        Fxₗ₍ₖ₎=maximum(last_xₘ)

        if Fzₖ+β*αₖ*nzₖᵢ₋yₖ<=Fxₗ₍ₖ₎
            vₗₐₛₜ=yₖ
            xₖ=zₖ
            Fxₖ=Fzₖ
            ∇fvₗₐₛₜ=∇fyₖ
            αₒᵣγₗₐₛₜ=αₖ
        else
            if k>1
                sxₖ=xₖ.-yₖ₋₁
                nsxₖ=sxₖ'sxₖ
                rxₖ=∇fxₖ.-∇fyₖ₋₁
                γₖ=nsxₖ/(sxₖ'rxₖ)
                if γₖ>γₘₐₓ || γₖ<γₘᵢₙ
                    γₖ=sqrt(nsxₖ/(rxₖ'rxₖ))
                end

                gr+=1
                T₀+=Tᵥ
            else
                γₖ=γ₀
            end

            while true
                vₖ=prox(γₖ, xₖ, ∇fxₖ)

                pr+=1

                Fvₖ=F(vₖ)

                if Fvₖ+δ*γₖ*norm(vₖ.-xₖ)^2<=Fxₗ₍ₖ₎
                    break
                end
            
                γₖ*=τ

                if isnan(γₖ) || γₖ<γₘᵢₙ
                    return F_best, Tₜ+Inf, pr+Inf, gr+Inf
                end
            end
            
            if Fvₖ<Fzₖ
                vₗₐₛₜ, xₖ=xₖ, vₖ 
                Fxₖ=Fvₖ
                ∇fvₗₐₛₜ=∇fxₖ
                αₒᵣγₗₐₛₜ=γₖ
            else
                vₗₐₛₜ=yₖ
                xₖ=zₖ
                Fxₖ=Fzₖ 
                ∇fvₗₐₛₜ=∇fyₖ
                αₒᵣγₗₐₛₜ=αₖ
            end
        end
        
        if F_best>Fxₖ
            F_best=Fxₖ
        end

        T₁=time()
        ∇fxₖ=∇f(xₖ)
        Tᵥ=T₁-time()

        Tₜ=T₁-T₀
        ⎷nψₖ=norm(∇fxₖ.-∇fvₗₐₛₜ.+(vₗₐₛₜ.-xₖ)./αₒᵣγₗₐₛₜ, p)
        T₀+=time()-T₁

        if ⎷nψₖ<ϵ || k==kₘₐₓ
            if ⎷nψₖ<ϵ
                flag=true
            end

            break
        end
        k+=1
        gr+=1
        
        popfirst!(last_xₘ)
        push!(last_xₘ, Fxₖ)
        tₖ₋₁, tₖ=tₖ, (1+sqrt(1+4*tₖ^2))/2
        yₖ₋₁, yₖ=yₖ, xₖ.+(tₖ₋₁/tₖ).*(zₖ.-xₖ).+((tₖ₋₁-1)/tₖ).*(xₖ.-xₖ₋₁)
        popfirst!(last_yₙ)
        push!(last_yₙ, F(yₖ))
        xₖ₋₁=xₖ
        syₖ=yₖ.-yₖ₋₁
        nsyₖ=syₖ'syₖ
        ∇fyₖ₋₁, ∇fyₖ=∇fyₖ, ∇f(yₖ)
        ryₖ=∇fyₖ.-∇fyₖ₋₁
        αₖ=nsyₖ/(syₖ'ryₖ)
        if αₖ>αₘₐₓ || αₖ<αₘᵢₙ
            αₖ=sqrt(nsyₖ/(ryₖ'ryₖ))
        end
    end 

    converg=!flag*Inf
    return F_best, Tₜ+converg, pr+converg, gr+converg
end