using LinearAlgebra 
# Para o cálculo de L=λₘₐₓ(ATA)
using KrylovKit
# Structs com campos pre-definidos
using Parameters: @with_kw

# Função para substituir parâmetros default por escolhas do usuário
function append_params(params_default, params_user)
    if !(typeof(params_user)<:Pair)
        for i=eachindex(params_default)
            for param ∈ params_user
                if params_default[i][1] == param[1]
                    params_default[i]=param
                end
            end
        end

        return params_default
    elseif typeof(params_user)<:Pair
        index=findfirst(param -> param[1]==params_user[1], params_default)
        
        if isnothing(index)
            return params_default
        else
            params_default[index] = params_user
            
            return params_default
        end
    else
        return params_default
    end
end

# Funções para o operador proximal da norma ℓ₀
include("group_sparse_functions.jl")

# Algoritmos
include("PG.jl")
include("FISTA.jl")
include("NSPG.jl")
include("ANSPG.jl")
include("nmAPGLS.jl")
include("newAPG_vs.jl")

# Cria struct Solver do tipo correto para o algoritmo
Solver(method:: Symbol; params=()) = eval(Symbol("Solver_"*String(method)))(params_user=params)

# Struct do problema quadrático com regularização ℓ₀ 
@with_kw struct Problem
    A
    b
    
    m:: Int64 = size(A, 1)
    n:: Int64 = size(A, 2)

    ATAx:: Function = (x:: Array{<:Number}; A=A) -> A'*(A*x)

    # Constantes do problema SCi
    L:: Number  = 1.01*real(eigsolve(ATAx, n, 1, :LM, eltype(A))[1][1]) # Margem de segurança
    λ:: Number  = 0.1*norm(A'b, Inf)^2/(2*L) # Pode ser qualquer c*norm(A'b, Inf)^2/(2*L), onde 0<c<1
    
    x₀:: Array{<:Number} = zeros(Float64, n) # Escolha clássica para soluções esparsas

    # Funções do problema SCi
    f:: Function = (x:: Array{<:Number}; A=A, b=b)  -> norm(A*x.-b)^2/2
    h:: Function = (x:: Array{<:Number}; λ=λ)       -> λ*norm(x, 0)
    F:: Function = (x:: Array{<:Number}; f=f, h=h)  -> f(x)+h(x)
    ∇f:: Function = (x:: Array{<:Number}; A=A, b=b) -> A'*(A*x.-b)

    # Operador proximal para todos os métodos
    prox:: Function = (αₖ:: Number, x:: Array{<:Number}, ∇fx:: Array{<:Number}; proxhL=proxhL, λ=λ) -> proxhL(1/αₖ, x.-αₖ.*∇fx, λ)
end

function solve(problem:: Problem, solver, ϵ:: Number, kₘₐₓ:: Int64)
    # Para acessar facilmente os parâmetros
    d = Dict(solver.params)

    # Verifica se usuário especificou parâmetro x₀ ou se é usado a escolha default (origem nesse caso)
    if isnothing(d[:x₀])
        x₀ = problem.x₀
    else
        x₀ = d[:x₀]
    end

    # Identifica o algoritmo do solver
    if solver.method==:NSPG
        # Verifica se usuário especificou parâmetro γ₀ ou se é usado a aproximação default
        if isnothing(d[:γ₀])
            γ₀ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))
        else
            γ₀ = d[:γ₀]
        end
        println("γ₀ = ", γ₀)

        # Executa o algoritmo no problema com os parâmetros selecionados
        return NSPG(problem.F, problem.∇f, problem.prox, x₀, γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
    elseif solver.method==:ANSPG
        if isnothing(d[:γ₀])
            γ₀ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))

            if isnothing(d[:α₀])
                α₀ = γ₀
            else
                α₀ = d[:α₀]
            end
        else
            γ₀ = d[:γ₀]
        end
        if isnothing(d[:α₀])
            α₀ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))
        end
        println("γ₀, α₀ = ", γ₀, ", ", α₀)

        return ANSPG(problem.F, problem.∇f, problem.prox, x₀, α₀, d[:ρ], d[:β], d[:αₘᵢₙ,], d[:αₘₐₓ], Int(d[:n]), γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
    elseif solver.method==:PG || solver.method==:FISTA
        if isnothing(d[:L])
            L = problem.L
        end

        return eval(solver.method)(problem.F, problem.∇f, problem.prox, x₀, L, kₘₐₓ; ϵ=ϵ)
    elseif solver.method==:nmAPGLS
        if isnothing(d[:α₀])
            α₀ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))
        end
        println("α₀ = ", α₀)

        return nmAPGLS(problem.F, problem.∇f, problem.prox, x₀, α₀, d[:ρ], d[:η], d[:δ], kₘₐₓ; ϵ=ϵ)
    else
        if isnothing(d[:λ₁])
            λ₁ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))
        else
            λ₁ = d[:λ₁]
        end
        println("λ₁ = ", λ₁)

        return newAPG_vs(problem.f, problem.h, problem.∇f, problem.prox, d[:Q], d[:E], x₀, λ₁, d[:μ₀], d[:μ₁], d[:c], d[:δ], kₘₐₓ; ϵ=ϵ)
    end

    return eval(solver.method)(x)
end     