module CompositeOptim
export solve, Solver, Problem

using LinearAlgebra 
# Para o cálculo de L=λₘₐₓ(ATA)
using KrylovKit
# Structs com campos pre-definidos
using Parameters: @with_kw
# Análise de function specification
using MethodAnalysis

include("l2l0.jl")

# Problem(class:: Symbol; variables=()) = eval(Symbol("Solver_"*string(class)))(variables=variables)

# Função para substituir parâmetros default por escolhas do usuário
function append_params(params_default, params_user)
    if typeof(params_user)<:Tuple{Vararg{<:Pair{Symbol, <:Any}}} 
        for i=eachindex(params_default)
            for param ∈ params_user
                if params_default[i][1] == param[1]
                    params_default[i]=param
                end
            end
        end

        return params_default
    elseif typeof(params_user)<:Pair{Symbol, <:Any}
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

# Algoritmos
include("PG.jl")
include("FISTA.jl")
include("NSPG.jl")
include("ANSPG.jl")
include("nmAPGLS.jl")
include("newAPG_vs.jl")

# Cria struct Solver do tipo correto para o algoritmo
Solver(method:: Symbol; params=()) = eval(Symbol("Solver_"*string(method)))(params_user=params)

function solve(problem, solver, ϵ:: Number, kₘₐₓ:: Int64)
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
        if problem.optmizable
            return NSPGopt(problem.Fopt, problem.∇fopt, problem.prox, x₀, γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
        else 
            return NSPG(problem.F, problem.∇f, problem.prox, x₀, γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
        end
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

        if problem.optmizable
            return ANSPGopt(problem.Fopt, problem.∇fopt, problem.prox, x₀, α₀, d[:ρ], d[:β], d[:αₘᵢₙ,], d[:αₘₐₓ], Int(d[:n]), γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
        else 
            return ANSPG(problem.F, problem.∇f, problem.prox, x₀, α₀, d[:ρ], d[:β], d[:αₘᵢₙ,], d[:αₘₐₓ], Int(d[:n]), γ₀, d[:τ], d[:δ], d[:γₘᵢₙ], d[:γₘₐₓ], Int(d[:m]), kₘₐₓ; ϵ=ϵ)
        end
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
        if problem.optmizable
            return nmAPGLSopt(problem.Fopt, problem.∇fopt, problem.prox, x₀, α₀, d[:ρ], d[:η], d[:δ], kₘₐₓ; ϵ=ϵ)
        else
            return nmAPGLS(problem.F, problem.∇f, problem.prox, x₀, α₀, d[:ρ], d[:η], d[:δ], kₘₐₓ; ϵ=ϵ)
        end
     else
        if isnothing(d[:λ₁])
            λ₁ = (sqrt(problem.n)*10^-5)/norm(problem.∇f(x₀).-problem.∇f(x₀.+10^-5))
        else
            λ₁ = d[:λ₁]
        end
        println("λ₁ = ", λ₁)

        if problem.optmizable
            return newAPG_vsopt(problem.fopt, problem.h, problem.∇fopt, problem.prox, d[:Q], d[:E], x₀, λ₁, d[:μ₀], d[:μ₁], d[:c], d[:δ], kₘₐₓ; ϵ=ϵ)
        else
            return newAPG_vs(problem.f, problem.h, problem.∇f, problem.prox, d[:Q], d[:E], x₀, λ₁, d[:μ₀], d[:μ₁], d[:c], d[:δ], kₘₐₓ; ϵ=ϵ)
        end
    end
end     

end