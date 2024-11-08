module CompositeOptim
export solve_composite, composite_solver, composite_model, CompositeExecutionStats

using LinearAlgebra 
# Para o cálculo de L = λₘₐₓ(ATA)
using KrylovKit
# Formatação de números
using Printf
# Structs com campos pre-definidos
using Parameters: @with_kw
# Análise de function specification
# using MethodAnalysis

# Funções para o operador proximal da norma ℓ₀, ℓ₁ e MCP
include("group_sparse_functions.jl")

# Structs e tipos abstratos de problemas e solvers
include("AbstractCompositeModel.jl")
include("AbstractSolverModel.jl")

@with_kw struct CompositeExecutionStats
    status:: Symbol = :nothing

    problem:: AbstractCompositeModel = CompositeModel(optimizable = false)
    solver:: AbstractCompositeSolver = SolverModel()

    solution              = nothing
    objective:: Number    = NaN
    criticality:: Number  = NaN # ψₖ da última iteração  
    total_iter:: Int64    = typemax(Int64)
    elapsed_time:: Number = NaN
    
    nF_hist:: Array{Int64}      = Int64[] # Histórico de avaliações de F
    pr_hist:: Array{Int64}      = Int64[] # Histórico de avaliações do prox
    gr_hist:: Array{Int64}      = Int64[] # Histórico de avaliações ∇f 
    F_hist:: Array{<:Number}    = Number[] # Histórico de F
    crit_hist:: Array{<:Number} = Number[] # Histórico de ψₖ 
end

# Modelos de problemas
include("l2l0.jl")
include("l2l1.jl")
include("l2MCP.jl")

# Cria struct Solver do tipo correto para o algoritmo
composite_model(class:: Symbol; params...) = eval(Symbol(string(class)*"Model"))(params = Dict(params); params...)

# Algoritmos
include("PG.jl")
include("FISTA.jl")
include("NSPG.jl")
include("NHSPG.jl")
include("ANSPG.jl")
include("ANHSPG.jl")
include("nmAPGLS.jl")
include("newAPG_vs.jl")

# Cria struct Model do tipo correto para o algoritmo
composite_solver(method:: Symbol; params...) = eval(Symbol(string(method)*"Model"))(params = Dict(params); params...)

#=
Struct do problema compósito 

  min_x F(x) = f(x) + h(x), com h(x) lsc

Input:
  ProblemModel: 
  SolverModel: 
Output:
  x: melhor/último iterando
  his: function history
  feval: number of function evals (total objective)
=#
function solve_composite(ProblemModel:: AbstractCompositeModel, SolverModel:: AbstractCompositeSolver)
    opt_str = ProblemModel.optimizable ? "opt" : ""

    # Executa o algoritmo no problema com os parâmetros selecionados
    return eval(Meta.parse(string(SolverModel.method)*opt_str))(ProblemModel, SolverModel)
end     

end