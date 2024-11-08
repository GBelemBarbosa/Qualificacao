export AbstractCompositeSolver, SolverModel

abstract type AbstractCompositeSolver end

#=
Struct do solver para o problema compósito 

  min_x F(x) = f(x) + h(x), com h(x) lsc
=#
@with_kw struct SolverModel <: AbstractCompositeSolver
    method:: Symbol = :nothing
      
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()

    ϵ:: Number     = eps()
    kₘₐₓ:: Int64   = typemax(Int64)
    Tₘₐₓ:: Float64 = Inf
    x₀             = nothing
end