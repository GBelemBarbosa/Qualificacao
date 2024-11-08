export AbstractCompositeModel, CompositeModel

abstract type AbstractCompositeModel end

#=
Struct do problema compósito 

  min_x F(x) = f(x) + h(x), com h(x) lsc
=#
@with_kw struct CompositeModel <: AbstractCompositeModel
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    x₀:: Any = nothing
    
    f:: Function    = () -> nothing
    h:: Function    = () -> nothing
    F:: Function    = () -> nothing
    ∇f:: Function   = () -> nothing
    prox:: Function = () -> nothing

    # Alguns métodos podem se beneficiar de versões otimizadas das funções do problema, economizando cálculos
    optimizable:: Bool

    #= 
    Versões otimizadas das funções caso optimizable == true que guardam uma variável aux para as demais usarem
    
    Repare que a ordem de execução de cada função nos métodos é fixa e igual a essa abaixo, portanto mantenha
    essa ordem na hora de construir as funções (calcular aux em Fopt para depois usar em fopt causará erros, por exemplo)

    Veja l2l0.jl para exemplos de como essas funções devem ser construídas
    =#
    fopt:: Function  = () -> nothing
    Fopt:: Function  = () -> nothing
    ∇fopt:: Function = () -> nothing
end