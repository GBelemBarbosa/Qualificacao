export Problem_l2l0

# Funções para o operador proximal da norma ℓ₀
include("group_sparse_functions.jl")

# Struct do problema quadrático com regularização ℓ₀ 
@with_kw struct Problem_l2l0{M<:AbstractMatrix{<:Number}, v<:AbstractVector{<:Number}}
    A:: M
    b:: v
    
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