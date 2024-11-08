export l2MCPModel

# Struct do problema quadrático com regularização MCP
@with_kw struct l2MCPModel{M<: AbstractMatrix{<:Number}, v<: AbstractVector{<:Number}} <: AbstractCompositeModel
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    A:: M
    b:: v

    ATAx:: Function = (x:: Array{<:Number}; A = A) -> A'*(A*x)

    x₀:: Array{<:Number} = zeros(Float64, size(A, 2)) # Escolha clássica para soluções esparsas

    L:: Number   = 1.01*real(eigsolve(ATAx, length(x₀), 1, :LM, eltype(A))[1][1]) # Margem de segurança
    aux:: Number = norm(A'b, Inf)
    λ₁:: Number  = 0.1*aux # Para esse x₀, pode ser qualquer c*norm(A'b, Inf), em que 0<c<1
    λ₀:: Number  = 0.1*aux^2/(2*L) # Para esse x₀, pode ser qualquer c*norm(A'b, Inf)^2/(2*L), em que 0<c<1
    α:: Number   = 2*λ₀/λ₁ # Constante do α-envelope subtraído de ℓ₁
        
    optimizable:: Bool = true

    # Funções do problema SCi
    f:: Function  = (x:: Array{<:Number}; A = A, b = b)   -> norm(A*x.-b)^2/2
    h:: Function  = (x:: Array{<:Number}; λ₁ = λ₁, α = α) -> reduce(+, [abs(i)≥α ? α*λ₁/2 : λ₁*(abs(i)-i^2/(2*α)) for i = x])
    F:: Function  = (x:: Array{<:Number}; f = f, h = h)   -> f(x)+h(x)
    ∇f:: Function = (x:: Array{<:Number}; A = A, b = b)  -> A'*(A*x.-b)

    fopt:: Function  = (x:: Array{<:Number}; A = A, b = b)       -> begin aux = A*x; norm(aux.-b)^2/2, aux end
    Fopt:: Function  = (x:: Array{<:Number}; fopt = fopt, h = h) -> begin fx, aux = fopt(x); fx+h(x), aux end
    ∇fopt:: Function = (x:: Array{<:Number}, aux; A = A, b = b) -> A'*(aux.-b)

    prox:: Function = (αₖ:: Number, x, ∇fx:: Array{<:Number}; proxhL = proxhL_MCP, λ₁ = λ₁, α = α) -> if typeof(x)<:Array{<:Number}; proxhL(1/αₖ, x.-αₖ.*∇fx, λ₁, α); else; proxhL(1/αₖ, x[1].-αₖ.*∇fx, λ₁, α), x[2] end
end