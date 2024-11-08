export l2l0Model

# Struct do problema quadrático com regularização ℓ₀ 
@with_kw struct l2l0Model{M<: AbstractMatrix{<:Number}, v<: AbstractVector{<:Number}} <: AbstractCompositeModel
    params:: Dict{Symbol, Any} = Dict{Symbol, Any}()
    
    A:: M
    b:: v

    ATAx:: Function = (x:: Array{<:Number}; A = A) -> A'*(A*x)

    x₀:: Array{<:Number} = zeros(Float64, size(A, 2)) # Escolha clássica para soluções esparsas

    L:: Number  = 1.01*real(eigsolve(ATAx, length(x₀), 1, :LM, eltype(A))[1][1]) # Margem de segurança
    λ:: Number  = 0.1*norm(A'b, Inf)^2/(2*L) # Para esse x₀, pode ser qualquer c*norm(A'b, Inf)^2/(2*L), em que 0<c<1
        
    f:: Function    = (x:: Array{<:Number}; A = A, b = b)  -> norm(A*x.-b)^2/2
    h:: Function    = (x:: Array{<:Number}; λ = λ)       -> λ*norm(x, 0)
    F:: Function    = (x:: Array{<:Number}; f = f, h = h)  -> f(x)+h(x)
    ∇f:: Function   = (x:: Array{<:Number}; A = A, b = b) -> A'*(A*x.-b)
    prox:: Function = (αₖ:: Number, x, ∇fx:: Array{<:Number}; proxhL = proxhL_l0, λ = λ) -> if typeof(x)<:Array{<:Number}; proxhL(1/αₖ, x.-αₖ.*∇fx, λ); else; proxhL(1/αₖ, x[1].-αₖ.*∇fx, λ), x[2] end

    optimizable:: Bool = true

    fopt:: Function  = (x:: Array{<:Number}; A = A, b = b)       -> begin aux = A*x; norm(aux.-b)^2/2, aux end
    Fopt:: Function  = (x:: Array{<:Number}; fopt = fopt, h = h) -> begin fx, aux = fopt(x); fx+h(x), aux end
    ∇fopt:: Function = (x:: Array{<:Number}, aux; A = A, b = b) -> A'*(aux.-b)
end