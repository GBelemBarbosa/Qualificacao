g(x:: Vector{<:Number}, group_indexs:: Vector{UnitRange{Int64}})=[@inbounds !iszero(x[group_indexs[i]]) for i=eachindex(group_indexs)]

g₀(x:: Vector{<:Number}; g=g)=sum(g(x))

Α(T:: Int64, group_indexs:: Vector{UnitRange{Int64}})=group_indexs[T]

Α(T:: Vector{Int64}, group_indexs:: Vector{UnitRange{Int64}})=reduce(vcat, @inbounds group_indexs[i] for i=T)

δCₛB(x:: Vector{<:Number}, s:: Int64; B=B)=0+(g₀(x)>s+!all(B(x)))*Inf

h(x:: Vector{<:Number}, λ:: Number; δCₛB=δCₛB, g₀=g₀)=λ*g₀(x)+δCₛB(x)

function UATPBTAT(x:: Vector{<:Number}, j:: Int64, n:: Int64, group_indexs:: Vector{UnitRange{Int64}}; A=A)
    y=zeros(Float64, n)
    y[Α(j, group_indexs)]=PDⱼ(x[Α(j, group_indexs)], j)

    return y
end

function UATPBTAT(x:: Vector{<:Number}, T:: Vector{Int64}, n:: Int64, group_indexs:: Vector{UnitRange{Int64}}; A=A)
    y=zeros(Float64, n)
    @inbounds for j=T
        y[Α(j, group_indexs)]=PDⱼ(x[Α(j, group_indexs)], j)
    end

    return y
end

function UATPBTAT(x:: Vector{<:Number}, j:: Int64, n:: Int64)
    y=zeros(Float64, n)
    y[j]=x[j]

    return y
end

function UATPBTAT(x:: Vector{<:Number}, T:: Vector{Int64}, n:: Int64)
    y=zeros(Float64, n)
    @inbounds for j=T
        y[j]=x[j]
    end

    return y
end

PBTAT(x:: Vector{<:Number}, j:: Int64, group_indexs:: Vector{UnitRange{Int64}}; A=A)=PDⱼ(x[Α(j, group_indexs)], j)

PBTAT(x:: Vector{<:Number}, T:: Vector{Int64}, group_indexs:: Vector{UnitRange{Int64}}; A=A)=reduce(vcat, @inbounds PDⱼ(x[Α(j, group_indexs)], j) for j=T)

ω(x:: Vector{<:Number}, dDⱼ:: Function, group_indexs:: Vector{UnitRange{Int64}}; A=A)=[@inbounds norm(x[Α(j, group_indexs)])^2-dDⱼ(x[Α(j, group_indexs)], j)^2 for j=eachindex(group_indexs)]

ω(x:: Vector{<:Number})=x.*x

ωₛ(x:: Vector{<:Number}, s=Inf64; ω=ω)=partialsort(ω(x), s, rev=true)

function Sₛ(x:: Vector{<:Number}, s=Inf64; ω=ω)
    ωx=ω(x)
    uωx=unique(ωx)
    uωxₛ=partialsort(uωx, min(s, length(uωx)), rev=true)

    return findall(x -> x>=uωxₛ, ωx)
end

I₁(x:: Vector{<:Number}, m:: Int64, group_indexs:: Vector{UnitRange{Int64}})=findall(!iszero, g(x, group_indexs))

I₀(x:: Vector{<:Number}, m:: Int64, group_indexs:: Vector{UnitRange{Int64}})=findall(iszero, g(x, group_indexs))

function I₊(x:: Vector{<:Number}, dDⱼ:: Function, m:: Int64, group_indexs:: Vector{UnitRange{Int64}}, s:: Int64, λ:: Int64)
    ωx=ω(x, dDⱼ, group_indexs)
    ωxₛ=partialsort(ωx, s, rev=true)

    return findall(x->x>max(ωxₛ, 2*λ), ωx)
end

function Iq(x:: Vector{<:Number}, dDⱼ:: Function, s:: Int64, λ:: Int64)
    ωx=ω(x, dDⱼ, group_indexs)
    ωxₛ=partialsort(ωx, s, rev=true)

    return findall(isequal(max(ωxₛ, 2*λ)), ωx)
end

T(ωx:: Vector{<:Number}, s:: Int64)=partialsortperm(ωx, 1:s, rev=true)

T(ωx:: Vector{<:Number})=sortperm(ωx, rev=true)

function proxhL(L:: Number, x:: Vector{<:Number}, s:: Int64, λ:: Number, n:: Int64, group_indexs:: Vector{UnitRange{Int64}}; T=T, ω=ω, UATPBTAT=UATPBTAT)
    ωx=ω(x)
    Tωx=T(ωx, s)

    return UATPBTAT(x, Tωx[1:searchsortedlast(ωx[Tωx], 2*λ/L, rev=true, lt=<=)], n, group_indexs)
end

function proxhL(L:: Number, x:: Vector{<:Number}, s:: Int64, λ:: Number, n:: Int64; T=T, ω=ω, UATPBTAT=UATPBTAT)
    ωx=ω(x)
    Tωx=T(ωx, s)

    return UATPBTAT(x, Tωx[1:searchsortedlast(ωx[Tωx], 2*λ/L, rev=true, lt=<=)], n)
end

proxhL(L:: Number, x:: Vector{<:Number}, λ:: Number, ω:: Function)=[@inbounds x[i]*(ω(x)[i]>2*λ/L) for i=eachindex(x)]

proxhL(L:: Number, x:: Vector{<:Number}, λ:: Number)=[@inbounds x[i]*(abs(x[i])>sqrt(2*λ/L)) for i=eachindex(x)]

proxhL1(L:: Number, x:: Vector{<:Number}, λ:: Number)=[max(x[i]-λ/L, 0)*sign(x[i]) for i=eachindex(x)]

TL(L:: Number, x:: Vector{<:Number}; ∇f=∇f)=x.-∇f(x)./L