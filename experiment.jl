using Plots
# Escrita matemática em plots
using LaTeXStrings
# Leitura e escrita de arquivos
using MAT
using FileIO, JLD2
# Plots de performance profile
using BenchmarkProfiles
# Para o formato dos problemas compilados
using SparseArrays

include("solver.jl")

# Para adicionar um novo Solver na lista abaixo, use 
# Solver(:nome_metodo, params=(:nome_parametro_1 => valor_1, :nome_parametro_2 => valor_2))
# params é opcional, cada método tem seus valores default. No exemplo acima, somente 
# nome_parametro_1 e nome_parametro_2 recebem valores do usuário, ode demais continuam default.
# Os nomes dos paramêtros de cada método no geral seguem a terminologia de seus artigos base.
# Veja o arquivo .jl referente a cada algoritmo para obter seus nomes e valores default. 
solvers = [Solver(:PG), Solver(:FISTA), Solver(:nmAPGLS), Solver(:newAPG_vs), Solver(:NSPG), Solver(:ANSPG)]

# Arrays contendo os dados de cada método
s = length(solvers)

T_hist  = Array{Float64}(undef, 25, s)
F_hist  = Array{Float64}(undef, 25, s)
pr_hist = Array{Float64}(undef, 25, s)
gr_hist = Array{Float64}(undef, 25, s)

# Dowload dos problemas pode ser feito em: 
# https://drive.google.com/drive/folders/1rbccpgxR_R4hfaauzl0hIKVLvEHfTbx4?usp=sharing
# (tamanho ≈ 1.5 GB)
# Sua localização dos dados
lassodir = "path/Data-Lasso/"

for i=1:25
    # Leitura dos arquivos dos problemas
    problem  = "SC"*string(i)
    vars     = matread(lassodir*problem*".mat")

    println(problem*":")

    A, b, λ1 = vars["A"], vec(vars["b"]), vars["lambda"] # λ1 da norma l1, descartável 

    # Problem struct define algumas variáveis e funções do problema quadrático
    problem = Problem(A=A, b=b) # Gera as funções e variáveis genéricas do problema quadrático ℓ₀
    ϵ       = problem.L*10^-5   # Critério de parada
    kₘₐₓ    = 5000

    println("λ, L, ϵ: ", problem.λ, ", ", problem.L, ", ", ϵ)

    # Solução com cada solver armazenada para comparação
    for j=eachindex(solvers)
        # solve executa o solver escolhido no problema, retornando algumas informações de desempenho
        F_hist[i, j], T_hist[i, j], pr_hist[i, j], gr_hist[i, j] = solve(problem, solvers[j], ϵ, kₘₐₓ)
        println(solvers[j].method, ":\nF_best, Tₜ, prox, ∇f = ", F_hist[i, j], ", ", T_hist[i, j], ", ", pr_hist[i, j], ", ", gr_hist[i, j])
    end
end

# Plots
names=[] # Nome dos métodos para os Plots
pltpr = performance_profile(PlotsBackend(), pr_hist, names)
plot!(pltpr, dpi=600, legend=:bottomright, title="Performance profile of prox calculations")

pltgr = performance_profile(PlotsBackend(), gr_hist, names)
plot!(pltgr, dpi=600, legend=:bottomright, title="Performance profile of gradient evaluations")

pltT = performance_profile(PlotsBackend(), T_hist, names)
plot!(pltT, dpi=600, legend=:bottomright, title="Performance profile of convergence time")

pltF = performance_profile(PlotsBackend(), F_hist, names)
plot!(pltF, dpi=600, legend=:bottomright, title="Performance profile of best function value")