using Plots
# Escrita matemática em plots
using LaTeXStrings
# Leitura e escrita de arquivos
using MAT # Para ler os problemas do Data-Lasso 
# using JLD2, UnPack # Para ler e escrever dados e plots dos experimentos (opicional)
# Plots de performance profile
using BenchmarkProfiles
# Struct das variáveis dos problemas do Data-Lasso
using SparseArrays
# Para mudar a seed
import Random

# Structs e solvers (rode o .jl previamente)
include("CompositeOptim.jl")
using .CompositeOptim

# Para o cálculo de Lf
Random.seed!(1)

# Para adicionar um novo Solver na lista abaixo, use 
# Solver(:nome_metodo, params = (:nome_parametro_1 => valor_1, :nome_parametro_2 => valor_2))
# params é opcional e cada método tem seus valores default. No exemplo acima, somente 
# nome_parametro_1 e nome_parametro_2 recebem valores do usuário, e os demais continuam default.
# Os nomes dos paramêtros de cada método no geral seguem a terminologia de seus artigos base.
# Veja o arquivo .jl referente a cada algoritmo para obter seus nomes e valores default. 

# Exemplo com diferentes métodos
# solvers = [Solver(:NSPG), Solver(:ANSPG), Solver(:nmAPGLS), Solver(:newAPG_vs), Solver(:PG), Solver(:FISTA)]
# Exemplo com métodos iguais e parâmetros diferentes
solvers = [Solver(:NSPG, params = (:m => 1)), Solver(:NSPG, params = (:m => 5)), Solver(:NSPG, params = (:m => 10))]

# Legenda dos plots

# Nome dos métodos para os Plots
# Para diferentes métodos, use
# method_names = [rstrip(string(solver.method)*", "*replace(string(solver.params_user), "(" => "", ")" => "", ">" => "", ":" => ""), (',', ' ')) for solver ∈ solvers] 
# Para métodos iguais com parâmetros diferentes, use
method_names = [replace(string(solver.params_user), "(" => "", ")" => "", ">" => "", ":" => "") for solver ∈ solvers]
# Para outros casos, redefina method_names apropriadamente

# Arrays contendo os dados de cada método
s = length(solvers)

T_hist  = Array{Float64}(undef, 25, s)
F_hist  = Array{Float64}(undef, 25, s)
pr_hist = Array{Float64}(undef, 25, s)
gr_hist = Array{Float64}(undef, 25, s)

# Dowload dos problemas pode ser feito em: 
# https://drive.google.com/file/d/1pCuItFylT0SgnyVvWyhIel2Tm8Rf5viC/view?usp = drive_link
# (tamanho ≈ 1.5 GB)
# Sua localização dos dados:
lassodir = "your_path_to_Data-Lasso/Data-Lasso/"

for i = 1:25
    # Leitura dos arquivos dos problemas
    problem_str = "SC$i"
    vars      = matread("$lassodir$problem_str.mat")

    println("$problem_str:")

    A, b, λ₁ = vars["A"], vec(vars["b"]), vars["lambda"] # λ₁ da norma l₁, descartável 

    # Problem struct define algumas variáveis e funções do problema quadrático
    problem = Problem_l2l0(A = A, b = b) # Gera as funções e variáveis genéricas do problema quadrático ℓ₀
    ϵ       = problem.L*10^-5 # Critério de parada
    kₘₐₓ    = 5000 # Número máximo de iterações

    # Algumas variáveis do problema
    println("λ, L, ϵ: ", problem.λ, ", ", problem.L, ", ", ϵ)

    # Solução com cada solver armazenada para comparação
    for j = eachindex(solvers)
        # solve executa o solver escolhido no problema, retornando algumas informações de desempenho
        F_hist[i, j], T_hist[i, j], pr_hist[i, j], gr_hist[i, j] = solve_composite(problem, solvers[j], ϵ, kₘₐₓ)
        println(solvers[j].method, ":\nF_best, Tₜ, prox, ∇f = ", F_hist[i, j], ", ", T_hist[i, j], ", ", pr_hist[i, j], ", ", gr_hist[i, j])
    end
end

# Plots
pltF = performance_profile(PlotsBackend(), F_hist, method_names)
plot!(pltF, dpi = 600, legend = :bottomright, title = "Performance profile of best function value")

pltT = performance_profile(PlotsBackend(), T_hist, method_names)
plot!(pltT, dpi = 600, legend = :bottomright, title = "Performance profile of convergence time")

pltpr = performance_profile(PlotsBackend(), pr_hist, method_names)
plot!(pltpr, dpi = 600, legend = :bottomright, title = "Performance profile of prox calculations")

pltgr = performance_profile(PlotsBackend(), gr_hist, method_names)
plot!(pltgr, dpi = 600, legend = :bottomright, title = "Performance profile of gradient evaluations")

# Salvando os plots e dados (opcional)
#=
experiment_name = "insert_name_here" # Nome do experimento para salvar os plots e dados

path = pwd() # path para salvar plots e dados

# Salvando os plots
savefig(pltpr, "$path/Plots/performance_pr_$experiment_name.png")
savefig(pltgr,  "$path/Plots/performance_gr_$experiment_name.png")
savefig(pltT,  "$path/Plots/performance_T_$experiment_name.png")
savefig(pltF,  "$path/Plots/performance_F_$experiment_name.png")

# Salvando os dados
jldsave("$path/Data/$experiment_name.jld2"; F_hist, T_hist, pr_hist, gr_hist)
#@unpack F_hist, T_hist, pr_hist, gr_hist = jldopen("$path/Data/$experiment_name.jld2", "r")
=#