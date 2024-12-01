using LinearAlgebra, Random
using TenSolver
using JuMP, Gurobi, HiGHS, IsingSolvers
using BenchmarkTools

if haskey(ENV, "SLURM_JOB_PARTITION") && ENV["SLURM_JOB_PARTITION"] == "gpu"
  const MODE = :gpu
  using CUDA, cuTENSOR
  CUDA.set_runtime_version!(v"12.0.1")
  @info "CUDA mode"
else
  const MODE = :cpu
  @info "BLAS mode"
end

function read_rudy(T::Type, filename::String)
  # Read the file and parse each line
  data, cte = open(filename, "r") do file
    lines = readlines(file)
    # This is the template
    # line row value
    # # Constant term of objective = <value>

    ([parse.(T, split(line)) for line in lines if line[1] != '#']
    , parse(T, last(split(lines[2], "=")))
    )
  end

  # Determine the size of the matrix
  max_i = Int(maximum(x -> x[1], data))
  max_j = Int(maximum(x -> x[2], data))

  # Initialize the matrix with zeros
  A = zeros(T, max_i+1, max_j+1)

  # Populate the matrix with the given values
  for (i, j, value) in data
    A[Int(i)+1, Int(j)+1] = value
  end


  return A, cte
end

read_rudy(fn::String) = read_rudy(Float32, fn)

function solve_vrp(n::Int)
  fname  = "TestSet/test_pb_$(n)_o.rudy"

  Q, cte = read_rudy(fname)
  # Regularization for Kernel
  Q += convert(typeof(Q), 1e-8 * diagm(one(cte):n))

  E, psi = TenSolver.solve(Q, nothing, cte
         , device = MODE == :gpu ? CUDA.cu : identity
                      ; eigsolve_krylovdim = 3
                      , eigsolve_verbosity = 3
                      , eigsolve_tol = 1e-8
                      , noise = [1E-4, 1E-7, 1E-8, 0.0]
                      , nsweeps = 5
                      )
  x = TenSolver.sample(psi)
  obj = dot(x, Q, x) + cte

  @info n, E, obj

  return obj, x
end

function solve_mip(n::Int)
  fname  = "TestSet/test_pb_$(n)_o.rudy"

  Q, c = read_rudy(fname)

  # m = Model(Gurobi.Optimizer)
  # Alternative using an Open Source Solver
  m = Model(IsingSolvers.ILP.Optimizer)
  set_attribute(m, "mip_solver", HiGHS.Optimizer)
  @variable m x[1:n] Bin

  @objective m Min dot(x, Q, x) + c

  optimize!(m)

  return objective_value(m), value.(x)
end

function main()
  @info Sys.CPU_THREADS BLAS.get_num_threads()
  println("cores = $(Sys.CPU_THREADS)")
  println("blas thr = $(BLAS.get_num_threads())")
  JOBID = haskey(ENV, "SLURM_JOBID") ? ENV["SLURM_JOBID"] : "local"
  open("result-$(MODE)-$(JOBID).csv", write = true) do io
    write(io, "var,obj,time\n")
  end

  for n in [10, 27, 49, 96, 217, 262, 541, 794]
    println("Solving for $(n) variables")

    try
      # This is a hack to store both the elapsed time and result of running the solver.
      # Unfortunately, BenchmarkTools has no equivalent to @timed
      global ref = Ref{Any}()
      trial = @benchmark(x = solve_mip($(n)), setup = (x = nothing), teardown = (ref[] = x))

      dt = median(trial)
      E, x = ref[]

      @info "Trial" n trial dt
      println("Trial $(n), $(dt)")
      open("result-mip-$(MODE)-$(JOBID).csv", write = true, append = true) do io
        write(io, "$(n),$(E),$(dt)\n")
      end
    catch err
      @error "ERROR: " exception=(err, catch_backtrace())
    end
  end
end

main()
