import os
default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

######################## set up the environment #########################


import numpy as np
import os

from sklearn.model_selection import train_test_split



from pysr import PySRRegressor





loss_function = r"""
function custom_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
  # evaluate the symbolic model
  ŷ, ok = eval_tree_array(tree, dataset.X, options)
  if !ok
    return L(Inf)
  end

  N = dataset.n
  y = dataset.y

  # (4) MSE part
  mse = sum((y .- ŷ) .^2 ./ (y .^ 2)) / N

  # (5) average quantile loss
  quantiles = (0.1, 0.25, 0.5, 0.75, 0.9)
  qloss = zero(mse)
  for τ in quantiles
    err = (y .- ŷ) ./ y
    qloss += sum( τ * max.( err, zero(err) )
                .+ (1-τ) * max.( -err, zero(err) ) ) / N
  end
  qloss /= length(quantiles)

  # (6) total
  return mse + qloss
end
"""

elementwise_loss = r"""
loss(ŷ, y) = begin

  # normalized MSE part
  sq = (y - ŷ)^2 / (y^2)

  # normalized quantile error
  err = (y - ŷ) / y
  q  = sum(
         τ * max(err, 0) + (1 - τ) * max(-err, 0)
         for τ in (0.1, 0.25, 0.5, 0.75, 0.9)
       ) / 5

  return sq + q
end
"""



"""
One needs to define X, halo properties as features, and y, the target variable.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


    



"""
Hyperparameters used in the paper
"""

model = PySRRegressor(
    populations=150,
    # ^ Assuming we have 4 cores, this means 2 populations per core, so one is always running. Better set a 3*proc
    #population_size=50,
    ncycles_per_iteration=500,
    # ^ Generations between migrations.
    niterations=300,  # Run forever
    early_stop_condition=(
        "stop_if(loss, complexity) = loss < 1e-4 && complexity < 50"
        # Stop early if we find a good and simple equation
    ),
    timeout_in_seconds=60 * 60 * 8,
    # ^ Alternatively, stop after 1 hours have passed.
    maxsize=100,
    # ^ Allow greater complexity.
    maxdepth=8,
    # ^ But, avoid deep nesting.
    binary_operators=["*","^","+"],
    unary_operators=["exp","log10"],
    constraints={
        "+": (-1, -1),
        #"-": (8, 8),
        "*": (10, 10),
        #"/": (3, 3),
        "^": (10, 2),
        "exp": 10,
        "log10": 10,
        #"log10": 3,
    },
    complexity_of_variables=3,
    # ^ Limit the complexity within each argument.
    # "inv": (-1, 9) states that the numerator has no constraint,
    # but the denominator has a max complexity of 9.
    # "exp": 9 simply states that `exp` can only have
    # an expression of complexity 9 as input.
    nested_constraints={
        #"^" : { "exp": 0, "log10": 0,  "^":0},
        "exp": { "exp": 0, "log10": 0,  "^":0},
        "log10": { "exp": 0, "log10": 0,  "^":0},},
    #"log10": { "exp": 0, "log": 0, "log10": 0, "^":0},
    #},
    # ^ Nesting constraints on operators. For example,
    # "square(exp(x))" is not allowed, since "square": {"exp": 0}.
    #complexity_of_operators={"/": 2, "exp": 3},
    # ^ Custom complexity of particular operators.
    complexity_of_constants=1,
    # ^ Punish constants more than variables
    #select_k_features=4,
    # ^ Train on only the 4 most important features
    progress=True,
    # ^ Can set to false if printing to a file.
    #weight_randomize=0.1,
    # ^ Randomize the tree much more frequently
    #precision=32,
    # ^ Higher precision calculations.
    #turbo=True,
    # ^ Faster evaluation (experimental),
    #batching=True,
    #batch_size=2000,
    #parallelism="multiprocessing",
    procs=50,
    turbo=True,
    weight_optimize=0.001,
    elementwise_loss=elementwise_loss,
    #loss_function=loss_function,
    #warm_start=True
    #parsimony=1e-5/5,
    #weight_optimize=0.001,
    #bumper=True
    verbosity=1,
    input_stream='devnull'
)

model.fit(X_train, y_train)




"""
One can start from a previous run by loading from a directory
"""
"""
warmstart = PySRRegressor.from_file(run_directory=r"./XXX")

# Check current state
print(f"Current iterations completed: {warmstart.niterations}")
print(f"Best equations found: {len(warmstart.equations_)}")

# To continue, increase niterations and fit
warmstart.set_params(niterations=200,warm_start=True)  
warmstart.fit(X, y)  # This should continue from where it left off
"""

