# Integer Programming from Quantum Annealing and Open Quantum Systems

This repository contains software and data associated with publication [Integer Programming from Quantum Annealing and Open Quantum Systems [2009.11970]](https://arxiv.org/abs/2009.11970).

It contains Python code for mapping [Integer Linear Programming problems](https://en.wikipedia.org/wiki/Integer_programming), in particular the [Minimum Dominating Set](https://mathworld.wolfram.com/MinimumDominatingSet.html) Problem, to [QUBOs or Ising Hamiltonians utilized as input for Quantum Annealers](https://docs.dwavesys.com/docs/latest/c_gs_3.html#qubo).
It furthermore contains code which simulates the quantum hardware.


## How to use this repository?

This repository contains two modules

1. `qlp` used for computations and
2. `qlpdb` for accessing and storing computation data

### Access publication data

This repository contains all the data presented in [2009.11970]](https://arxiv.org/abs/2009.11970).
To access this data you must host a [PostgreSQL database](https://www.postgresql.org/about/) (usage of `JSON` and `ArrayFields`).
We provide more information in the [`qlpdb/README.md`](qlpdb/README.md).

## How to install it?

### Install the computation module
The repository source code can be installed via pip:
```bash
pip install [--user] .
```

### Install the data module

To interface with the publication data, you also have to install the data module `qlpdb`
```bash
cd qlpdb
pip install [--user] .
```

## Usage

### `qlp`

The module `qlp` contains two major components

* The first submodule, `mds`, was used to map the Minimum Dominating Set Problem to annealing Hardware:
    ```python
    from qlp.mds.graph_tools as generate_nn_graph
    from qlp.mds.qubo import get_mds_qubo

    # generate line graph of length 4: o-o-o-o
    graph, tag = gt.generate_nn_graph(4)
    # Generate mds qubo
    qubo = get_mds_qubo(
        graph,          # The graph as a set of edges
        directed=False, # no directed edges
        penalty=2       # strength of penalty term
    )
    ```
    The solution to the MDS problem is given by the bit vector `psi` which minimizes
    ```python
    E = psi.T@qubo@psi
    ```
    The QUBO serves as input for the annealing hardware.
* the second submodule, `tdse`, simulates the time dependent Schrödinger equation for given input (Ising) Hamiltonians:
    ```python
    from qlp.mds.mds_qlpdb import QUBO_to_Ising, graph_summary
    from qlp.tdse import TDSE

    Jij, hi, c = QUBO_to_Ising(qubo.todense().tolist())
    ising_params = {
        "Jij": [list(row) for row in Jij],
        "hi": list(hi),
        "c": c,
        "energyscale": annealing_time * 1000.0,
        "qubo_constant": penalty * nvertices,
        "penalty": penalty,
    }
    graph_params = graph_summary(tag, graph, qubo)
    ...
    # Initialize the solver
    tdse = TDSE(
        graph_params,  # Graph parameters
        ising_params,  # Ising Hamiltonian parameters
        offset_params, # how is the annealing curve implemented
        solver_params, # numerical parameters for solving the time dependent equation
    )
    # Compute the starting wave density
    rho = tdse.init_densitymatrix(
        temp,        # full counting decoherence parameters
        temp_local,  # local decoherence parameters
        initial_wavefunction="transverse",
    )
    tdse.gamma = gamma              # set FC decoherence rate
    tdse.gamma_local = gamma_local  # set local decoherence rate
    # Solve time dependent Schrödinger equation
    sol_densitymatrix = tdse.solve_mixed(rho)
    ```

See also the [`notebooks`](notebooks) folder, especially the [`notebooks/runs`](notebooks/runs) for reproducing computations.

### `qlpdb`







## Who is responsible for it?
* [@cchang5](https://github.com/cchang5)
* [@ckoerber](https://www.ckoerber.com)
* [@lastyoru](https://github.com/lastyoru)

Feel free to reach out for questions.

## License

BSD 3-Clause License. See also the [LICENSE](LICENSE.md) file.
