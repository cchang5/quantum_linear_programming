# Integer Programming from Quantum Annealing and Open Quantum Systems

This repository contains software and data associated with publication [Integer Programming from Quantum Annealing and Open Quantum Systems [2009.11970]](https://arxiv.org/abs/2009.11970).

It contains Python code mapping [Integer Linear Programming problems](https://en.wikipedia.org/wiki/Integer_programming), in particular the [Minimum Dominating Set](https://mathworld.wolfram.com/MinimumDominatingSet.html) Problem, to [QUBOs or Ising Hamiltonians utilized as input for Quantum Annealers](https://docs.dwavesys.com/docs/latest/c_gs_3.html#qubo).
It furthermore contains code which simulates the quantum hardware.


## What does EspressoDB provide?

EspressoDB provides an easy to use database interface which helps you make educated decisions fast.

Once you have created your Python project (e.g., `my_project`) with EspressoDB

* you can use it in all your Python apps to query your data. For example,
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
    The solution to the MDS problem is given by the vector bit `psi` which minimizes
    ```python
    E = psi.T@qubo@psi
    ```
* you can generate web views which summarize your tables and data.
    ![Docpage example](https://raw.githubusercontent.com/callat-qcd/espressodb/master/doc-src/_static/webview-example.png)
    Because the web pages use a Python API as well, you can completely customize views with code you have already developed.
    E.g., you can automate plots and display summaries in your browser.
    If you want to, you can also make your web app public (with different layers of accessibility) and share results with others.


See also the [Documentation](https://espressodb.readthedocs.io/en/latest/) for more detailed usage instructions.

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

### Access publication data

This repository contains all the data presented in [2009.11970]](https://arxiv.org/abs/2009.11970).
To access this data you must host a [PostgreSQL database](https://www.postgresql.org/about/) (usage of `JSON` and `ArrayFields`).
We provide more information in the [`qlpdb/README.md`](qlpdb/README.md).


## What's the story behind it?

EspressoDB was developed when we created [LatteDB](https://www.github.com/callat-qcd/lattedb) -- a database for organizing Lattice Quantum Chromodynamics research.
We intended to create a database for several purposes, e.g. to optimize the scheduling of architecture-dependent many-node jobs and to help in the eventual analysis process.
For this reason, we started to abstract our thinking of how to organize physics objects.

It was the goal to have easily shareable and completely reproducible snapshots of our workflow while being flexible and not restricting ourselves too much -- in the end science is full of surprises.
The challenges we encountered were:
1. How can we program a table structure which can be easily extended in the future?
2. How do we write a database such that users not familiar with the database concept can start using this tool with minimal effort?

The core module of LatteDB, EspressoDB, is trying to address those challenges.

## Who is responsible for it?
* [@cchang5](https://github.com/cchang5)
* [@cchang5](https://github.com/cchang5)
* [@ckoerber](https://www.ckoerber.com)

Feel free to reach out for questions.

## License

BSD 3-Clause License. See also the [LICENSE](LICENSE.md) file.
