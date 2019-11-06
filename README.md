# qlp

Module for converting set of inequalities to matrix which can be implemented on a quantum annealer.

## Description

The module `qlp` converts a set of `sympy` equations to a `numpy` matrix which corresponds to a constrained minimization problem.

## Install
Install via pip
```bash
pip install [-e] [--user] .
```

## Run
For example
```python
from sympy import symbols
from qlp import eqns_to_matrix

a1, a2, b1, b2, x = symbols("a1, a2, b1, b2, x")

eqns = [
    a1 * x <= b1,
    a2 * x >= b2,
]

m, v, s = eqns_to_matrix(eqns, [x])

m @ deps + v + s
```
returns
```
a1 x - b1 - s1
a2 x - b2 - s2
```

## Authors
* {author}
