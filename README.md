# qlp

Module for converting set of inequalities to matrix which can be implemented on a quantum annealer.

## Description

The module `qlp` converts a set of `sympy` equations to a `numpy` matrix which corresponds to a constrained minimization problem.

## Install
Install via pip
```bash
pip install [-e] [--user] .
```

## Compile the doc
Run once to get dependencies
```bash
pip install [-e] [--user] .
pip install -r requirements-dev.txt
```
Compile the docs
```bash
cd docs
make html
open build/html/index.html
```
