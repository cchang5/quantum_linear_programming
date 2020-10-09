# qlp

Module for converting set of inequalities to matrix which can be implemented on a quantum annealer.

## Description

The module `qlp` converts a set of `sympy` equations to a `numpy` matrix which corresponds to a constrained minimization problem.

## Install
Install via pip
```bash
pip install [-e] [--user] .
```

## Access to published data

### Setup

The publication data is stored in PostgreSQL format which can be interfaced through `qlpdb`.
This repository provides a dump file which populates an empty PostgreSQL database with the publication data.
To access the data, you have
1. To install and setup `psql`
2. Specify the connection information in `qlpdb/db-config.yaml`
3. Run `make initdb` in `qlpdb`

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
