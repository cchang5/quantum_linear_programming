# qlpdb

Data module for publication [2009.11970]](https://arxiv.org/abs/2009.11970).
It allows interfacing, searching and saving database structures through Python objects.


## Requirements

The implemented data structures require a [PostgreSQL database](https://www.postgresql.org/about/) (`JSON` and `ArrayFields`).
Thus, you first have to setup a [PostgreSQL database server](https://www.postgresql.org/docs/9.1/server-start.html).

If you prefer a shortcut, we also specify a high-level [Docker setup](https://docs.docker.com/get-docker/).

## Docker
Once you have [started Docker](https://docs.docker.com/get-started/), you can install the repository software and initialize data structures by running
```bash
docker-compose build
```
in the repository root directory.
Note that this might take a while.

In case data structures have changed, you have to run once
```bash
docker-compose run qlpdb python manage.py migrate
```

Finally, to launch the database interface, you can run
```bash
docker-compose up
```
This data server can be access through any web browser at the address
```
http://localhost:8000
```

## Details
The data is stored in a `sql` dump file which is contained in the `data.tar.gz`.
We also provide a `SHA-512` checksum for verifying the data integrity.

You can create a custom PostgreSQL database and user for loading in the dump file
```bash
psql [-U {USER}] {DBNAME} < data/qlp-dump.sql
```

Next, after `pip` installing `qlpdb`, you have to specify the connection information.
This is done by creating and adjusting the `db-config.yaml` file in the `qlpdb` directory.
We provide the example file `db-config.example.yaml` in the same directory.
See also the [EspressoDB docs](https://espressodb.readthedocs.io/en/latest/Usage.html?highlight=db-config#configure-your-project).

**NOTE: Do not commit or share this file in case the database credentials are sensitive.**

After credentials are matched to the database, you may have to update data structures by running
```bash
python manage.py migrate
```
and can launch the data browser by running
```bash
python manage.py runserver
```
