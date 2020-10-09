CREATEDB=createdb
PSQL=psql
CHECKSUM=shasum -a 512

DBNAME:= $(shell sed -n 's/^NAME:[[:blank:]]*\(..*\)/\1/p' db-config.yaml)
USER:= $(shell sed -n 's/^USER:[[:blank:]]*\(..*\)/\1/p' db-config.yaml)

initdb: data/qlp-dump.sql
	$(info Creating db $(DBNAME))
	$(CREATEDB) $(DBNAME)

	$(info Creating user $(USER) (if not exists))
	$(PSQL) -d $(DBNAME) -tc "SELECT 1 FROM pg_user WHERE usename = '$(USER)';" | grep -q 1 || \
		$(PSQL) -d $(DBNAME) -c "CREATE USER $(USER);"

	$(info Loading dump file. This might take a while)
	$(PSQL) $(DBNAME) < data/qlp-dump.sql

data/qlp-dump.sql: data.tar.gz
	$(info Extracting data)
	tar -xvf data.tar.gz
	$(info Checksum)
	cat data/sha512.txt
	$(info local checksum)
	$(CHECKSUM) data/qlp-dump.sql
