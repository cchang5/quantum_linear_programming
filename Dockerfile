# Load in python image
FROM python:3.7-buster

# Who to contact
MAINTAINER Christopher Körber

# Environment variables and dirs used
ENV APP_DIR=/opt/app

# Create dirs
RUN mkdir -p $APP_DIR
RUN mkdir -p $APP_DIR/qlp
RUN mkdir -p $APP_DIR/qlpdb
RUN mkdir -p $APP_DIR/.pip_cache

# Install requirements
RUN pip install --upgrade pip --cache-dir $APP_DIR/.pip_cache

## Install qlp
WORKDIR $APP_DIR/qlp
COPY requirements.txt .
COPY README.md .
COPY setup.py .
COPY qlp qlp
RUN pip install -r requirements.txt --cache-dir $APP_DIR/.pip_cache
RUN pip install .

# Install qlpdb
# DO NOT COPY FILES LIKE db-config.yaml or settings.yaml if they contain secrete passwords
WORKDIR $APP_DIR/qlpdb
COPY qlpdb/requirements.txt .
COPY qlpdb/setup.py .
COPY qlpdb/settings.yaml .
COPY qlpdb/db-config.example.yaml db-config.yaml
COPY qlpdb/README.md .
COPY qlpdb/manage.py .
COPY qlpdb/qlpdb qlpdb
RUN pip install -r requirements.txt --cache-dir $APP_DIR/.pip_cache
RUN pip install .

# Port to expose
EXPOSE 8000

# Run entrypoint script
CMD ["python", "manage.py", "runserver"]
