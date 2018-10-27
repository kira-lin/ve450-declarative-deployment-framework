FROM tensorflow/serving:latest
#install airflow
ENV SLUGIFY_USES_TEXT_UNIDECODE=yes

RUN apt-get update -y && apt-get install -y \
        wget \
        python-dev \
        python-pip \
        libczmq-dev \
        libcurlpp-dev \
        curl \
        libssl-dev \
        git \
        inetutils-telnet \
        bind9utils \
        zip \
        unzip \
    && apt-get clean

RUN pip install --upgrade pip

RUN pip install -U setuptools && \
    pip install kubernetes && \
    pip install cryptography && \
    pip install psycopg2-binary==2.7.4

COPY airflow.tar.gz /tmp/

RUN pip install /tmp/airflow.tar.gz
