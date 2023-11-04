#!/bin/bash

# Set proper HTTP proxies 
export http_proxy=http://10.141.0.165:3128
export https_proxy=http://10.141.0.165:3128
echo "Acquire::http::Proxy \"http://10.141.0.165:3128\";" > /etc/apt/apt.conf.d/proxy.conf
echo "Acquire::https::Proxy \"http://10.141.0.165:3128\";" >> /etc/apt/apt.conf.d/proxy.conf

# Export basic config variables
export CONF_PATH=/etc
export ODBC_LIB_PATH=/opt/cloudera/impalaodbc/lib/64
export CLOUDERA_PKG=ClouderaImpalaODBC-2.6.14.1016-1.x86_64.rpm
export DEBIAN_FRONTEND=noninteractive
export IMPALA_INSTALL_PATH=/srv/impala
export ORIGIN_DIR=$(pwd)

# Install firstly system wide basic libraries
apt update && apt install -y --no-install-recommends\
    libcurl4-gnutls-dev \
    libssl-dev \
    unixodbc-dev \
    odbcinst1debian2 odbcinst libodbc1 unixodbc libsasl2-dev wget alien libaio1\
    ca-certificates
apt-get -yqq install krb5-user libpam-krb5 && apt-get -yqq clean

#Cloudera package
mkdir -p $IMPALA_INSTALL_PATH
wget https://downloads.cloudera.com/connectors/impala_odbc_2.6.14.1016/Linux/$CLOUDERA_PKG 
mv $CLOUDERA_PKG $IMPALA_INSTALL_PATH
cd $IMPALA_INSTALL_PATH
alien -i $CLOUDERA_PKG

# Complete the configuration
cd $ORIGIN_DIR

# Copy krb5.conf to /etc
cp krb5.conf $CONF_PATH

# Copy odbc config files
cp odbc.ini $CONF_PATH

# Copy the trusted certificates
cp accprd-truststore.crt $ODBC_LIB_PATH

# set /etc/hosts to point to Proxy for the impala
echo "10.141.0.164 impala.mit01.ecb.de" >> /etc/hosts
echo "10.141.0.164 impala.devo.escb.eu" >> /etc/hosts