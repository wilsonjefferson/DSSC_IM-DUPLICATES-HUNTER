#!/bin/bash

# Set proper HTTP proxies
export http_proxy=http://10.141.0.165:3128
export https_proxy=http://10.141.0.165:3128

# Export basic config variables
export CONF_PATH=/etc
export ODBC_LIB_PATH=/opt/cloudera/impalaodbc/lib/64
export CLOUDERA_PKG=ClouderaImpalaODBC-2.6.14.1016-1.x86_64.rpm
export DEBIAN_FRONTEND=noninteractive
export IMPALA_INSTALL_PATH=/srv/impala
export ORIGIN_DIR=$(pwd)

# Copy krb5.conf to /etc
cp krb5.conf $CONF_PATH

# Copy odbc config files
cp odbc.ini $CONF_PATH
