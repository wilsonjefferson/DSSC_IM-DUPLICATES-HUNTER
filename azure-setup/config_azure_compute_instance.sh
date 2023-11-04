#!/bin/bash

# This shell script will be ran while creating the compute instance
# Set proper HTTP proxies 
export http_proxy=http://10.141.0.165:3128
export https_proxy=http://10.141.0.165:3128
echo "Acquire::http::Proxy \"http://10.141.0.165:3128\";" > /etc/apt/apt.conf.d/proxy.conf
echo "Acquire::https::Proxy \"http://10.141.0.165:3128\";" >> /etc/apt/apt.conf.d/proxy.conf

echo "channels:
- https://artifactory.sofa.dev/artifactory/conda-forge/

show_channel_urls: True
allow_other_channels: True

proxy_servers:
    http: http://10.141.0.165:3128
    https: http://10.141.0.165:3128
ssl_verify: True" > /home/azureuser/.condarc

# change the first line "test-starupshell8" to the compute instance name you going to create
echo "127.0.0.1 localhost $1

10.141.0.9 virtuallab-ml.escb.eu
# The following lines are desirable for IPv6 capable hosts
::1 ip6-localhost ip6-loopback
fe00::0 ip6-localnet
ff00::0 ip6-mcastprefix
ff02::1 ip6-allnodes
ff02::2 ip6-allrouters
ff02::3 ip6-allhosts" > /etc/hosts

sudo -u azureuser -i <<'EOF'
sudo apt-get install -y liblttng-ust0

cd /home/azureuser/cloudfiles/code/Users/pietro.morichetti/filtering-of-recognisable-duplicate-tickets/azure_setup/
sudo ./ml_install.sh
sudo ./copy_config.sh

EOF