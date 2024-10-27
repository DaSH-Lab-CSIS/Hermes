#!/bin/bash
  
# Update package list
sudo apt-get update

# Install Python 3 pip
sudo apt install -y python3-pip

# Install necessary Python packages
pip3 install tensorflow --no-cache-dir
pip3 install pyzmq
pip3 install pysftp

# Remove openssh-client
sudo apt-get purge -y openssh-client

# Install ssh
sudo apt install -y ssh

# Remove OpenSSL directory if it exists
sudo rm -rf /usr/lib/python3/dist-packages/OpenSSL/

# Install additional Python packages
pip3 install pandas numpy scikit-learn asyncio psutil openpyxl kafka-python

mkdir distml
