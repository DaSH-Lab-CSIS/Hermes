#!/bin/bash

# Update package list
sudo apt-get update

# Install Python 3 pip
sudo apt install -y python3-pip

# Install necessary Python packages
pip3 install tensorflow --no-cache-dir
pip3 install pyzmq
pip3 install pysftp
pip3 install kafka-python

# Remove openssh-client
sudo apt-get purge -y openssh-client

# Install ssh
sudo apt install -y ssh git

# Remove OpenSSL directory if it exists
sudo rm -rf /usr/lib/python3/dist-packages/OpenSSL/

# Install additional Python packages
pip3 install pandas numpy scikit-learn asyncio psutil openpyxl

mkdir distml

# Install Java
sudo apt install openjdk-11-jre-headless

# Install Kafka
wget https://downloads.apache.org/kafka/3.8.0/kafka_2.13-3.8.0.tgz
tar xvf kafka_2.13-3.8.0.tgz

echo "listeners=INTERNAL://0.0.0.0:9092
listener.security.protocol.map=INTERNAL:PLAINTEXT
advertised.listeners=INTERNAL://[REPLACE-WITH-MASTER-IP]:9092
inter.broker.listener.name=INTERNAL" >> kafka_2.13-3.8.0/config/server.properties
