from kafka import KafkaProducer
import pysftp
import os
import socket
import pandas as pd
import json
import gzip
import pickle
import uuid

class Communication:
    def __init__(self, node_registry, protocol):
        self.node_registry = node_registry
        self.protocol = protocol
        
        with open('../conf/master.json') as f:
            self.master_config = json.load(f)

        if protocol.lower() == "kafka":
            producer_ip = self.master_config["master"]["ip"] + ":9092"
            # Kafka Producer
            self.producer_config = {
                "bootstrap_servers": [producer_ip],
                "batch_size": 32768, # 32 KB
                "linger_ms": 10, # added latency for better batching
            }
            self.producer = KafkaProducer(**self.producer_config)
    
    def checkIfNodeIsConnected(self, node_id, protocol = "SFTP"):
        if protocol == "SFTP":
            ip = self.node_registry.get_node_ip(node_id)
            path = self.node_registry.get_node_path(node_id)
            worker_name = self.node_registry.get_node_worker_name(node_id)
            password = self.node_registry.get_node_password(node_id)
            
            try:
                socket.setdefaulttimeout(5)
                with pysftp.Connection(ip, username = node_id, password = password) as sftp:
                    return True
            except:
                return False
        elif protocol == "Kafka":
            raise NotImplementedError("Kafka protocol not yet implemented")
        else:
            raise ValueError("Invalid protocol")
    
    def sendDataFrame(self, node_id, df, protocol = "SFTP"):
        if df == None:
            return
        
        if protocol == "SFTP":
            self.sendDataFrameSFTP(node_id, df)
        elif protocol == "Kafka":
            self.sendDataFrameKafka(node_id, df)
        else:
            raise ValueError("Invalid protocol")
        
    def sendModel(self, node_id, model):
        self.sendModelSFTP(node_id, model)
    
    # TODO: Implement Kafka protocol for sending models (low priority)
    def sendModelSFTP(self, node_id, model):
        ip = self.node_registry.get_node_ip(node_id)
        path = self.node_registry.get_node_path(node_id)
        worker_name = self.node_registry.get_node_worker_name(node_id)
        password = self.node_registry.get_node_password(node_id)
        try:
            with pysftp.Connection(ip, username = node_id, password = password) as sftp:
                with sftp.cd(path + 'hermes/data/model-files/'):
                    uuid_v = str(uuid.uuid4().hex)
                    model.save(uuid_v + ".h5")
                    # Delete if model.h5 exists
                    try:
                        sftp.remove("model.h5")
                    except:
                        pass
                    
                    try:
                        sftp.remove("model.tmp")
                    except:
                        pass
                    
                    sftp.put(uuid_v + ".h5", "model.tmp")
                    sftp.rename("model.tmp", "model.h5")
            try:            
                os.remove(uuid_v + ".h5")
            except:
                pass
        
        except Exception as e:
            print(e)
            print("Error sending model to node " + node_id)
            self.sendModelSFTP(node_id, model)
    
    
    def sendDataFrameSFTP(self, node_id, np_array):
        ip = self.node_registry.get_node_ip(node_id)
        path = self.node_registry.get_node_path(node_id)
        worker_name = self.node_registry.get_node_worker_name(node_id)
        password = self.node_registry.get_node_password(node_id)
        
        x_train = np_array["x_train"]
        y_train = np_array["y_train"]
        
        with pysftp.Connection(ip, username = node_id, password = password) as sftp:
            with sftp.cd(path + 'hermes/data/dataset-files/'):
                with sftp.open("dataset.tmp", 'w') as f:
                    with pd.ExcelWriter(f) as writer:
                        df_x = pd.DataFrame(x_train)
                        df_y = pd.DataFrame(y_train)
                        df_x.to_excel(writer, sheet_name='x_train', index = False)
                        df_y.to_excel(writer, sheet_name='y_train', index = False)
                
                sftp.rename("dataset.tmp", "dataset.xlsx")
    
    def sendDataFrameKafka(self, node_id, np_array):
        events_topic = 'node-' + node_id + '-kafka'

        # Create pickle dump of np_array
        data = gzip.compress(pickle.dumps(np_array))
        # gzip.compress can cause bottleneck while compressing larger data 
        # (took ~15 seconds to compress the entire mnist training dataframe instead of the first 2000 elements)

        data_size = len(data)
        batch_size = self.producer_config["batch_size"]
        i = 0

        while True:
            batch = data[i:i+batch_size]
            i += batch_size

            if(i >= data_size):
                self.producer.send(events_topic, data[i - batch_size:])
                self.producer.send(events_topic, b"DONE: " + str(data_size).encode() + b" BYTES.")
                break

            self.producer.send(events_topic, batch)
        
