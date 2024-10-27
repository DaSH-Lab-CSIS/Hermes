from kafka import KafkaConsumer
import json
import getpass
import time
import gzip
import pickle
import multiprocessing

class CommunicationWorker:
    def __init__(self, api_worker):
        with open('../conf/master.json') as f:
            self.master_config = json.load(f)
            
        self.api_worker = api_worker
            
        self.node_id = getpass.getuser()
        self.ip = self.master_config["master"]["ip"]
        self.username = self.master_config["master"]["username"]
        self.password = self.master_config["master"]["password"]
        self.path = self.master_config["master"]["path"]   
        
        consumer_ip = self.ip + ":9092"
        
        consumer_config_kwargs = {
            "bootstrap_servers": [consumer_ip],
            "fetch_min_bytes": 32768, # 32 KB,
            "fetch_max_wait_ms": 10
        }
        self.events_topic = 'node-' + self.node_id + '-kafka'
        self.consumer = KafkaConsumer(self.events_topic, **consumer_config_kwargs)   
    
    def send_gradients(self, gradients, api_worker):
        #This is an EagerTensor, need to make it JSON serializable
        api_worker.setGradients(self.node_id, gradients)
        
    def wait_until_data_received(self):
        time.sleep(15)
        if self.flag == False:
            print("Data has not been received in last 15 seconds. Pushing no gradients.")
            self.flag = True
        
    def receive_dataframe_kafka(self):
        received_data = b''
        self.api_worker.setNeedsData(self.node_id, True)
        self.flag = False
        count = 0
        
        # Use function to push past gradients if no data received in last 15 seconds
        thread = multiprocessing.Process(target=self.wait_until_data_received)
        thread.start()
        for chunk in self.consumer:
            if self.flag == True:
                return -1
            
            chunk = chunk.value
            if chunk[:4] != b"DONE":
                count += 1
                if count == 1:
                    thread.terminate()
                    thread.join()
                    print("Thread terminated.")
                received_data += chunk
                
            else:
                received_data_size = int(chunk.split()[1])
                break
        self.api_worker.setNeedsData(self.node_id, False)

        if len(received_data) != received_data_size:
            print("WARNING: Mismatch in sent and received data.")
            return -1 # ideally this shouldnt happen

        return pickle.loads(gzip.decompress(received_data))
    
