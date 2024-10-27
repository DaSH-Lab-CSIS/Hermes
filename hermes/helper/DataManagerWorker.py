import pandas as pd
import getpass
import os
import tensorflow as tf

class DataManagerWorker:
    def __init__(self, logger, node_registry, api_worker, communication):
        self.logger = logger
        self.node_registry = node_registry
        self.api_worker = api_worker
        self.communication = communication
        
        self.node_id = getpass.getuser()
        self.ip = self.node_registry.get_node_ip(self.node_id)
        self.path = self.node_registry.get_node_path(self.node_id)
        self.worker_name = self.node_registry.get_node_worker_name(self.node_id)
        
        self.firstRun = True
        self.model = None
        self.x_currentBatch = None
        self.y_currentBatch = None
        self.x_test = None
        self.y_test = None
        
        self.x_nextBatch = None
        self.y_nextBatch = None
        
        # Clear all files in the dataset-files directory
        for file in os.listdir(self.path + 'hermes/data/dataset-files/'):
            os.remove(self.path + 'hermes/data/dataset-files/' + file)
            
        # Clear all files in the model-files directory
        for file in os.listdir(self.path + 'hermes/data/model-files/'):
            os.remove(self.path + 'hermes/data/model-files/' + file)
            
        for file in os.listdir(self.path + 'hermes/data/centralized-logs/'):
            os.remove(self.path + 'hermes/data/centralized-logs/' + file)
                
    def is_first_run(self):
        return self.firstRun
    
    def set_first_run(self):
        self.firstRun = False
        
    def datasetExists(self):
        return (self.x_currentBatch is not None or self.x_nextBatch is not None)
    
    def modelSetting(self):
        if self.model is None:
            num_files = 0
            for file in os.listdir(self.path + 'hermes/data/model-files/'):
                if file.endswith('.h5'):
                    num_files += 1
            
            if num_files > 0:
                try:
                    self.model = tf.keras.models.load_model(self.path + 'hermes/data/model-files/model.h5')
                except:
                    os.remove(self.path + 'hermes/data/model-files/model.h5')
                    return
                self.logger.log("Model has been successfully loaded!")
                self.api_worker.setModelPresent(self.node_id, True)
                try:
                    os.remove(self.path + 'hermes/data/model-files/model.h5')
                except:
                    pass
            else:
                pass
        else:
            pass

    def removeModel(self):
        self.api_worker.setModelPresent(self.node_id, False)
        self.model = None
    
    def getModel(self):
        return self.model
    
    def doesModelExist(self):
        return self.model is not None
        
    def batchSetting(self, kafka=False):
        if kafka == False:
            self.api_worker.setModelPresent(self.node_id, True)

            if self.x_currentBatch is None:
                # Agreeably, this is a bit of a hacky way to check if the dataset exists; however, it is the most efficient way to do so
                # Until I get another idea, this will have to do
                num_files = len([file for file in os.listdir(self.path + 'hermes/data/dataset-files/') if file.endswith('.xlsx')])
                
                if num_files > 0:
                    self.x_currentBatch = pd.read_excel(self.path + 'hermes/data/dataset-files/dataset.xlsx', sheet_name='x_train')
                    self.y_currentBatch = pd.read_excel(self.path + 'hermes/data/dataset-files/dataset.xlsx', sheet_name='y_train')
                    self.x_currentBatch = self.x_currentBatch.to_numpy()
                    self.y_currentBatch = self.y_currentBatch.to_numpy()
                    
                    # 85-15 split between train and test
                    split = int(len(self.x_currentBatch) * 0.85)
                    self.x_test = self.x_currentBatch[split:]
                    self.y_test = self.y_currentBatch[split:]
                    self.x_currentBatch = self.x_currentBatch[:split]
                    self.y_currentBatch = self.y_currentBatch[:split]
                    self.logger.log("Found dataset; setting batch!")

                    os.remove(self.path + 'hermes/data/dataset-files/dataset.xlsx')
                else:
                    pass
            else:
                pass
        else:
            np_array = self.communication.receive_dataframe_kafka()
            if(np_array == -1):
                return -1
            
            self.x_currentBatch = np_array["x_train"]
            self.y_currentBatch = np_array["y_train"]
            
            # 85-15 split between train and test
            split = int(len(self.x_currentBatch) * 0.85)
            
            self.x_test = self.x_currentBatch[split:]
            self.y_test = self.y_currentBatch[split:]
            self.x_currentBatch = self.x_currentBatch[:split]
            self.y_currentBatch = self.y_currentBatch[:split]
            self.logger.log("Found dataset; setting batch!")
            
    def getBatch(self):
        return {'x': self.x_currentBatch, 'y': self.y_currentBatch}
    
    def resetBatch(self):            
        self.x_currentBatch = None
        self.y_currentBatch = None
