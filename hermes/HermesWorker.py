import api.APINode as APINode
import helper.CommunicationWorker as CommunicationWorker
import helper.Logger as Logger
import helper.NodeRegistry as NodeRegistry
import helper.DataManagerWorker as DataManagerWorker
import helper.TrainerWorker as TrainerWorker
import helper.Settings as Settings

# External libraries
import getpass
import os
import psutil
import numpy as np
import tensorflow as tf

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

class HermesWorker:
    def __init__(self):
        self.node_id = getpass.getuser()
        self.api_worker = APINode.APINode(self.node_id)
        self.communication = CommunicationWorker.CommunicationWorker(self.api_worker)
        self.logger = Logger.Logging(self.api_worker, self.node_id)
        self.node_registry = NodeRegistry.NodeRegistry("../conf/config.json")
        self.data_manager = DataManagerWorker.DataManagerWorker(self.logger, self.node_registry, self.api_worker, self.communication)
        self.training = TrainerWorker.TrainerWorker(self.logger, self.data_manager)
        self.settings = Settings.Settings("../conf/settings.json", self.node_registry)
        
        self.logger.log("Worker has been successfully initialized!")
        self.api_worker.setNodeTraining(self.node_id, False)
        
        # Async variables
        self.training_fin = False
        self.iterations = 0
        
        # Hermes variables
        self.window = []
        self.last_update = 0
        
        # tf policy for mp-training
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)

                
    def first_run(self):
        if(self.data_manager.is_first_run()):
            if(self.data_manager.doesModelExist() == False):
                self.data_manager.modelSetting()
            else:
                self.data_manager.set_first_run()
                
                self.api_worker.setNodeMemory(self.node_id, self.measure_memory())
                
    def measure_memory(self):
        try:
            memory_available = psutil.virtual_memory().available
            return memory_available
        except Exception as e:
            print(e)
            print("Error measuring memory available. Skipping measurement.")
            return None
                  
    def trainModel(self, batch_size):
        endTime = self.training.train(self.data_manager.getModel(), batch_size)
        self.api_worker.setTrainingTime(self.node_id, endTime)
        self.api_worker.setUPDATE(self.node_id)
        return 0

    def beginTraining(self):
        # Check API Version
        if self.api_worker.getVersion() != "1.0.0":
            raise ValueError("Invalid API version / API connection failed!")
                
        self.z_thres = self.settings.get_alpha()       
        while True:
            if(self.api_worker.isStopReached()):
                self.logger.log("Master has stopped; training completed!")
                break
            
            if(self.training_fin == True):
                self.logger.log("Training complete for one epoch, time to decide if major or not...")

                if len(self.window) == self.settings.get_window_size():
                    self.window.pop(0)
                    
                self.window.append(self.training.loss)
                
                major_update = False
                z_score = 0.0
                
                if len(self.window) == 1:
                    major_update = False
                else:
                    mu = np.mean(self.window)
                    sigma = np.std(self.window)
                    z_score = (self.training.loss - mu) / sigma
                    if(z_score <= self.z_thres):
                        major_update = True
                        
                if major_update:
                    # This is a major update
                    self.logger.log("Major update detected; z_score = " + str(z_score) + "; z_thres = " + str(self.z_thres) + "; sending gradients to master...")
                    self.data_manager.removeModel()
                    self.api_worker.setNodeTraining(self.node_id, False)
                    self.communication.send_gradients(grads, self.api_worker)
                    self.logger.log("Gradients sent to master; waiting for the global model...")

                    self.window = []
                    self.last_update = 0
                    self.logger.log("The window needs to be repopulated; clearing for new model...")
                                
                    # Be ready to receive the global model
                    while(self.data_manager.doesModelExist() == False):
                        self.data_manager.modelSetting()  
                    print(self.data_manager.model.trainable_variables[5])      
                    self.api_worker.log("Model received from master; training again!")
                    self.training_fin = False
                else:
                    self.logger.log("Minor update detected; z_score = " + str(z_score) + "; z_thres = " + str(self.z_thres) + "; training again; last_update = " + str(self.last_update) + ";")
                    self.training_fin = False
                    self.last_update += 1
                    if self.last_update >= self.settings.get_lambda():
                        self.z_thres = min(0.0, self.z_thres * (1 - self.settings.get_beta()))
                    
                    # Send empty gradients to get data
                    self.communication.send_gradients([], self.api_worker)
                
            else:                
                if(self.data_manager.is_first_run()):
                    self.first_run()
                    continue
                
                if(self.data_manager.datasetExists() == False):
                    valid = self.data_manager.batchSetting(self.settings.get_protocol() == "Kafka")
                    if valid == -1:
                        # This means a mismatch in sent and received data
                        self.logger.log("Mismatch in sent and received data; skipping and resending gradients...")
                        self.training_fin = True
                        continue
                
                if(self.data_manager.datasetExists()):
                    self.logger.log("Dataset found: let's start training!")

                    self.api_worker.setNodeTraining(self.node_id, True)
                    
                    print("Iteration: " + str(self.iterations + 1))
                    self.iterations += 1
                    self.trainModel(self.api_worker.getMiniBatchSize(self.node_id))
                    grads = self.training.get_sum_grads()
                    self.api_worker.setNodeTraining(self.node_id, False)
                    self.training_fin = True
 
