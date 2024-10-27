import api.API as API
import api.APINode as APINode
import helper.Communication as Communication
import helper.Logger as Logger
import helper.NodeRegistry as NodeRegistry
import helper.DataManager as DataManager
import helper.Settings as Settings

# External libraries
import threading
import time
import tensorflow as tf
import numpy as np
import zipfile
import threading
import json
import os

# os.chdir(os.path.dirname(os.path.abspath(__file__)))

class HermesMaster:
    def __init__(self, model, x_train, y_train, x_test, y_test, convergence_iter, learning_rate, num_classes):
        # Initialize model and dataset
        self.model = model
        self.x_train = x_train
        self.x_test = x_test 
        self.y_train = y_train
        self.y_test = y_test
        self.currentEpoch = 0
        self.learning_rate = learning_rate
        self.num_classes = num_classes
        
        # Initialize helper classes
        self.api = API.API()
        self.api_master = APINode.APINode("master")
        self.logger = Logger.Logging(self.api_master, "master")
        self.node_registry = NodeRegistry.NodeRegistry("../conf/config.json")
        self.settings = Settings.Settings("../conf/settings.json", self.node_registry)
        self.communication = Communication.Communication(self.node_registry, self.settings.get_protocol())
        self.data_manager = DataManager.DataManager(self.logger, self.x_train, self.y_train, self.settings, self.model, self.communication, self.api_master)
        self.api_framework = self.api.api_framework
        
        # Initialize master variables
        self.threading = threading.Thread(target=self.api.setup)
        self.threading.start()
        self.connected_nodes = None
        self.saved_grads = []
        self.base_model = model
        tf.keras.models.save_model(self.base_model, '../data/model-files/baseline_model.h5')
        
        # Convergence variables
        self.best_loss = None
        self.iterations = 0
        self.last_update = 0
        self.conv_iter = convergence_iter
        self.conv_flag = False
        self.last_loss = None
        self.best_acc = 0
        
        # mp-training / testing
        policy = tf.keras.mixed_precision.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.set_global_policy(policy)
        
        # Binary Search variables
        self.firstRun = True
        self.median = 0
        
        # TEMP LINE - NEED TO FIX - categorial
        self.y_test = tf.keras.utils.to_categorical(self.y_test, self.num_classes)

        self.logger.log("Master has been successfully initialized!")
    
    ##### SYNCHRONOUS TRAINING #####
    def getConnectedNodes(self):
        connected_nodes = []
        
        for node_id in self.node_registry.get_all_node_ids():
            if self.communication.checkIfNodeIsConnected(node_id):
                connected_nodes.append(node_id)
        
        return connected_nodes
        
    def doTasks(self, node_id, grads, aggregate = False, first_run = False):
        # Aggregate Gradients
        if aggregate:
            grads = json.loads(grads)
            if grads != []: 
                self.updateCentralWeights(grads, node_id)
                print(self.model.trainable_variables[5])
                self.api_framework.resetNodeGradients(node_id)
                self.logger.localLogger("Gradients aggregated from node " + node_id)
                self.logger.manuallyCentralizeLogger("Gradients aggregated from node " + node_id)
                
                # Testing model
                loss, acc = self.testModel(self.model) 
                if loss != 0.0 or acc != 0.0:
                    self.iterations += 1  
                    self.last_loss = loss
                    if self.best_loss == None:
                        self.best_loss = loss
                    else:
                        if(self.best_loss >= loss):
                            self.best_loss = loss
                            self.best_acc = acc
                            self.last_update = 0
                            self.logger.localLogger("Best loss updated to " + str(self.best_loss) + " at iteration " + str(self.iterations))
                            self.logger.manuallyCentralizeLogger("Best loss updated to " + str(self.best_loss) + " at iteration " + str(self.iterations))
                        else:
                            self.last_update += 1
                            self.logger.localLogger("No improvement in loss; last update was " + str(self.last_update) + " iterations ago.")
                            self.logger.manuallyCentralizeLogger("No improvement in loss; last update was " + str(self.last_update) + " iterations ago.")
                            
                # Convergence check
                if self.last_update >= self.conv_iter:
                    self.logger.manuallyCentralizeLogger("Convergence reached; stopping training...")
                    self.conv_flag = True
                
                # Send model
                self.data_manager.send_model(node_id, self.model)
            else:
                self.logger.localLogger("No gradients found for node " + node_id + "; apparently the node hasn't received data.")
                self.logger.manuallyCentralizeLogger("No gradients found for node " + node_id + "; apparently the node hasn't received data.")
                self.api_framework.resetNodeGradients(node_id)

        if first_run == False:
            while True:
                # Check if node is in the dict training time
                if node_id in self.api_framework.getTrainingTime() and self.api_framework.isUPDATE(node_id):
                    self.api_framework.resetUPDATE(node_id)
                    break
                
            self.storage[node_id].append([self.settings.batch_nodes[node_id], self.api_framework.getTrainingTime()[node_id]])
            self.updateBinarySearch(node_id)
        
        while True:
            if self.api_framework.isNeedsData(node_id):
                break
        
        time.sleep(2)
        if(self.data_manager.check_if_next_batch_valid_async() == False):
            self.data_manager.reset_data()
        self.data_manager.send_batch_node_async(node_id)
        
    def updateCentralWeights(self, grads, node_id):
        self.saved_grads = self.aggregateGradients(grads, node_id)
        self.model = tf.keras.models.load_model('../data/model-files/baseline_model.h5')
        self.model.optimizer.apply_gradients(zip(self.saved_grads, self.model.trainable_variables))
        self.logger.manuallyCentralizeLogger("Model updated with aggregated gradients from node " + node_id)      
                
    def aggregateGradients(self, grads, node_id):
        for i in range(len(grads)):
            grads[i] = tf.convert_to_tensor(grads[i])
            
        if self.saved_grads == [] or self.last_loss == None:
            self.saved_grads = grads
        else:
            w1 = 1 / self.last_loss
            # print("Node model test: ---")
            tf.keras.backend.clear_session()
            test_model = tf.keras.models.load_model('../data/model-files/baseline_model.h5')
            test_model.optimizer.apply_gradients(zip(grads, test_model.trainable_variables))
            
            y_pred = test_model(self.x_test)
            loss_fn = tf.keras.losses.get(test_model.loss)
            loss = np.mean(loss_fn(self.y_test, y_pred).numpy())
            
            w2 = 1 / loss
            self.saved_grads = [((w1 * self.saved_grads[i]) + (w2 * grads[i])) / (w1 + w2) for i in range(len(self.saved_grads))]
        
        return self.saved_grads
    
    def binarySearch(self, time_to_search_for, k, node_id):
        common_batch_size = [2, 4, 8, 16, 32, 64]
        
        valid_answers = []
        for i in range(self.x_train.shape[0]):
            high = len(common_batch_size) // 2
            while True:
                if high >= len(common_batch_size) or high < 0:
                    break
                
                val = i * k * 5 / common_batch_size[high]
                 
                if val >= time_to_search_for - 1 and val <= time_to_search_for + 1:
                    valid_answers.append((i, common_batch_size[high]))
                    break  
                elif val > time_to_search_for + 1:
                    if high <= 0:
                        break
                    high = high // 2
                elif val < time_to_search_for - 1:
                    high = high + (high // 2)
                
        # sort and get first element
        valid_answers = sorted(valid_answers, key=lambda x: x[0])
        self.logger.manuallyCentralizeLogger("Optimal batch size: " + str(valid_answers[0][0]))
        self.logger.localLogger("Optimal batch size: " + str(valid_answers[0][1]))
        
        self.logger.manuallyCentralizeLogger("Optimal mini batch size: " + str(valid_answers[0][1]))
        self.logger.localLogger("Optimal mini batch size: " + str(valid_answers[0][1]))
        
        self.api_framework.setMiniBatchSize(node_id, valid_answers[0][1])
        self.api_framework.setBatchSize(node_id, valid_answers[0][0])
        return valid_answers[0]
    
    def setDefaultBatchSize(self, node_id, batch_size):
        self.api_framework.setBatchSize(node_id, batch_size)  
        self.api_framework.setMiniBatchSize(node_id, 16)
        self.logger.manuallyCentralizeLogger("Optimal mini batch size for node " + node_id + " is 16")
        self.logger.manuallyCentralizeLogger("Optimal batch size for node " + node_id + " is " + str(batch_size))
        
        return 2500
    
    def updateBinarySearch(self, node_id):        
        times = self.api_framework.getTrainingTime() 
        if len(times) != len(self.connected_nodes):
            return None
        else:
            # Ensure that this is the first time
            if self.firstRun:
                self.median = np.median([float(value) for key, value in times.items()])
                self.firstRun = False
            
        temp_storage = float(times[node_id])
        times_arr = []
        for key, value in times.items():
            if key == node_id:
                continue
            times_arr.append(float(value))
    
        times_arr = np.array(times_arr)
        k = temp_storage / (5 * self.settings.batch_nodes[node_id] / self.api_framework.getMiniBatchSize(node_id))
                    
        Q1 = np.percentile(times_arr, 25)
        Q3 = np.percentile(times_arr, 75)
        
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q1 - 1.5 * IQR
        print("Upper bound: " + str(upper_bound))
        
        if float(temp_storage) >= upper_bound or float(temp_storage) <= lower_bound:
            self.logger.manuallyCentralizeLogger("Node " + node_id + " has an unusual training; updating batch size...")
            self.logger.localLogger("Node " + node_id + " has exceeded training time; updating batch size...")
            size = self.binarySearch(self.median, k, node_id)
            if size != -1:
                self.batch_nodes[node_id] = size[0]
                self.settings.batch_nodes = self.batch_nodes
    
    ###############################
    
    #### ASYNCHRONOUS TRAINING ####        
    #def checkWhichNodesAreAvailable(self):
    #    available_nodes = []
    #    
    #    for node_id in self.connected_nodes:
    #        if self.api_master.getNodeTraining(node_id) == False and self.api_master.isGradPresent(node_id) == True:
    #            available_nodes.append(node_id)
    #    
    #    return available_nodes  
    
    def checkIfGradPresent(self):
        for node_id in self.connected_nodes:
            if self.api_master.isGradPresent(node_id):
                grads = self.api_master.getGradients()[node_id]
                self.api_master.resetNodeGradients(node_id)
                thread = threading.Thread(target=self.doTasks, args=(node_id, grads, True, False))
                thread.start()    
    
    ###############################
    #### ARCHIVE FUNCTION ####
    def archiveEverything(self):
        str_datetime = time.strftime("%Y%m%d-%H%M%S")
        with zipfile.ZipFile('../data/centralized-logs/logs_' + str_datetime + '.zip', 'w') as zipf:
            for files in os.listdir('../data/centralized-logs/'):
                if files.endswith('><.zip'):
                    zipf.write('../data/centralized-logs/' + files, arcname='./' + files)
                    
            zipf.write('../data/device-logs/local-logs.txt', arcname='./master-logs.txt')
            zipf.write('../data/centralized-logs/distml-central.txt', arcname='./distml-central.txt')
            
        # Clean up the directory
        for files in os.listdir('../data/centralized-logs/'):
            if files.endswith('><.zip'):
                os.remove('../data/centralized-logs/' + files)
                
        os.remove('../data/centralized-logs/distml-central.txt')
        #os.remove('../data/api-logs/api-logs.txt')      
        os.remove('../data/device-logs/local-logs.txt')
        
        for files in os.listdir('../data/model-files/'):
            if files.endswith('.h5'):
                os.remove('../data/model-files/' + files)
                
    ################################
    #### TESTING FUNCTION ####
    def testModel(self, model):
        #TEMP LINE - NEED TO FIX     
        y_pred = model(self.x_test)
        loss_fn = tf.keras.losses.get(model.loss)
        loss = np.mean(loss_fn(self.y_test, y_pred).numpy())
        accuracy = tf.reduce_mean(tf.keras.metrics.categorical_accuracy(self.y_test, y_pred)).numpy()
        self.logger.localLogger("[GLOBAL] Loss: " + str(loss) + " || Accuracy: " + str(accuracy))
        self.logger.manuallyCentralizeLogger("[GLOBAL] Loss: " + str(loss) + " || Accuracy: " + str(accuracy))  
        return loss, accuracy  
    
    ###############################
    #### TRAINING FUNCTION ####    
    def beginTraining(self):        
        # Async ONLY
        self.added_grads = {}        
        self.batch_nodes = {}
        self.storage = {}
        for node_id in self.connected_nodes:
            self.added_grads[node_id] = True
            self.storage[node_id] = [2500]
        
        # Log for sync / async
        self.logger.log("Training type [experiments]: " + str(self.settings.get_training_type()))
        
        # First benchmark to find optimal batch size
        self.logger.log("Binary search for optimal batch size started!")
        
         # Wait until all nodes have pushed their available memory
        while True:
            flag = False
            for node_id in self.connected_nodes:
                if self.api_master.isNodeMemorySet(node_id) == True:
                    flag = True
                else:
                    flag = False
                    break
            
            if flag == True:
                break
            
        # First search 
        for node_id in self.connected_nodes:
            size = self.setDefaultBatchSize(node_id, 2500) # You can change this according to your node natures. 
            if size != -1:
                self.batch_nodes[node_id] = size
                self.settings.batch_nodes = self.batch_nodes
        
        # First iteration
        for node_id in self.connected_nodes:
            thread = threading.Thread(target=self.doTasks, args=(node_id, [], False, True))
            thread.start()
            
        while True:
            self.checkIfGradPresent()
            
            if self.conv_flag == True:
                break

        self.logger.log("Number of iterations: " + str(self.iterations))
        self.logger.log("Best loss: " + str(self.best_loss))       
        self.logger.log("Goodbye, hope you had a great simulation!")
        self.logger.log("Best accuracy: " + str(self.best_acc))
        print(self.storage)
        self.api_master.setTrainingCompleted(True)
        self.api_master.setStopReached(True)
        self.api_master.resetAPI()
        self.api_master.disable()
        self.archiveEverything()
    
    ###############################
    #### MAIN FUNCTION ####
    def start(self):
        # Check API Version
        if self.api_master.getVersion() != "1.0.0":
            raise ValueError("Invalid API version / API connection failed!")
        
        # Get the connected nodes
        self.connected_nodes = self.getConnectedNodes()
        self.logger.log("Number of connected devices: " + str(len(self.connected_nodes)))
        
        inp = input("Ensure that all your worker scripts are running before continuing. This will send your model to the devices. ([y]/n)")
        if(inp == "n"):
            exit()
        
        # Send the model to the connected nodes
        self.logger.log("Sending model to connected nodes...")
        for node_id in self.connected_nodes:
            thread = threading.Thread(target=self.data_manager.send_model, args=(node_id, self.model))
            thread.start()
        
        inp = input("Ensure that all your worker scripts are running before continuing. This will start training. ([y]/n)")
        if(inp == "n"):
            exit()
            
        # Begin training
        self.beginTraining()

