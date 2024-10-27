import datetime

class APIFramework:
  def __init__(self):    
    # Logger (active / inactive)
    self.api_logger = False
    
    # API Variables
    self.node_training = {}
    self.training_completed = False
    self.stop_reached = False
    self.gradients = {}
    self.model_present = {}
    self.needs_data = []
    self.mem_av = {}
    self.batch_sizes = {}
    self.mini_batch_sizes = {}
    self.training_times = {}
    self.UPDATE = []
    
    self.logger("APIFramework initialized.")
    
  def setUPDATE(self, node_id):
    self.logger("A device has set the UPDATE flag for node " + node_id + ".")
    self.UPDATE.append(node_id)
  
  def isUPDATE(self, node_id):
    self.logger("A device has requested the UPDATE flag for node " + node_id + ".")
    return node_id in self.UPDATE
 
  def resetUPDATE(self, node_id):
    self.logger("A device has requested to reset the UPDATE flag.")
    if node_id in self.UPDATE:
      self.UPDATE.remove(node_id)
  
  def setTrainingTime(self, node_id, time):
    self.logger("A device has set the training time for node " + node_id + " to " + str(time) + ".")
    self.training_times[node_id] = time
    
  def getTrainingTime(self): 
    self.logger("A device has requested the training times.")
    return self.training_times
  
  def isNeedsData(self, node_id):
    self.logger("A device has requested the data status of node " + node_id + ".")
    return node_id in self.needs_data
  
  def setNeedsData(self, node_id, value):
    self.logger("A device has set the data status of node " + node_id + " to " + str(value) + ".")
    
    if value == True and node_id not in self.needs_data:
      self.needs_data.append(node_id)
    elif value == False and node_id in self.needs_data:
      self.needs_data.remove(node_id)
      
  def isStopReached(self):
    self.logger("A device has requested the stop status.")
    return self.stop_reached
  
  def setStopReached(self, value):
    self.logger("A device has set the stop status to " + str(value) + ".")
    self.stop_reached = value
    
  def isTrainingCompleted(self):
    self.logger("A device has requested the training status.")
    return self.training_completed
  
  def setTrainingCompleted(self, value):
    self.logger("A device has set the training status to " + str(value) + ".")
    self.training_completed = value
    
  def isNodeTraining(self, node_id):
    self.logger("A device has requested the training status of node " + node_id + ".")
    
    if node_id in self.node_training:
      return self.node_training[node_id]
    else:
      return False
    
  def isModelPresent(self, node_id):
    self.logger("A device has requested the presence of the model for node " + node_id + ".")
    return self.model_present.get(node_id, False)
  
  def setModelPresent(self, node_id, value):
    self.logger("A device has set the model status of node " + node_id + " to " + str(value) + ".")
    self.model_present[node_id] = value
    
  def setGradients(self, node_id, gradients):
    self.logger("A device has set the gradients for node " + node_id + ".")
    self.gradients[node_id] = gradients
  
  def getGradients(self):
    self.logger("A device has requested the gradients.")
    return self.gradients
  
  def isGradPresent(self, node_id):
    self.logger("A device has requested the presence of gradients for node " + node_id + ".")
    return node_id in self.gradients
  
  def resetNodeGradients(self, node_id):
    self.logger("A device has requested to reset the gradients for node " + node_id + ".")
    if node_id in self.gradients:
      del self.gradients[node_id]
        
  def resetGradients(self):
    self.logger("A device has requested to reset the gradients.")
    self.gradients = {}
    self.gradients_updated = 0
    self.central_gradients = None
  
  def setNodeTraining(self, node_id, value):
    self.logger("A device has set the training status of node " + node_id + " to " + str(value) + ".")
    self.node_training[node_id] = value
    
  def getBatchSize(self, node_id):
    self.logger("A device has requested the batch size for node " + node_id + ".")
    return self.batch_sizes.get(node_id, None)
  
  def setBatchSize(self, node_id, batch_size):
    self.logger("A device has set the batch size for node " + node_id + " to " + str(batch_size) + ".")
    self.batch_sizes[node_id] = batch_size
    
  def isBatchSizeSet(self, node_id):
    self.logger("A device has requested the presence of the batch size for node " + node_id + ".")
    return node_id in self.batch_sizes
  
  def getMiniBatchSize(self, node_id):
    self.logger("A device has requested the batch size for node " + node_id + ".")
    return self.mini_batch_sizes.get(node_id, None)
  
  def setMiniBatchSize(self, node_id, batch_size):
    self.logger("A device has set the batch size for node " + node_id + " to " + str(batch_size) + ".")
    self.mini_batch_sizes[node_id] = batch_size
    
  def isMiniBatchSizeSet(self, node_id):
    self.logger("A device has requested the presence of the batch size for node " + node_id + ".")
    return node_id in self.mini_batch_sizes
  
  def getNodeMemory(self, node_id):
    self.logger("A device has requested the memory for node " + node_id + ".")
    return self.mem_av.get(node_id, None)
  
  def setNodeMemory(self, node_id, mem):
    self.logger("A device has set the memory for node " + node_id + " to " + str(mem) + ".")
    self.mem_av[node_id] = mem
  
  def isNodeMemorySet(self, node_id):
    self.logger("A device has requested the presence of memory for node " + node_id + ".")
    return node_id in self.mem_av
    
  def resetAPI(self):
    self.logger("The master has requested to reset the API.")
    self.node_training = {}
    self.training_completed = False
    pass
  
  def logger(self, message):
    if(self.api_logger == False):
      return
    logMessage = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - API] - " + message
    with open("../data/api-logs/api-logs.txt", "a") as f:
      f.write(logMessage + "\n")
      
  def archiveLastLogs(self):
    self.logger("The API is now closing.")
    """time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    with zipfile.ZipFile("../data/api-logs/api-logs-" + str(time) + ".zip", "w") as z:
      z.write("../data/api-logs/api-logs.txt")"""
      
  def centralizedLogger(self, message, node_id):
    logMessage = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + node_id + "] - " + message
    with open("../data/centralized-logs/distml-central.txt", "a") as f:
      f.write(logMessage + "\n")

