import zmq
import json

class APINode:
    def __init__(self, node_id):
        self.node_id = node_id
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        
        with open('../conf/master.json') as f:
            self.master_config = json.load(f)
            
        connection = "tcp://" + self.master_config["master"]["ip"] + ":5555"  
        self.socket.connect(connection)
        
    def isStopReached(self):
        """
        Output: a boolean indicating whether the stop has been reached
        """
        self.socket.send_multipart([b'IS_STOP_REACHED', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def setStopReached(self, value):
        """
        Input: value - a boolean indicating whether the stop has been reached
        Output: None
        """
        self.socket.send_multipart([b'SET_STOP_REACHED', str(value).encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
        
    def isTrainingCompleted(self):
        """
        Output: a boolean indicating whether training is completed
        """
        self.socket.send_multipart([b'IS_TRAINING_COMPLETED', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def setTrainingCompleted(self, value):
        """
        Input: value - a boolean indicating whether training is completed
        Output: None
        """
        self.socket.send_multipart([b'SET_TRAINING_COMPLETED', str(value).encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
        
    #def getNodeTraining(self, node_id):
    #    """
    #    Output: a boolean indicating whether the node is training
    #    """
    #    self.socket.send_multipart([b'GET_NODE_TRAINING', node_id.encode(), b"NOTNEEDED"])
    #    response = self.socket.recv()
    #    return response.decode() == "True"
    
    def setNodeTraining(self, node_id, value):
        """
        Input: value - a boolean indicating whether the node is training
        Output: None
        """
        self.socket.send_multipart([b'SET_NODE_TRAINING', node_id.encode(), str(value).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def getGradients(self):
        """
        Output: a dictionary containing the gradients
        """
        self.socket.send_multipart([b'GET_GRADIENTS', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return json.loads(response.decode())
    
    def setGradients(self, node_id, gradients):
        """
        Input: gradients - a list containing the gradients (which are Tensors)
        Output: None
        """
        # Convert all tensors to numpy arrays to lists
        if gradients is not None:
            gradients = [gradient.numpy().tolist() for gradient in gradients]
        else:
            gradients = []
        self.socket.send_multipart([b'SET_GRADIENTS', node_id.encode(), json.dumps(gradients).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def resetNodeGradients(self, node_id):
        """
        Output: None
        """
        self.socket.send_multipart([b'RESET_NODE_GRADIENTS', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def isGradPresent(self, node_id):
        """
        Output: a boolean indicating whether gradients are present for the node
        """
        self.socket.send_multipart([b'IS_GRADIENT_PRESENT', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def resetGradients(self):
        """
        Output: None
        """
        self.socket.send_multipart([b'RESET_GRADIENTS', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def getTrainingTime(self):
        """
        Output: a dictionary containing the training time of the node
        """
        self.socket.send_multipart([b'GET_TRAINING_TIME', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return json.loads(response.decode())
    
    def setTrainingTime(self, node_id, time):
        """
        Input: time - a dictionary containing the training time of the node
        Output: None
        """
        self.socket.send_multipart([b'SET_TRAINING_TIME', node_id.encode(), json.dumps(time).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def setUPDATE(self, node_id):
        """
        Output: None
        """
        self.socket.send_multipart([b'SET_UPDATE', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"

    def getUPDATE(self, node_id):
        """
        Output: a boolean indicating whether the node needs to be updated
        """
        self.socket.send_multipart([b'GET_UPDATE', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def resetUPDATE(self, node_id):
        """
        Output: None
        """
        self.socket.send_multipart([b'REMOVE_UPDATE', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def isModelPresent(self, node_id):
        """
        Output: a boolean indicating whether the model is present for the node
        """
        self.socket.send_multipart([b'IS_MODEL_PRESENT', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def setModelPresent(self, node_id, value):
        """
        Input: value - a boolean indicating whether the model is present for the node
        Output: None
        """
        self.socket.send_multipart([b'SET_MODEL_PRESENT', node_id.encode(), str(value).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def needsData(self, node_id):
       """
       Output: a boolean indicating whether the node needs data
       """
       self.socket.send_multipart([b'IS_NEEDS_DATA', node_id.encode(), b"NOTNEEDED"])
       response = self.socket.recv()
       return response.decode() == "True"
    
    def setNeedsData(self, node_id, value):
        """
        Input: value - a boolean indicating whether the node needs data
        Output: None
        """
        self.socket.send_multipart([b'SET_NEEDS_DATA', node_id.encode(), str(value).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def getMiniBatchSize(self, node_id):
        """
        Output: an integer containing the batch size of the node
        """
        self.socket.send_multipart([b'GET_MINI_BATCH_SIZE', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return int(json.loads(response.decode()))
    
    def setMiniBatchSize(self, node_id, batch_size):
        """
        Input: batch_size - an integer containing the batch size of the node
        Output: None
        """
        self.socket.send_multipart([b'SET_MINI_BATCH_SIZE', node_id.encode(), str(batch_size).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def isMiniBatchSizeSet(self, node_id):
        """
        Output: a boolean indicating whether the batch size is present for the node
        """
        self.socket.send_multipart([b'IS_MINI_BATCH_SIZE_SET', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def getBatchSize(self, node_id):
        """
        Output: an integer containing the batch size of the node
        """
        self.socket.send_multipart([b'GET_BATCH_SIZE', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return int(json.loads(response.decode()))
    
    def setBatchSize(self, node_id, batch_size):
        """
        Input: batch_size - an integer containing the batch size of the node
        Output: None
        """
        self.socket.send_multipart([b'SET_BATCH_SIZE', node_id.encode(), str(batch_size).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def isBatchSizeSet(self, node_id):
        """
        Output: a boolean indicating whether the batch size is present for the node
        """
        self.socket.send_multipart([b'IS_BATCH_SIZE_SET', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def getNodeMemory(self, node_id):
        """
        Output: a dictionary containing the memory of the node
        """
        self.socket.send_multipart([b'GET_NODE_MEM', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return json.loads(response.decode())
    
    def setNodeMemory(self, node_id, mem):
        """
        Input: mem - a dictionary containing the memory of the node
        Output: None
        """
        self.socket.send_multipart([b'SET_NODE_MEM', node_id.encode(), json.dumps(mem).encode()])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def isNodeMemorySet(self, node_id):
        """
        Output: a boolean indicating whether the memory is present for the node
        """
        self.socket.send_multipart([b'IS_NODE_MEM_SET', node_id.encode(), b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "True"
    
    def resetAPI(self):
        """
        Output: None
        """
        self.socket.send_multipart([b'RESET_API', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def getVersion(self):
        """
        Output: a string containing the version of the API
        """
        self.socket.send_multipart([b'GET_VERSION', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode()
    
    def disable(self):
        """
        Output: None
        """
        self.socket.send_multipart([b'DISABLE', b"NOTNEEDED", b"NOTNEEDED"])
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def log(self, message):
        """
        Input: message - a string containing the message to log
        Output: None
        """
        self.socket.send_multipart([b'LOG', message.encode(), self.node_id.encode()])
        #self.socket.send(b"LOG")
        response = self.socket.recv()
        return response.decode() == "OK"
    
    def __del__(self):
        self.socket.close()
        self.context.term()
