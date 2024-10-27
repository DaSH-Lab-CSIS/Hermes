from APIFramework import APIFramework
import zmq
import json

class API:
    def __init__(self):
        self.api_framework = APIFramework()
        self.disable = False
        
        self.version = "1.0.0"
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        
        with open('../conf/master.json') as f:
            self.master_config = json.load(f)
            
        connection = "tcp://" + self.master_config["master"]["ip"] + ":5555" 
        self.socket.bind(connection)
        
        self.api_calls = 0
        
    def setup(self):
        while True:
            if(self.disable):
                break
            action, data1, data2 = self.socket.recv_multipart()
            if action == b'GET_NODE_TRAINING': 
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isNodeTraining(node_id)).encode()
            elif action == b'SET_NODE_TRAINING': 
                self.api_calls += 1
                node_id, value = data1.decode(), data2.decode()
                self.api_framework.setNodeTraining(node_id, value == "True")
                response = b'OK'
            elif action == b'GET_NODE_MEM':
                self.api_calls += 1
                node_id = data1.decode()
                response = json.dumps(self.api_framework.getNodeMemory(node_id)).encode()
            elif action == b'SET_NODE_MEM':
                self.api_calls += 1
                node_id, mem = data1.decode(), data2.decode()
                self.api_framework.setNodeMemory(node_id, mem)
                response = b'OK'
            elif action == b'IS_NODE_MEM_SET':
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isNodeMemorySet(node_id)).encode()
            elif action == b'IS_NEEDS_DATA':
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isNeedsData(node_id)).encode()
            elif action == b'SET_NEEDS_DATA':
                self.api_calls += 1
                node_id, value = data1.decode(), data2.decode()
                self.api_framework.setNeedsData(node_id, value == "True")
                response = b'OK'
            elif action == b'IS_TRAINING_COMPLETED': 
                self.api_calls += 1
                response = str(self.api_framework.isTrainingCompleted()).encode()
            elif action == b'SET_TRAINING_COMPLETED': 
                self.api_calls += 1
                value = data1.decode()
                self.api_framework.setTrainingCompleted(value == "True")
                response = b'OK'
            elif action == b'SET_BATCH_SIZE':
                self.api_calls += 1
                node_id, batch_size = data1.decode(), data2.decode()
                self.api_framework.setBatchSize(node_id, batch_size)
                response = b'OK'
            elif action == b'GET_BATCH_SIZE':
                self.api_calls += 1
                node_id = data1.decode()
                response = json.dumps(self.api_framework.getBatchSize(node_id)).encode()
            elif action == b'IS_BATCH_SIZE_SET':
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isBatchSizeSet(node_id)).encode()
            elif action == b'SET_MINI_BATCH_SIZE':
                self.api_calls += 1
                node_id, batch_size = data1.decode(), data2.decode()
                self.api_framework.setMiniBatchSize(node_id, batch_size)
                response = b'OK'
            elif action == b'GET_MINI_BATCH_SIZE':
                self.api_calls += 1
                node_id = data1.decode()
                response = json.dumps(self.api_framework.getMiniBatchSize(node_id)).encode()
            elif action == b'IS_MINI_BATCH_SIZE_SET':
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isMiniBatchSizeSet(node_id)).encode()
            elif action == b'RESET_API': 
                self.api_calls += 1
                self.api_framework.resetAPI()
                response = b'OK'
            elif action == b'GET_VERSION': 
                self.api_calls += 1
                response = self.version.encode()
            elif action == b'DISABLE': 
                self.api_calls += 1
                self.disable = True
                self.archiveLastLogs()
                response = b'OK'
            elif action == b'LOG': 
                self.api_calls += 1
                message, node_id = data1.decode(), data2.decode()
                self.api_framework.centralizedLogger(message, node_id)
                response = b'OK'
            elif action == b'GET_GRADIENTS': 
                self.api_calls += 1
                response = json.dumps(self.api_framework.getGradients()).encode()
            elif action == b'SET_GRADIENTS': 
                self.api_calls += 1
                node_id, gradients = data1.decode(), data2.decode()
                self.api_framework.setGradients(node_id, gradients)
                response = b'OK'
            elif action == b'RESET_GRADIENTS': 
                self.api_calls += 1
                self.api_framework.resetGradients()
                response = b'OK'
            elif action == b'RESET_NODE_GRADIENTS':
                self.api_calls += 1
                node_id = data1.decode()
                self.api_framework.resetNodeGradients(node_id)
                response = b'OK'
            elif action == b'IS_GRADIENT_PRESENT': 
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isGradPresent(node_id)).encode()
            elif action == b'SET_STOP_REACHED':
                self.api_calls += 1
                value = data1.decode()
                self.api_framework.setStopReached(value == "True")
                response = b'OK'
            elif action == b'IS_STOP_REACHED':
                self.api_calls += 1
                response = str(self.api_framework.isStopReached()).encode()
            elif action == b'IS_MODEL_PRESENT':
                self.api_calls += 1
                node_id = data1.decode()
                response = str(self.api_framework.isModelPresent(node_id)).encode()
            elif action == b'SET_MODEL_PRESENT':
                self.api_calls += 1
                node_id, value = data1.decode(), data2.decode()
                self.api_framework.setModelPresent(node_id, value == "True")
            elif action == b'SET_TRAINING_TIME':
                self.api_calls += 1
                node_id, time = data1.decode(), data2.decode()
                self.api_framework.setTrainingTime(node_id, time)
                response = b'OK'
            elif action == b'GET_TRAINING_TIME':
                self.api_calls += 1
                response = json.dumps(self.api_framework.getTrainingTime()).encode()
            elif action == b'SET_UPDATE':
                self.api_calls += 1
                node_id = data1.decode()
                self.api_framework.setUPDATE(node_id)
                response = b'OK'
            elif action == b'GET_UPDATE':
                self.api_calls += 1
                node_id = data1.decode()
                response = json.dumps(self.api_framework.getUPDATE(node_id)).encode()
            elif action == b'REMOVE_UPDATE':
                self.api_calls += 1
                node_id = data1.decode()
                self.api_framework.resetUPDATE(node_id)
                response = b'OK'
            else:
                response = b'INVALID ACTION'
                
            self.socket.send(response)
        
    def disable(self):
        self.disable = True
        
    def archiveLastLogs(self):
        self.api_framework.centralizedLogger("Number of API calls: " + str(self.api_calls), "API")
        self.api_framework.archiveLastLogs()

    def __del__(self):
        self.socket.close()
        self.context.term()
  
