import datetime
import os

class Logging:
    def __init__(self, api_node, node_id):
        self.api = api_node
        self.node_id = node_id
        
        # print(os.getcwd())
    
    def localLogger(self, message):
        logMessage = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + self.node_id + "] - " + message
        with open("../data/device-logs/local-logs.txt", "a") as f:
            f.write(logMessage + "\n")

    def manuallyCentralizeLogger(self, message):
        logMessage = "[" + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + " - " + self.node_id + "] - " + message
        with open("../data/centralized-logs/distml-central.txt", "a") as f:
            f.write(logMessage + "\n")
     
    def log(self, message):
        self.localLogger(message)
        self.api.log(message)

