import numpy as np

class DataManager:
    def __init__(self, logger, x_df, y_df, settings, model, communication, api_master):
        self.logger = logger
        self.x_train = x_df
        self.y_train = y_df
        self.settings = settings
        self.current_row = 0
        self.size = x_df.shape[0]
        self.model = model
        self.communication = communication
        self.api_master = api_master
        
    def reset_data(self):
        self.current_row = 0
        perm = np.random.permutation(len(self.x_train))
        
        return self.x_train[perm], self.y_train[perm]
            
    def check_if_next_batch_valid_async(self):
        return self.current_row < self.size
    
    def get_next_batch_async(self, node_id):
        x_train, y_train = self.reset_data()
        endRow = self.settings.get_batch_node(node_id)
        
        if self.current_row >= self.size:
            self.logger.localLogger("No more data to send to node " + node_id + "!")
            self.logger.manuallyCentralizeLogger("No more data to send to node " + node_id + "!")
            return None
        
        if endRow > self.size:
            endRow = self.size
            
        batch = x_train[:endRow]
        y_batch = y_train[:endRow]
        
        self.logger.localLogger("Batch sent to node " + node_id + " with " + str(len(batch)) + " rows!")
        
        self.logger.manuallyCentralizeLogger("Batch sent to node " + node_id + " with " + str(len(batch)) + " rows!")
        
        return {"x_train": batch, "y_train": y_batch}
    
    def send_model(self, node_id, model):
        self.communication.sendModel(node_id, model)
    
    def send_batch_node_async(self, node_id):
        protocol = self.settings.get_protocol()
        df = self.get_next_batch_async(node_id)
        self.communication.sendDataFrame(node_id, df, protocol)
 
