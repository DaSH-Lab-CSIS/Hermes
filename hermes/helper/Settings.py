import json

class Settings:
    def __init__(self, config_file, node_registry):
        with open(config_file) as f:
            self.settings = json.load(f)
            
        self.node_registry = node_registry
        self.batch_nodes = {} # API ONLY
        
    def get_batch_node(self, node_id):
        if(self.get_batch_size() == "simple"):
            return self.get_batch_size()
        elif(self.get_batch_size() == "custom"):
            return self.get_custom_batch_size()[node_id]
        else:
            return self.get_api_batch_size()[node_id]
    
    # only if get_batch_type() == "simple"
    def get_batch_size(self):
        return self.settings["batch_size"]
    
    # only if get_batch_type() == "api"
    def get_api_batch_size(self):
        return self.batch_nodes
    
    # only if get_batch_type() == "custom"
    def get_custom_batch_size(self):
        node_batches = {}
        for node in self.node_registry.get_all_node_ids():
            node_batches[node] = self.settings["batches"][node]
            
        return node_batches
    
    def get_protocol(self):
        return self.settings["protocol"]
    
    def get_training_type(self):
        return self.settings["experiment"]
    
    def get_alpha(self):
        return self.settings["alpha"]
    
    def get_lambda(self):
        return self.settings["lambda"]
    
    def get_beta(self):
        return self.settings["beta"]
    
    def get_window_size(self):
        return self.settings["window_size"]
 
