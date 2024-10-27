import tensorflow as tf
import time
import numpy as np


class TrainerWorker:
    def __init__(self, logger, data_manager):
        self.logger = logger
        self.data_manager = data_manager
        self.sum_grads = None
        self.loss = None
        pass
        
    def train(self, model, batch_size):        
        self.logger.log("Training model...")
        data = self.data_manager.getBatch()
        
        x_train = data['x'][1:, :]
        y_train = data['y'][1:, :]
            
        optimizer = model.optimizer
        startTime = time.time()
        
        for _ in range(5):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size] 
                
                with tf.GradientTape() as tape:
                    y_pred = model(x_batch, training=True) 
                    y_pred = tf.convert_to_tensor(y_pred)
                    y_batch = tf.convert_to_tensor(y_batch)
                        
                    loss = model.compiled_loss(y_batch, y_pred) 
                        
                grads = tape.gradient(loss, model.trainable_variables)
                if self.sum_grads == None:
                    self.sum_grads = grads
                else:
                    self.sum_grads = [self.sum_grads[i] + grads[i] for i in range(len(grads))]
                optimizer.apply_gradients(zip(grads, model.trainable_variables))    
            last_loss = loss.numpy()
            print("[local-Epoch] Train Loss: " + str(loss.numpy()))
        print("[local-SGD] Train Loss: " + str(last_loss))
        endTime = time.time() - startTime
        self.logger.log("Training time: " + str(endTime))
        self.logger.log("Training complete! Loss: " + str(loss.numpy())) 
        
        # Testing time!                
        y_pred = model(self.data_manager.x_test)
        loss_fn = tf.keras.losses.get(model.loss)
        self.loss = np.mean(loss_fn(self.data_manager.y_test, y_pred).numpy()) 
        self.data_manager.resetBatch()
        self.logger.log("Testing complete! Loss: " + str(self.loss))  
        
        return endTime  

    def get_sum_grads(self):
        return self.sum_grads
    
    def evaluate(self, model, data):
        return model.evaluate(data)