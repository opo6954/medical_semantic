'''
basic class for famous network model
'''



class PreTrainedClassifyNet():
    def __init__(self, x_shape, y_shape, keep_prob, num_classes, skip_layer, batchSize, weights_path=''):
        self.x = x_shape
        self.y = y_shape
        self.keep_prob = keep_prob
        self.num_classes = num_classes
        self.skip_layer = skip_layer
        self.batchSize = batchSize
        self.weights_path = weights_path


    # build the basic graph of pre-trainedClassifyNet
    def build_basic_graph(self):
        pass

    def load_initial_weights(self):
        pass

    def train_graph(self):
        pass

    def design_loss(self):
        pass
