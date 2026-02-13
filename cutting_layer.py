import tensorflow as tf
from base_model import pre_trained_model

base_model = pre_trained_model()

# cutting the layer 'mixed7' inception block
last_layer_of_base_model = base_model.get_layer('mixed7')
last_output = last_layer_of_base_model.output

def getting_layer():
    
    print(f'output of last layer of base model : {last_layer_of_base_model.output}')

if __name__ == '__main__':
    getting_layer()