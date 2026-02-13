import tensorflow as tf
from custom_layers import x
from base_model import pre_trained_model

basemodel = pre_trained_model()

model = tf.keras.Model(inputs = basemodel.input, outputs = x)

# model compiling
def model_compile():
    model.compile(
        optimizer = 'RMSprop',
        loss = 'binary_crossentropy',
        metrics = ['accuracy']
    )

if __name__ == '__main__':
    model_compile()
    print(f'\n Model successfully compiled \n')

