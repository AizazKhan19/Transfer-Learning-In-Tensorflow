import tensorflow as tf


# I am using InceptionV3 model as my base model
# initilizing base model
def pre_trained_model():
    Base_Model = tf.keras.applications.inception_v3.InceptionV3(
        include_top = False,
        input_shape = (150, 150, 3),
        weights = 'imagenet'
    )

    Base_Model.trainable = False

    return Base_Model

if __name__ == '__main__':
    basemodel=pre_trained_model()
    print('Base Model Successfully Initilized')