import pdb
from tensorflow.keras.applications.resnet50 import ResNet50
from dataloader import dataloader
from keras.layers import Flatten, Input
from keras.models import Model
import keras
import ssl
import tensorflow as tf
from tensorflow.python.client import device_lib 
#from positional_embeddings import calc_positional_embeddings


ssl._create_default_https_context = ssl._create_unverified_context



def build_resnet():
    input_shape = (50,200,3)
    train_generator, valid_generator, test_generator = dataloader()
    inputs = keras.Input(shape=input_shape)
    
    base_model = ResNet50(weights='imagenet', pooling=max, include_top = False)

    inp = base_model.layers[0].input
    output = base_model.layers[-2].output

    model = Model(inputs=inp, outputs=output)
    features = model.predict(train_generator)
    features_reduce = features.squeeze()
    length, width, feature = features_reduce.shape[1::]
    resnet_embeddings = tf.reshape(features_reduce,(-1,length*feature,width))
    
    
    fc1 = tf.keras.models.Sequential()
    fc1.add(tf.keras.Input(shape=resnet_embeddings.transpose(0,2,1).shape[-2::]))
    fc1.add(tf.keras.layers.Dense(feature))
    print(fc1.output_shape)

    fc1_output = fc1(resnet_embeddings.transpose(0,2,1))
    #pos_embeddings = calc_positional_embeddings(feature,width)
    fc_bar = fc1_output + pos_embeddings 

    fc2 = tf.keras.models.Sequential()
    fc2.add(tf.keras.Input(shape=fc_bar.shape[-2::]))
    fc2.add(tf.keras.layers.Dense(feature))

    fc2_output = fc2(fc_bar)
    fc2_output = tf.transpose(fc2_output, perm = [0,2,1])

    return fc2_output
    
def main():
    build_resnet()

if __name__ == '__main__':
    main()
