from htnet_model import htnet
import tensorflow as tf
from keras import backend as K


K.set_image_data_format('channels_first')
tf.config.experimental_run_functions_eagerly(True)


X_train = tf.random.uniform(shape=[63, 1, 9, 500])
Y_train = tf.random.uniform(shape=[63, 1]) > tf.constant(0.5)
Y_train = tf.concat([Y_train, ~Y_train], axis=1)

X_test = tf.random.uniform(shape=[31, 1, 9, 500])
Y_test = tf.random.uniform(shape=[31, 1]) > tf.constant(0.5)
Y_test = tf.concat([Y_test, ~Y_test], axis=1)

model = htnet(nb_classes=2, Chans=9, Samples=50)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 1,
                                verbose = 2, validation_data=(X_test, Y_test))