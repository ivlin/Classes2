import tensorflow as tf

class Generator:
    def __init__(self):
        pass

    def generate(self,x,keep,is_training):
        with tf.variable_scope("GAN/Generator",reuse=False):
            i_flat=tf.nn.dropout(tf.layers.dense(x,units=49),keep,name="drop0")
            i1=tf.reshape(i_flat,[-1,7,7,1])

            conv1=tf.nn.dropout(tf.layers.conv2d_transpose(i1,filters=512,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="dconv1"), keep, name="drop4")#7x7
            n1=tf.layers.batch_normalization(conv1,training=is_training,name="batch_norm1")

            conv2=tf.nn.dropout(tf.layers.conv2d_transpose(n1,filters=128,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv2"), keep, name="drop4")#14
            n2=tf.layers.batch_normalization(conv2,training=is_training,name="batch_norm2")

            conv3=tf.nn.dropout(tf.layers.conv2d_transpose(n2,filters=32,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="dconv3"), keep, name="drop4")#14
            n3=tf.layers.batch_normalization(conv3,training=is_training,name="batch_norm3")

            conv4=tf.nn.dropout(tf.layers.conv2d_transpose(n3,filters=4,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv4"), keep, name="drop4")#28
            n4=tf.layers.batch_normalization(conv4,training=is_training,name="batch_norm4")

            out=tf.layers.conv2d_transpose(n4,filters=1,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="dconv5")#128x128
        return out

class Discriminator:
    def __init__(self):
        pass

    def classify(self,x,keep,is_training,reuse=False):
        with tf.variable_scope("GAN/Discriminator",reuse=reuse):
            a1=tf.nn.dropout(tf.layers.conv2d(x,filters=4,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv1"),keep,name="drop1")#28
            n1=tf.layers.batch_normalization(a1,training=is_training,name="batch_norm1")

            a2=tf.nn.dropout(tf.layers.conv2d(n1,filters=32,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv2"),keep,name="drop2")#28
            n2=tf.layers.batch_normalization(a2,training=is_training,name="batch_norm2")

            a3=tf.nn.dropout(tf.layers.conv2d(n2,filters=128,kernel_size=(3,3),strides=(1,1),padding="SAME",activation=tf.nn.leaky_relu,name="conv3"),keep,name="drop3")#14
            n3=tf.layers.batch_normalization(a3,training=is_training,name="batch_norm3")

            a4=tf.nn.dropout(tf.layers.conv2d(n3,filters=512,kernel_size=(3,3),strides=(2,2),padding="SAME",activation=tf.nn.leaky_relu,name="conv4"),keep,name="drop4")#14
            n4=tf.layers.batch_normalization(a4,training=is_training,name="batch_norm4")

            y=tf.layers.dense(tf.reshape(n4,[-1,512*7*7]),units=1,activation=tf.nn.leaky_relu, name="out")
        return y