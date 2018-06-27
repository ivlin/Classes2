import sys
#ams math
import numpy as np
import tensorflow as tf
#visualization
import seaborn as sb
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
sb.set()

BATCH_SIZE=100
NUM_FEATURES=2
D_TO_G_STEPS=20 #k - ratio of discriminator to generator optimization steps
GENERATOR_LAYERS=[10,10,10]
DISCRIMINATOR_LAYERS=[10,10,10]
NUM_ITERATIONS=5000

help_clause="""USAGE: python my_gan.py [--generator-layers (int list)] [--discriminator (int list)] [--batch-size (int)] [--features (int)] [--k-ratio (int)]
        --generator-layers          takes a python-formatted list of integers representing the width of each hidden layer
        --discriminator-layers      takes a python-formatted list of integers representing the width of each hidden layer
        --batch-size                takes an integer representing the batch size
        --k-ratio                   takes an integer representing the ratio of discriminator updates to generator updates
        --features                  takes an integer representing the number of features (visualization not available for more than 2 features)
        --iterations                takes an integer representing the number of iterations
"""


# ARG PARSING
def load_params(arglist):
    global GENERATOR_LAYERS, DISCRIMINATOR_LAYERS, BATCH_SIZE, NUM_FEATURES, D_TO_G_STEPS
    for i in xrange(len(arglist)):
        if arglist[i]=="--help" or arglist[i]=="-h":
            print help_clause
        if arglist[i]=="--generator-layers":
            GENERATOR_LAYERS=[int(num) for num in arglist[i+1][1:-1].split(",")]
        if arglist[i]=="--discriminator-layers":
            DISCRIMINATOR_LAYERS=[int(num) for num in arglist[i+1][1:-1].split(",")]
        if arglist[i]=="--batch-size":
            BATCH_SIZE=int(arglist[i+1])
        if arglist[i]=="--k-ratio":
            D_TO_G_STEPS=int(arglist[i+1])
        if arglist[i]=="--features":
            NUM_FEATURES=int(arglist[i+1])
        if arglist[i]=="--iterations":
            NUM_ITERATIONS=int(arglist[i+1])
#PART 1: DATA GENERATION

def generate_true_data(num_samples):
    sample_input=np.random.normal(-1.0,1.0,(num_samples, NUM_FEATURES))
    for i in sample_input:
        i[1]=i[0]*i[0]
    return sample_input

def generate_uniform_data(num_samples):
    return np.random.uniform(-1.0,1.0,size=[num_samples, NUM_FEATURES])

#PART 2: GENERATOR AND DISCRIMINATOR

def generator(input_data, layer_sizes, reuse=False):
    #create shared variables that can be reaccessed later
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        layers=[]
        for layer_size in layer_sizes:
            #tf.layers defines the interface for a densely defined layer that performs activation(input*kernel+bias)
            #returns input_data with the last dimension of size units
            layers.append(tf.layers.dense(inputs=input_data,\
                units=layer_size,\
                activation=tf.nn.leaky_relu))
        output=tf.layers.dense(layers[-1], NUM_FEATURES)
    return output

def discriminator(input_data, layer_sizes, reuse=False):
    #create shared variables that can be reaccessed later
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        layers=[]
        for layer_size in layer_sizes:
            #tf.layers defines the interface for a densely defined layer that performs activation(input*kernel+bias)
            #returns input_data with the last dimension of size units
            layers.append(tf.layers.dense(inputs=input_data,\
                units=layer_size,\
                activation=tf.nn.leaky_relu))
        output=tf.layers.dense(layers[-1], NUM_FEATURES)
    return output

if __name__=="__main__":
    ####################
    # LOAD PARAMS FROM USER INPUT
    ####################
    load_params(sys.argv)

    np.random.seed(0)
    tf.set_random_seed(0)
    ####################
    # VARIABLE INTIALIZATION STEP
    ####################
    #use placeholders to simulate operations on net input
    noise=tf.placeholder(tf.float32, [None, NUM_FEATURES])
    train_data=tf.placeholder(tf.float32, [None, NUM_FEATURES])
    #call the parameters that are to be optimized - the shared variables defined in gen and disc
    generator_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    discriminator_variables= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    ####################
    # FORWARD PROP
    ####################
    #generate data from generator
    generated_data=generator( noise, GENERATOR_LAYERS )
    #run discriminator
    generated_guesses=discriminator( generated_data, DISCRIMINATOR_LAYERS )
    #set reuse to be true since we're using same net - not updating between runs
    real_guesses=discriminator( train_data, DISCRIMINATOR_LAYERS, True)

    ####################
    # LOSS CALCULATION
    ####################
    #calculate losses - currently sigmoid cross entropy with logits
    #1->real data, 0->fake data
    sample_losses=tf.nn.sigmoid_cross_entropy_with_logits(logits=real_guesses,labels=tf.ones_like(real_guesses))\
                + tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_guesses,labels=tf.zeros_like(generated_guesses))
    discriminator_loss=tf.reduce_mean(sample_losses)
    generator_loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=generated_guesses,labels=tf.ones_like(generated_guesses)))

    ####################
    # BACKPROP + UPDATE STEP
    ####################
    #Call shared variables defined in layer calculations
    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

    #Update rule using rmsprop
    gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(generator_loss,var_list = gen_vars)
    disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(discriminator_loss,var_list = disc_vars)

    ####################
    # TENSORFLOW SESSION INITIALIZATION
    ####################
    init = tf.global_variables_initializer()#create global var initializer
    sess=tf.Session()                       #create new session for operation
    sess.run(init)                          #run the initializer

    ####################
    # INITIALIZE LOGGING
    ####################
    f = open('loss_logs.csv','w')
    f.write('Iteration,Discriminator Loss,Generator Loss\n')

    ####################
    # TRAINING
    ####################
    iterations=NUM_ITERATIONS
    #run the training
    for it in xrange(iterations):
        true_data=generate_true_data(BATCH_SIZE)
        uniform_noise=generate_uniform_data(BATCH_SIZE)
        #train discriminator for k steps
        for i in xrange(D_TO_G_STEPS):
            #1st argument: fetch: runs necessary graph fragments to generate each tensor in fetch
            d_loss, ds=sess.run([discriminator_loss,disc_step],feed_dict={noise:uniform_noise, \
                train_data:true_data})
            #print d_loss

        #train generator
        true_data=generate_true_data(BATCH_SIZE)
        d_loss, ds, g_data, g_loss, gs=sess.run([discriminator_loss,disc_step,generated_data,generator_loss,gen_step],\
            feed_dict={noise:generate_uniform_data(BATCH_SIZE), train_data:true_data})
        print it, " iteration: discriminator losss:", d_loss, " generator loss:", g_loss

        if it%10 == 0:
            f.write("%d,%f,%f\n"%(it,d_loss,g_loss))
        if it%100 == 0 and NUM_FEATURES==2:
            plt.figure()

            xax = plt.scatter(true_data[:,0], true_data[:,1],color="r")
            gax = plt.scatter(g_data[:,0], g_data[:,1],color="b")

            plt.legend((xax,gax), ("Real Data","Generated Data"))
            #plt.legend((xax), ("Real Data"))
            plt.title('Samples at Iteration %d'%it)
            plt.tight_layout()
            plt.savefig('./iterations/iteration_%d.png'%it)
            plt.close()
    f.close()