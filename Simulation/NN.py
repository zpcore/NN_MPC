from __future__ import division
import argparse
import numpy as np
import pickle
import tensorflow as tf
from data_construct import data_construct
import data_gen
import matplotlib.pyplot as plt



# Create model
def multilayer_perceptron(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	# Hidden layer with RELU activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
	layer_2 = tf.nn.relu(layer_2)
	# Output layer with linear activation
	out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
	return out_layer

def main():
	# Parameters
	learning_rate = 0.001
	training_epochs = 30
	batch_size = 10#100
	display_step = 1

	#training data parameters
	N = 4 # number of states (x)
	M = 2 # number of inputs (u)
	P = 2 # number of sensor signals (y)
	Hp = 10 # prediction horizon
	Hm = 10 # input horizon
	mem_horizon = 3 # time stampes of sensor signal as input to NN. (y(k-Hp)...y(k-Hp-horizon+1))

	# Network Parameters
	n_hidden_1 = 4 # 1st layer number of features
	n_hidden_2 = 4 # 2nd layer number of features
	n_input =  M*Hm+P*mem_horizon # data input
	n_output=	1

	parser = argparse.ArgumentParser(description='Number of Testing Samples')
	parser.add_argument('-n', action="store", dest="ts", type=int)
	sim_len = parser.parse_args().ts

	# tf Graph input
	x = tf.placeholder("float", [None, n_input])
	y = tf.placeholder("float", [None, n_output])
	# Store layers weight & bias
	weights = {
		'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
		'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
	}
	biases = {
		'b1': tf.Variable(tf.random_normal([n_hidden_1])),
		'b2': tf.Variable(tf.random_normal([n_hidden_2])),
		'out': tf.Variable(tf.random_normal([n_output]))
	}

	# Construct model
	pred = multilayer_perceptron(x, weights, biases)
	# Define loss and optimizer
	#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	cost = tf.reduce_mean(tf.nn.l2_loss(t=pred-y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	# Initializing the variables
	init = tf.global_variables_initializer()

	with open(r'./traindata.log', 'rb') as _load_file:
		train_data = pickle.load(_load_file)

	with open(r'./testdata.log', 'rb') as _load_file2:
		test_data = pickle.load(_load_file2)

	dc=data_construct(predict_horizon,train_data)
	tc=data_construct(predict_horizon,test_data)

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		# Training cycle
		for epoch in range(training_epochs):
				avg_cost = 0.
				total_batch = int((dc.num_example-predict_horizon+1)/batch_size)
				# Loop over all batches
				for i in range(total_batch):
						batch_x=[]
						batch_y=[]
						batch_x, batch_y = dc.next_batch(batch_x,batch_y,batch_size)
						# Run optimization op (backprop) and cost op (to get loss value)
						_, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})
						avg_cost += c / total_batch
				# Display logs per epoch step
				if epoch % display_step == 0:
					print "Epoch:", '%04d' % (epoch+1), "cost=", \
						"{:.9f}".format(avg_cost)
				dc.clr_count()

		print "Optimization Finished!"
		
		print "Testing the Neural Network with ",sim_len," steps..."
		# batch_x=[]
		# batch_y=[]
		# #for _ in range(100):
		# batch_x, batch_y = tc.next_batch(batch_x,batch_y,100)
		# testing_cost = sess.run(cost, feed_dict={x: batch_x, y: batch_y})/100
		# print "L2 cost per batch:",testing_cost
		x_plot=[]
		y_plot=[]
		x_plot,y_plot=tc.plot_data(x_plot,y_plot,sim_len)


		test_x=[]
		test_y=[]
		#prediction=[]	
		test_x, batch_y = tc.next_batch(test_x,test_y,sim_len)
		prediction=sess.run(pred,feed_dict={x:test_x})
		
		f, axarr = plt.subplots(2, sharex=True, sharey=True)
		axarr[0].plot(np.arange(sim_len), y_plot)
		axarr[0].set_title('True Result')
		axarr[1].plot(np.arange(sim_len), prediction)
		axarr[1].set_title('Prediction Result')
		plt.show()

		#Save the trained neural network into a file
		#saver = tf.train.Saver()
		#saver.save(sess, "NN.log")






if __name__ == "__main__":
    main()