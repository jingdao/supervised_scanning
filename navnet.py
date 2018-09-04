import tensorflow as tf

class NavNet():
	def __init__(self,batch_size, img_size, frontier_size, pos_size, output_size, seq_len):
		self.image_pl = tf.placeholder(tf.float32, shape=(batch_size, seq_len, img_size, img_size))
		self.pos_pl = tf.placeholder(tf.float32, shape=(batch_size, seq_len, pos_size))
		self.frontier_pl = tf.placeholder(tf.float32, shape=(batch_size, seq_len, frontier_size, 2))
		self.label_pl = tf.placeholder(tf.float32, shape=(batch_size, seq_len, output_size))
		self.seq_pl = tf.placeholder(tf.int32, shape=(batch_size))

		self.conv1 = tf.reshape(self.image_pl, [batch_size*seq_len, img_size, img_size, 1])
		self.kernel1 = tf.get_variable('kernel1', [5,5,1,10], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias1 = tf.get_variable('bias1', [10], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv1 = tf.nn.conv2d(self.conv1, self.kernel1, [1,1,1,1], padding='VALID')
		self.conv1 = tf.nn.bias_add(self.conv1, self.bias1)
		self.conv1 = tf.nn.relu(self.conv1)
		self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='pool1')

		self.kernel2 = tf.get_variable('kernel2', [5,5,10,20], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias2 = tf.get_variable('bias2', [20], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv2 = tf.nn.conv2d(self.pool1, self.kernel2, [1,1,1,1], padding='VALID')
		self.conv2 = tf.nn.bias_add(self.conv2, self.bias2)
		self.conv2 = tf.nn.relu(self.conv2)
		self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='pool1')

		self.fc3 = tf.reshape(self.pool2, [batch_size, seq_len, 20*9*9])
		self.kernel3 = tf.get_variable('kernel3', [1,20*9*9,100], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias3 = tf.get_variable('bias3', [100], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc3 = tf.nn.conv1d(self.fc3, self.kernel3, 1, padding='VALID')
		self.fc3 = tf.nn.bias_add(self.fc3, self.bias3)
		self.fc3 = tf.nn.relu(self.fc3)

		self.conv4 = tf.reshape(self.frontier_pl, [batch_size*seq_len, frontier_size, 2])
		self.kernel4 = tf.get_variable('kernel4', [1,2,10], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias4 = tf.get_variable('bias4', [10], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv4 = tf.nn.conv1d(self.conv4, self.kernel4, 1, padding='VALID')
		self.conv4 = tf.nn.bias_add(self.conv4, self.bias4)
		self.conv4 = tf.nn.relu(self.conv4)

		self.kernel5 = tf.get_variable('kernel5', [1,10,20], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias5 = tf.get_variable('bias5', [20], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv5 = tf.nn.conv1d(self.conv4, self.kernel5, 1, padding='VALID')
		self.conv5 = tf.nn.bias_add(self.conv5, self.bias5)
		self.conv5 = tf.nn.relu(self.conv5)
		self.conv5 = tf.reduce_max(self.conv5, axis=1)
		self.conv5 = tf.reshape(self.conv5, [batch_size, seq_len, 20])

		self.fc6 = tf.concat(axis=2, values=[self.fc3, self.pos_pl, self.conv5])
		self.kernel6 = tf.get_variable('kernel6', [1,120 + pos_size,100], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias6 = tf.get_variable('bias6', [100], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc6 = tf.nn.conv1d(self.fc6, self.kernel6, 1, padding='VALID')
		self.fc6 = tf.nn.bias_add(self.fc6, self.bias6)
		self.fc6 = tf.nn.relu(self.fc6)

		self.fc7, _ = tf.nn.dynamic_rnn(tf.contrib.rnn.GRUCell(100), self.fc6, dtype=tf.float32, sequence_length = self.seq_pl)
		self.kernel7 = tf.get_variable('kernel7', [1,100,output_size], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias7 = tf.get_variable('bias7', [output_size], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc7 = tf.nn.conv1d(self.fc7, self.kernel7, 1, padding='VALID')
		self.output = tf.nn.bias_add(self.fc7, self.bias7)

		self.mask = tf.where(tf.cast(tf.sign(tf.reduce_max(tf.abs(self.label_pl), 2)), tf.bool))
		self.label_masked = tf.gather_nd(self.label_pl, self.mask)
		self.output_masked = tf.gather_nd(self.output, self.mask)
		self.loss = tf.losses.huber_loss(self.label_masked, self.output_masked)

		self.batch = tf.Variable(0)
		self.learning_rate = tf.train.exponential_decay(0.001,self.batch,3000,0.5,staircase=True)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)

