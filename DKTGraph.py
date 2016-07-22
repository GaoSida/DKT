# coding:utf-8
import tensorflow as tf

class DKTGraph(object):
    def get_optimizer(self, hps):
        name = hps.optimizer
        if name == "adam":
            return tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        elif name == "adadelta":
            return tf.train.AdadeltaOptimizer()

    def standard_lstm_cell(self, output_state_tuple, i):
        o, state = tf.unpack(output_state_tuple)
        x = tf.concat(1, [i, o])
        input_gate = tf.sigmoid(tf.matmul(x, self.input_W) + self.input_b)
        forget_gate = tf.sigmoid(tf.matmul(x, self.forget_W) + self.forget_b)
        update = tf.tanh(tf.matmul(x, self.update_W) + self.update_b)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(x, self.output_W) + self.output_b)
        return tf.pack([output_gate * tf.tanh(state), state])

    def peephole_lstm_cell(self, output_state_tuple, i):
        o, state = tf.unpack(output_state_tuple)
        x = tf.concat(1, [i, o])
        input_gate = tf.sigmoid(tf.matmul(x, self.input_W) + tf.matmul(state, self.input_Peep) + self.input_b)
        forget_gate = tf.sigmoid(tf.matmul(x, self.forget_W) + tf.matmul(state, self.forget_Peep) + self.forget_b)
        update = tf.tanh(tf.matmul(x, self.update_W) + self.update_b)
        state = forget_gate * state + input_gate * update
        output_gate = tf.sigmoid(tf.matmul(x, self.output_W) + tf.matmul(state, self.forget_Peep) + self.output_b)
        return tf.pack([output_gate * tf.tanh(state), state])

    def get_lstm_cell(self, hps):
        if hps.lstm_cell == 'standard':
            return self.standard_lstm_cell
        elif hps.lstm_cell == 'peephole':
            return self.peephole_lstm_cell

    def __init__(self, hps, data_generator):
        batch_size = hps.batch_size
        num_hidden = hps.num_hidden
        num_actions = hps.num_actions
        num_skills = hps.num_skills
        init_mean = hps.init_mean
        init_stddev = hps.init_stddev
        clipping_norm = hps.clipping_norm
        dropout_keep = hps.dropout_keep

        self.graph = tf.Graph()
        with self.graph.as_default():
            tf.set_random_seed(1234)        # fix seed
            # Parameters
            # Input gate
            self.input_Peep = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], init_mean, init_stddev), name='input_Peep')
            self.input_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='input_W')
            self.input_b = tf.Variable(tf.zeros([1, num_hidden]), name='input_b')
            # Forget gate
            self.forget_Peep = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], init_mean, init_stddev), name='forget_Peep')
            self.forget_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='forget_W')
            self.forget_b = tf.Variable(tf.zeros([1, num_hidden]), name='forget_b')
            # Update cell:                             
            self.update_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='update_W')
            self.update_b = tf.Variable(tf.zeros([1, num_hidden]), name='update_b')
            # Output gate:
            self.output_Peep = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], init_mean, init_stddev), name='output_Peep')
            self.output_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='output_W')
            self.output_b = tf.Variable(tf.zeros([1, num_hidden]), name='output_b')
            # initial output and state
            initial_output = tf.Variable(tf.truncated_normal([1, num_hidden], init_mean, init_stddev), name='initial_output')
            initial_state = tf.Variable(tf.truncated_normal([1, num_hidden], init_mean, init_stddev), name='initial_state')
            # Classifier weights and biases.
            classify_W = tf.Variable(tf.truncated_normal([num_hidden, num_skills], init_mean, init_stddev), name='classify_W')
            classify_b = tf.Variable(tf.zeros([num_skills]), name='classify_b')

            # Input data.
            self.inputs = tf.placeholder(tf.float32, shape=[None, batch_size, num_actions])
            self.skill_labels = tf.placeholder(tf.float32, shape=[None, batch_size, num_skills])
            self.result_labels = tf.placeholder(tf.float32, shape=[None, batch_size])
            
            # Forward propagate.
            output_state_tuples = tf.scan(self.get_lstm_cell(hps), self.inputs,
                                          initializer=tf.pack([tf.matmul(tf.ones([batch_size, 1]), initial_output), 
                                                               tf.matmul(tf.ones([batch_size, 1]), initial_state)]))
            outputs = output_state_tuples[:, 0, :, :]    # take out outputs
            all_outputs = tf.concat(0, [[tf.matmul(tf.ones([batch_size, 1]), initial_output)], outputs])

            # prediction and loss
            logits = tf.matmul(tf.nn.dropout(tf.reshape(all_outputs, [-1, num_hidden]), dropout_keep), classify_W) + classify_b
            all_skill_labels = tf.reshape(self.skill_labels, [-1, num_skills])
            logits_of_interest = tf.reduce_sum(tf.mul(logits, all_skill_labels), 1)
            
            self.prediction = tf.sigmoid(logits_of_interest)
            truth = tf.reshape(self.result_labels, [-1])
            # mask out irrelevant losses
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits_of_interest, truth)
            mask = tf.reduce_sum(all_skill_labels, 1)
            self.loss = tf.reduce_sum(cross_entropy * mask) / tf.reduce_sum(all_skill_labels)
            
    
            self.optimizer = self.get_optimizer(hps)
            gradients, var = zip(*self.optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clipping_norm)
            self.optimizer = self.optimizer.apply_gradients(zip(gradients, var))

            # Testing
            test_logits = tf.matmul(tf.reshape(all_outputs, [-1, num_hidden]), classify_W) + classify_b
            test_logits_of_interest = tf.reduce_sum(tf.mul(test_logits, tf.reshape(self.skill_labels, [-1, num_skills])), 1)
    
            self.test_status = tf.sigmoid(test_logits)
            self.test_prediction = tf.sigmoid(test_logits_of_interest)
    
            self.saver = tf.train.Saver()     # To save models


