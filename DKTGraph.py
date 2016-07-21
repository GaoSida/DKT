import tensorflow as tf

class DKTGraph(object):
    def get_optimizer(self, name):
        if name == "adam":
            return tf.train.AdamOptimizer()
        elif name == "adadelta":
            return tf.train.AdadeltaOptimizer()

    def __init__(self, hps):
        batch_size = hps.batch_size
        num_hidden = hps.num_hidden
        num_actions = hps.num_actions
        num_skills = hps.num_skills
        init_mean = hps.init_mean
        init_stddev = hps.init_stddev
        clipping_norm = hps.clipping_norm
        dropout_keep = hps.dropout_keep
        optimizer_name = hps.optimizer

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Parameters
            # Input gate
            input_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='input_W')
            input_b = tf.Variable(tf.zeros([1, num_hidden]), name='input_b')
            # Forget gate
            forget_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='forget_W')
            forget_b = tf.Variable(tf.zeros([1, num_hidden]), name='forget_b')
            # Update cell:                             
            update_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='update_W')
            update_b = tf.Variable(tf.zeros([1, num_hidden]), name='update_b')
            # Output gate:
            output_W = tf.Variable(tf.truncated_normal([num_actions + num_hidden, num_hidden], init_mean, init_stddev), name='output_W')
            output_b = tf.Variable(tf.zeros([1, num_hidden]), name='output_b')
            # initial output and state
            initial_output = tf.Variable(tf.truncated_normal([1, num_hidden], init_mean, init_stddev), name='initial_output')
            initial_state = tf.Variable(tf.truncated_normal([1, num_hidden], init_mean, init_stddev), name='initial_state')
            # Classifier weights and biases.
            classify_W = tf.Variable(tf.truncated_normal([num_hidden, num_skills], init_mean, init_stddev), name='classify_W')
            classify_b = tf.Variable(tf.zeros([num_skills]), name='classify_b')
  
            def lstm_cell(i, o, state):
                x = tf.concat(1, [i, o])
                input_gate = tf.sigmoid(tf.matmul(x, input_W) + input_b)
                forget_gate = tf.sigmoid(tf.matmul(x, forget_W) + forget_b)
                update = tf.tanh(tf.matmul(x, update_W) + update_b)
                state = forget_gate * state + input_gate * update
                output_gate = tf.sigmoid(tf.matmul(x, output_W) + output_b)
                return output_gate * tf.tanh(state), state

            # Input data.
            inputs = list()
            skill_labels = list()
            result_labels = list()
            maxlen = tf.placeholder(tf.int32)
            for _ in range(tf.to_int32(maxlen)):
                inputs.append(tf.placeholder(tf.float32, shape=[batch_size, num_actions]))
                skill_labels.append(tf.placeholder(tf.float32, shape=[batch_size, num_skills]))
                result_labels.append(tf.placeholder(tf.float32, shape=[batch_size, ]))
            
            # Forward propagate.
            outputs = list()
            output = tf.matmul(tf.ones([batch_size, 1]), initial_output)
            state = tf.matmul(tf.ones([batch_size, 1]), initial_state)
            outputs.append(output)
            for i in inputs[:-1]:
                output, state = lstm_cell(i, output, state)
                outputs.append(output)

            # prediction and loss
            logits = tf.matmul(tf.dropout(tf.concat(0, outputs), dropout_keep), classify_W) + classify_b
            logits_of_interest = tf.reduce_sum(tf.mul(logits, tf.concat(0, skill_labels)), 1)
            prediction = tf.sigmoid(logits_of_interest)
            truth = tf.reshape(tf.concat(0, result_labels), [-1])
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits_of_interest, truth))
    
            optimizer = self.get_optimizer(optimizer_name)
            gradients, var = zip(*optimizer.compute_gradients(loss))
            gradients, _ = tf.clip_by_global_norm(gradients, clipping_norm)
            optimizer = optimizer.apply_gradients(zip(gradients, var))
    
            test_outputs = list()
            test_output = initial_output
            test_state = initial_state
            test_outputs.append(test_output)
            for i in inputs[:-1]:
                test_output, test_state = lstm_cell(i, test_output, test_state)
                test_outputs.append(test_output)

            test_logits = tf.matmul(tf.concat(0, test_outputs), classify_W) + classify_b
            test_logits_of_interest = tf.reduce_sum(tf.mul(test_logits, tf.concat(0, question_labels)), 1)
    
            test_status = tf.sigmoid(test_logits)
            test_prediction = tf.sigmoid(test_logits_of_interest)
    
            saver = tf.train.Saver()     # To save models




