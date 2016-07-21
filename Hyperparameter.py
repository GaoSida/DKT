class Hyperparameter(object):
    def __init__(self, dataset_file, split_file,
                 skill_cut=150, student_cut=5000, padding_scheme='end_0',
                 num_hidden = 200, init_mean=0, init_stddev=0.001, batch_size=100,
                 clipping_norm=1.0, dropout_keep=1.0, optimizer='adam',
                 num_epochs=100, test_frequency=1, save_frequency=10):
        
        self.dataset_file = dataset_file
        self.split_file = split_file

        # control dataset size
        self.skill_cut = skill_cut
        self.student_cut = student_cut

        # set from data set
        self.num_skills = -1
        self.num_actions = -1    # 2 * num_skills, action: every skill correct/wrong

        # About Data Generator. Currently not support multiple schemes.
        self.padding_scheme = padding_scheme

        # LSTM Specification
        self.num_hidden = num_hidden
        self.init_mean = init_mean
        self.init_stddev = init_stddev
        self.batch_size = batch_size
        self.clipping_norm = clipping_norm
        self.dropout_keep = dropout_keep
        self.optimizer = optimizer

        # Running Specification
        self.num_epochs = num_epochs
        self.test_frequency = test_frequency
        self.save_frequency = save_frequency




