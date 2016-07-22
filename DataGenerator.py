import pandas as pd
import numpy as np
import math

class DKTBatch(object):
    def __init__(self, maxlen, sequences, inputs, skill_labels, result_labels):
        self.maxlen = maxlen
        self.sequences = sequences
        self.inputs = inputs
        self.skill_labels = skill_labels
        self.result_labels = result_labels

class DataGenerator(object):
    def __init__(self, hps):
        # hps for hyperparameters
        # convert file to sequence
        dataset = pd.read_csv(hps.dataset_file)
        # cut by skill
        dataset = dataset[dataset['skill'] < hps.skill_cut]
        # fill in dataset parameters
        hps.num_skills = len(dataset['skill'].value_counts())
        hps.num_actions = 2 * hps.num_skills

        dataset = dataset.values
        seqs = list()
        last_student = -1
        for i in range(len(dataset)):
            if dataset[i][0] != last_student:    # a new student
                last_student = dataset[i][0]
                seqs.append([(dataset[i][1], dataset[i][2])])  # (skill, correct)
            else:     # same student
                seqs[-1].append((dataset[i][1], dataset[i][2]))
        del dataset
        
        tot_seqs = min(len(seqs), hps.student_cut)
        print "total: %d student sequences" % tot_seqs
        
        # split train and test
        split_bitmap = pd.read_csv(hps.split_file).values
        train_seqs = list()
        test_seqs = list()
        train_record_cnt = 0
        test_record_cnt = 0
        
        # filter len < minlen
        for i in range(tot_seqs):
            if split_bitmap[i] == 0 and len(seq[i]) >= minlen:
                train_seqs.append(seqs[i])
                train_record_cnt += len(seqs[i])
            elif split_bitmap[i] == 1 and len(seq[i]) >= minlen:
                test_seqs.append(seqs[i])
                test_record_cnt += len(seqs[i])
        print "%d valid sequences, %d records for train" % (len(train_seqs), train_record_cnt)
        print "%d valid sequences, %d records for test" % (len(test_seqs), test_record_cnt)
        
        # takes around 2GB memory:
        self.train_batches = list()
        self.generate_batch(hps, train_seqs, self.train_batches)
        
        self.test_batches = list()
        self.generate_batch(hps, test_seqs, self.test_batches)
        
        print "all batch generated"
        
        self.train_cursor = -1
        self.test_cursor = -1

    def generate_batch(self, hps, all_seqs, batches):
        # sort according to length, then we do binning
        len_list = list()
        for seq in all_seqs:
            len_list.append(len(seq))
        len_order = np.argsort(len_list)

        batch_size = hps.batch_size
        for start in range(0, len(all_seqs), batch_size):
            end = min(len(all_seqs), start + batch_size)
            seq_idx = len_order[start : end]    # index in all_seqs
            maxlen = max(len_list[len_order[end - 1]], hps.pad_to)    

            # padded sequences: [skill, correct], empty [-1, -1]
            sequences = list()
            for _ in range(maxlen):
                sequences.append([])
                for _ in range(batch_size):
                    sequences[-1].append([-1, -1])

            for i in range(start, end):
                idx_in_batch = i - start
                seq = all_seqs[seq_idx[idx_in_batch]]
                for j in range(len(seq)):
                    if hps.padding_scheme == 'end_0':
                        sequences[j][idx_in_batch][0] = seq[j][0]
                        sequences[j][idx_in_batch][1] = seq[j][1]
                    elif hps.padding_scheme == 'start_0':
                        sequences[- j - 1][idx_in_batch][0] = seq[- j - 1][0]
                        sequences[- j - 1][idx_in_batch][1] = seq[- j - 1][1]

            # inputs
            inputs = list()
            skill_labels = list()
            result_labels = list()
            for i in range(maxlen):
                inputs.append([])
                skill_labels.append([])
                result_labels.append([])
                for j in range(batch_size):
                    inputs[-1].append(np.zeros(hps.num_actions))
                    skill_labels[-1].append(np.zeros(hps.num_skills))
                    result_labels[-1].append(0)
                    if sequences[i][j][0] != -1:
                        inputs[-1][-1][2 * sequences[i][j][0] + sequences[i][j][1]] = 1
                        skill_labels[-1][-1][sequences[i][j][0]] = 1
                        result_labels[-1][-1] = sequences[i][j][1]

            batches.append(DKTBatch(maxlen, sequences, inputs, skill_labels, result_labels))


































    
