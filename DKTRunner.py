import tensorflow as tf
import numpy as np
from sklearn import metrics
import random
import pickle

class DKTRunner(object):
    
    def train(self, hps, dkt_graph, data_generator):
        num_skills = hps.num_skills
        random.seed(1234)
        with tf.Session(graph=dkt_graph.graph) as session:
            # Initialize
            tf.initialize_all_variables().run()
            
            for epoch in range(hps.num_epochs):
                loss_sum = 0
                pred_all = []
                truth_all = []
                for batch in data_generator.train_batches:
                    feed_dict = dict()    
                    feed_dict[dkt_graph.inputs] = batch.inputs[:-1]
                    feed_dict[dkt_graph.skill_labels] = batch.skill_labels
                    feed_dict[dkt_graph.result_labels] = batch.result_labels
                        
                    _, l, pred = session.run([dkt_graph.optimizer, dkt_graph.loss, dkt_graph.prediction], feed_dict=feed_dict)
                    loss_sum += l
                    skill_label_all = np.reshape(batch.skill_labels, [-1, num_skills])
                    result_label_all = np.reshape(batch.result_labels, [-1])
                    # Exclude padded actions
                    for i in range(len(pred)):
                        if np.sum(skill_label_all[i]) != 0:
                            pred_all.append(pred[i])
                            truth_all.append(result_label_all[i])
                
                print "epoch " + str(epoch) + ": loss = " + str(loss_sum / len(data_generator.train_batches))
                print "Train AUC = " + str(metrics.roc_auc_score(truth_all, pred_all))
                
                # shuffle the train set after every epoch
                random.shuffle(data_generator.train_batches)
                
                if epoch % hps.test_frequency == 0:
                    pred_all = []
                    truth_all = []
                    all_status_pred = []
                    for batch in data_generator.test_batches:
                        feed_dict = dict()    
                        feed_dict[dkt_graph.inputs] = batch.inputs[:-1]
                        feed_dict[dkt_graph.skill_labels] = batch.skill_labels
                        feed_dict[dkt_graph.result_labels] = np.zeros([batch.maxlen ,hps.batch_size])    # No need to give the target
                            
                        if epoch % hps.save_frequency == 0:
                            status_pred, pred = session.run([dkt_graph.test_status, dkt_graph.test_prediction], feed_dict=feed_dict)
                            all_status_pred.append(status_pred)
                        else:
                            pred = dkt_graph.test_prediction.eval(feed_dict)

                        skill_label_all = np.reshape(batch.skill_labels, [-1, num_skills])
                        result_label_all = np.reshape(batch.result_labels, [-1])
                        for i in range(len(pred)):
                            if np.sum(skill_label_all[i]) != 0:
                                pred_all.append(pred[i])
                                truth_all.append(result_label_all[i])
                    

                    print "Test AUC = " + str(metrics.roc_auc_score(truth_all, pred_all))
                    print "Test accuracy = " + str(metrics.accuracy_score(truth_all, np.array(pred_all) > 0.5)) + "    "
                    
                    if epoch % hps.save_frequency == 0:
                        pred_action = file("prediction@epoch_" + str(epoch) + '.csv', 'w')
                        pred_action.write('pred,truth\n')
                        for i in range(len(pred_all)):
                            pred_action.write(str(pred_all[i]) + ',' + str(truth_all[i]) + '\n')
                        pred_action.flush()
                        pred_action.close()
                    
                        all_status_file = file("all_status@epoch_" + str(epoch) + '.pickle', 'w')
                        pickle.dump(all_status_pred, all_status_file)
                        all_status_file.close()
                
                        dkt_graph.saver.save(session, "model@epoch" + str(epoch) + ".ckpt")


