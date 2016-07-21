import tensorflow as tf
from sklearn import metrics
import random
import pickle

class DKTRunner(object):
    
    def train(self, hps, graph, data_generator):
        with tf.Session(graph=graph) as session:
            # Initialize
            tf.initialize_all_variables().run()
            
            for epoch in range(hps.num_epochs):
                loss_sum = 0
                pred_all = []
                truth_all = []
                for batch in data_generator.train_batches:
                    feed_dict = {maxlen : batch.maxlen}
                    for i in range(batch.maxlen):
                        feed_dict[inputs[i]] = batch.inputs[i]
                        feed_dict[skill_labels[i]] = batch.skill_labels[i]
                        feed_dict[result_labels[i]] = batch.result_labels[i]
                        
                    _, l, pred = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
                    loss_sum += l
                    skill_label_all = np.concatenate(batch.skill_labels, axis=0)
                    # Exclude padded actions
                    for i in range(len(pred)):
                        if np.sum(skill_label_all[i]) != 0:
                            pred_all.append(pred[i])
                            truth_all.append(batch.result_labels[i])
                
                print len(pred_all)
                print "epoch " + str(epoch) + ": loss = " + str(loss_sum)
                print "Train AUC = " + str(metrics.roc_auc_score(truth_all, pred_all))
                
                # shuffle the train set after every epoch
                random.shuffle(data_generator.train_batches)
                
                if epoch % hps.test_frequency == 0:
                    pred_all = []
                    truth_all = []
                    all_status_pred = []
                    for batch in data_generator.test_batches:
                        feed_dict = {maxlen : batch.maxlen}
                        for i in range(batch.maxlen):
                            feed_dict[inputs[i]] = batch.inputs[i]
                            feed_dict[skill_labels[i]] = batch.skill_labels[i]
                            feed_dict[result_labels[i]] = np.zeros([hps.batch_size, ])    # No need to give the target
                            
                        if epoch % hps.save_frequency == 0:
                            status_pred, pred = session.run([test_status, test_prediction], feed_dict=feed_dict)
                            all_status_pred.append(status_pred)
                        else:
                            pred = test_prediction.eval(feed_dict)

                        skill_label_all = np.concatenate(batch.skill_labels, axis=0)
                        for i in range(len(pred)):
                            if np.sum(skill_label_all[i]) != 0:
                                pred_all.append(pred[i])
                                truth_all.append(batch.result_labels[i])
                            
                    print len(pred_all)
                    print "Test AUC = " + str(metrics.roc_auc_score(truth_all, pred_all))
                    print "Test accuracy = " + str(metrics.accuracy_score(truth_all, np.array(pred_all) > 0.5)) + "    "
                    
                    if epoch % hp.save_frequency == 0:
                        pred_action = file("prediction@epoch_" + str(epoch) + '.csv', 'w')
                        pred_action.write('pred,truth\n')
                        for i in range(len(pred_all)):
                            pred_action.write(str(pred_all[i]) + ',' + str(truth_all[i]) + '\n')
                        pred_action.flush()
                        pred_action.close()
                    
                        all_status_file = file("all_status@epoch_" + str(epoch) + '.pickle', 'w')
                        pickle.dump(all_status_pred, all_status_file)
                        all_status_file.close()
                
                        saver.save(session, "model@epoch" + str(epoch) + ".ckpt")
                        
                        