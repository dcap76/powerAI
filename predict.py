#coding:utf-8
import codecs
import os
import sys
import json
import shutil
import pickle
import logging
import parser
import numpy as np
import pandas as pd
import tensorflow as tf
import jieba
from text_cnn_rnn import TextCNNRNN
from datetime import datetime
reload(sys) 
sys.setdefaultencoding('utf-8')

logging.getLogger().setLevel(logging.INFO)

def load_trained_params(trained_dir):
	params = json.loads(open(trained_dir + 'trained_parameters.json').read())
	words_index = json.loads(codecs.open(trained_dir + 'words_index.json', encoding="utf-8").read())
	labels = json.loads(codecs.open(trained_dir + 'labels.json', encoding="utf-8").read())

	with open(trained_dir + 'embeddings.pickle', 'rb') as input_file:
		fetched_embedding = pickle.load(input_file)
	embedding_mat = np.array(fetched_embedding, dtype = np.float32)
	return params, words_index, labels, embedding_mat

def load_test_data(filepath):
	test_examples = []
	filenames = os.listdir(filepath)
	for filename in filenames:
		with codecs.open(os.path.join(filepath, filename), 'r', encoding='utf-8') as file:
			line_list = []
			for line in file.readlines():
				line_list = line_list + jieba.lcut(line, cut_all=False)
			readstr = ' '.join(line_list)
			test_examples.append(parser.clean_str(readstr).split())
	return test_examples, filenames


def map_word_to_index(examples, words_index):
	x_ = []
	for example in examples:
		temp = []
		temp_0 = []
		for word in example:
			if word in words_index:
				temp.append(words_index[word])
			else:
				temp_0.append(0)
		x_.append(temp + temp_0)
	return x_

def predict_unseen_data():
	trained_dir = sys.argv[1]
	if not trained_dir.endswith('/'):
		trained_dir += '/'
	filepath = './data/predict'

	params, words_index, labels, embedding_mat = load_trained_params(trained_dir)
	x_, filename_ = load_test_data(filepath)
	x_ = parser.pad_sentences(x_, forced_sequence_length=params['sequence_length'])
	x_ = map_word_to_index(x_, words_index)
	#一行一个索引数组
	x_test, filename_test = np.asarray(x_), np.asarray(filename_)
	#x_test, y_test = np.asarray(x_), None
	#if y_ is not None:
	#	y_test = np.asarray(y_)

	timestamp = trained_dir.split('/')[-2].split('_')[-1]
	predicted_dir = './predicted_results_' + timestamp + '/'
	if os.path.exists(predicted_dir):
		shutil.rmtree(predicted_dir)
	os.makedirs(predicted_dir)

	with tf.Graph().as_default():
		session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
		sess = tf.Session(config=session_conf)
		with sess.as_default():
			cnn_rnn = TextCNNRNN(
				embedding_mat = embedding_mat,
				non_static = params['non_static'],
				hidden_unit = params['hidden_unit'],
				sequence_length = len(x_test[0]),
				max_pool_size = params['max_pool_size'],
				filter_sizes = map(int, params['filter_sizes'].split(",")),
				num_filters = params['num_filters'],
				num_classes = len(labels),
				embedding_size = params['embedding_dim'],
				l2_reg_lambda = params['l2_reg_lambda'])

			def real_len(batches):
				return [np.ceil(np.argmin(batch + [0]) * 1.0 / params['max_pool_size']) for batch in batches]

			def predict_step(x_batch):
				feed_dict = {
					cnn_rnn.input_x: x_batch,
					cnn_rnn.dropout_keep_prob: 1.0,
					cnn_rnn.batch_size: len(x_batch),
					cnn_rnn.pad: np.zeros([len(x_batch), 1, params['embedding_dim'], 1]),
					cnn_rnn.real_len: real_len(x_batch),
				}
				score,predictions = sess.run([cnn_rnn.scores,cnn_rnn.predictions], feed_dict)
				return score,predictions

			def proba(x):
				e_x= np.exp(x-np.max(x))
				return e_x / e_x.sum()	

			checkpoint_file = trained_dir + 'best_model.ckpt'
			saver = tf.train.Saver(tf.all_variables())
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file[:-5]))
			saver.restore(sess, checkpoint_file)
			logging.critical('{} has been loaded'.format(checkpoint_file))


			predict_labels_index, predict_labels, predict_filename,probabs = [], [],[],[]

			#batch不为1的写法
			batches = parser.batch_iter(list(x_test), params['batch_size'], 1, shuffle=False)
			for x_batch in batches:
				tmp = predict_step(x_batch)
				for indexOfLable in tmp[1]:
					predict_labels_index.append(indexOfLable)
					predict_labels.append(labels[indexOfLable])
				for score in tmp[0]:
					probabs.append(proba(score).max())
			batches2 = parser.batch_iter(list(filename_test), params['batch_size'], 1, shuffle=False)
			for tmp in batches2:
				for filename in tmp:
					predict_filename.append(filename)
            

			#batch为1的简化写法 
			#for indexOfLable in predict_step(x_test)[1]:
			#	predict_labels_index.append(indexOfLable)
			#	predict_labels.append(labels[indexOfLable])
			#for score in predict_step(x_test)[0]:
			#	probabs.append(proba(score).max())
			#for filename in filename_test:
			#	predict_filename.append(filename)


			infoList=[]
			for i in range(len(predict_labels_index)):
				info={}  
				info["prob"]= float(probabs[i])
				info["sampleId"]=predict_filename[i]  
				info["label"]=predict_labels[i]
				infoList.append(info)

			allJson={}  
			allJson["type"]="Fiance Product Classifcation"  
			allJson["result"]=infoList


			with codecs.open(predicted_dir + 'predictions_all.json', 'w', encoding="utf-8") as outfile:
				json.dump(allJson, outfile, indent=4, ensure_ascii=False)
			

			#df['PREDICTED'] = predict_labels
			#df.to_json(path_or_buf=predicted_dir + 'predictions_all.json', orient='records', lines=True)

			#if y_test is not None:
			#	y_test = np.array(np.argmax(y_test, axis=1))
			#	accuracy = sum(np.array(predictions) == y_test) / float(len(y_test))
			#	logging.critical('The prediction accuracy is: {}'.format(accuracy))

			logging.critical('Prediction is complete, all files have been saved: {}'.format(predicted_dir))

if __name__ == '__main__':
	logging.critical("Get started at " + str(datetime.now()))
	predict_unseen_data()
	logging.critical("Prediction completed at " + str(datetime.now()))
