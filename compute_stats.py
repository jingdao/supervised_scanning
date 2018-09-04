#!/usr/bin/python

import numpy
import sys

results_folder = 'results/'

nf_results = []
f = open(results_folder+'/nf_ref.txt','r')
for l in f:
	nf_results.append(float(l.split()[2]))
f.close()

rnn_results = []
f = open(results_folder+'/rnn_ref.txt','r')
for l in f:
	rnn_results.append(float(l.split()[2].split('/')[0]))
f.close()

baseline_results = []
f = open(results_folder+'/baseline.txt','r')
for l in f:
	baseline_results.append(float(l))
f.close()

train_subset = 0.9
rnn_results = numpy.array(rnn_results)
nf_results = numpy.array(nf_results)
norm = numpy.array(baseline_results)[:len(rnn_results)]

norm = norm.reshape((-1,5)).mean(axis=1)
nf_results = nf_results.reshape((-1,5)).mean(axis=1)
rnn_results = rnn_results.reshape((-1,5)).mean(axis=1)

num_train = int(train_subset * len(nf_results))
num_test = len(nf_results) - num_train
for s in ['NF','RNN']:
	path_diff = (nf_results if s=='NF' else rnn_results) - norm
	print(s)
	print("path_diff train : %.1f %.2f %.2f %.1f"% (path_diff[:num_train].min(),path_diff[:num_train].mean(),numpy.median(path_diff[:num_train]),path_diff[:num_train].max()))
	print("path_diff test  : %.1f %.2f %.2f %.1f"% (path_diff[num_train:].min(),path_diff[num_train:].mean(),numpy.median(path_diff[num_train:]),path_diff[num_train:].max()))

print('')
for s in ['NF','RNN']:
	path_norm = (nf_results if s=='NF' else rnn_results) / norm
	print(s)
	print("path_norm train : %.3f %.3f %.3f %.3f" % (path_norm[:num_train].min(),path_norm[:num_train].mean(),numpy.median(path_norm[:num_train]),path_norm[:num_train].max()))
	print("path_norm test  : %.3f %.3f %.3f %.3f" % (path_norm[num_train:].min(),path_norm[num_train:].mean(),numpy.median(path_norm[num_train:]),path_norm[num_train:].max()))

combined = numpy.vstack((nf_results,rnn_results)).transpose()
best_policy = numpy.argmin(combined, axis=1)
ptrain = numpy.zeros(5)
ptest = numpy.zeros(5)
i,c = numpy.unique(best_policy[:num_train], return_counts=True)
ptrain[i] = c
i,c = numpy.unique(best_policy[num_train:], return_counts=True)
ptest[i] = c
print('')
print('best_policy train (%4d): %4d %4d'%(num_train,ptrain[0],ptrain[1]))
print('best_policy test  (%4d): %4d %4d'%(num_test,ptest[0],ptest[1]))
