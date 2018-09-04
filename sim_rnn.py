import sys
import numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys
import math
import time
import heapq
import h5py
import scipy.misc
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from navnet import NavNet

area_id = int(sys.argv[1])
data_folder = 'data'
model_file = 'models/finetune_lstm.ckpt'
max_seq_len = 66
num_frontiers = 200
draw = '--draw' in sys.argv
net = NavNet(1,50,num_frontiers,2,2,max_seq_len)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
sess = tf.Session(config=config)
saver = tf.train.Saver()
saver.restore(sess, model_file)

raster_file = 'map%s.npy' % area_id
traj_file = 'traj_' + raster_file
context_size = 50
raster = numpy.load(data_folder+'/'+raster_file)
path = numpy.load(data_folder+'/'+traj_file).astype(int)
color_map = {
	'unknown': (0,0,0),
	'occupied': (255,255,255),
	'unoccupied': (100,100,100),
	'frontier': (100,255,0),
}
floor_map = numpy.zeros((raster.shape[0],raster.shape[1],3), dtype=numpy.uint8)
floor_map[:,:,:] = color_map['unknown']
drivable = set()
navigable = set()
frontier = numpy.zeros((raster.shape[0], raster.shape[1]), dtype=bool)
unoccupied = numpy.nonzero(raster==0)
unoccupied = set(zip(unoccupied[1], unoccupied[0]))
occupied = numpy.nonzero(raster==1)
occupied = set(zip(occupied[1], occupied[0]))
lut_occupied = {}
lut_unoccupied = {}
Q = [tuple(path[0])]
while len(Q) > 0:
	q = Q[-1]
	del Q[-1]
	if q in navigable:
		continue
	navigable.add(q)
	n1 = (q[0]-1, q[1])
	n2 = (q[0]+1, q[1])
	n3 = (q[0], q[1]-1)
	n4 = (q[0], q[1]+1)
	if n1 in unoccupied and not n1 in navigable:
		Q.append(n1)
	if n2 in unoccupied and not n2 in navigable:
		Q.append(n2)
	if n3 in unoccupied and not n3 in navigable:
		Q.append(n3)
	if n4 in unoccupied and not n4 in navigable:
		Q.append(n4)

def shortest_path(x1,y1, x2,y2, drivable):
	closed = set()
	opened = set()
	cameFrom = {}
        F = {(x1,y1):math.sqrt((x1-x2)**2 + (y1-y2)**2)}
	G = {(x1,y1):0}
        H = {(x1,y1):math.sqrt((x1-x2)**2 + (y1-y2)**2)}
	PQ = []
	start = (x1,y1)
	goal = (x2,y2)
	heapq.heappush(PQ, [F[start], start])
	opened.add(start)
	while len(PQ) > 0:
		_,current = heapq.heappop(PQ)
		if current == goal:
			path = [current]
			while current in cameFrom:
				current = cameFrom[current]
				path.append(current)
			path.reverse()
			return path[1:]
		closed.add(current)
		opened.remove(current)
		neighbors = [
			(current[0]-1,current[1]),
			(current[0]+1,current[1]),
			(current[0],current[1]-1),
			(current[0],current[1]+1),
			(current[0]-1,current[1]-1),
			(current[0]-1,current[1]+1),
			(current[0]+1,current[1]-1),
			(current[0]+1,current[1]+1),
		]
		for i in range(len(neighbors)):
			n = neighbors[i]
			cost = 1 if i < 4 else 1.4142135623730951
			if not n in drivable or n in closed:
				continue
			if not n in opened:
				G[n] = G[current] + cost
				H[n] = math.sqrt((n[0]-x2)**2 + (n[1]-y2)**2)
				F[n] = G[n] + H[n]
				cameFrom[n] = current
				opened.add(n)
				heapq.heappush(PQ, [F[n], n])
			elif G[current] + cost < G[n]:
				G[n] = G[current] + cost
				F[n] = G[n] + H[n]
				cameFrom[n] = current
				#decrease key
				for i in range(len(PQ)):
					if PQ[i][1]==n:
						PQ[i][0] = F[n]
						heapq._siftdown(PQ, 0, i)

def batch_shortest_path(x0,y0, targets, drivable):
	closed = set()
	opened = set()
	target_set = set([tuple(t) for t in targets])
	G = {(x0,y0):0}
	cameFrom = {}
	PQ = []
	start = (x0,y0)
	heapq.heappush(PQ, [G[start], start])
	opened.add(start)
	while len(PQ) > 0:
		_,current = heapq.heappop(PQ)
		closed.add(current)
		opened.remove(current)
		if len(closed.intersection(target_set)) == len(target_set):
			distances = []
			paths = []
			for t in targets:
				gval = G[tuple(t)]
				distances.append(gval)
				current = tuple(t)
				path = [current]
				while current in cameFrom:
					current = cameFrom[current]
					path.append(current)
				path.reverse()
				paths.append(path[1:])
			return distances, paths
		neighbors = [
			(current[0]-1,current[1]),
			(current[0]+1,current[1]),
			(current[0],current[1]-1),
			(current[0],current[1]+1),
			(current[0]-1,current[1]-1),
			(current[0]-1,current[1]+1),
			(current[0]+1,current[1]-1),
			(current[0]+1,current[1]+1),
		]
		for i in range(len(neighbors)):
			n = neighbors[i]
			cost = 1 if i < 4 else 1.4142135623730951
			if not n in drivable or n in closed:
				continue
			if not n in opened:
				G[n] = G[current] + cost
				cameFrom[n] = current
				opened.add(n)
				heapq.heappush(PQ, [G[n], n])
			elif G[current] + cost < G[n]:
				G[n] = G[current] + cost
				cameFrom[n] = current
				#decrease key
				for i in range(len(PQ)):
					if PQ[i][1]==n:
						PQ[i][0] = G[n]
						heapq._siftdown(PQ, 0, i)

def collect_scan_lut(x, y):
	for xi,yi in lut_occupied[(x,y)]:
		floor_map[yi,xi,:] = color_map['occupied']
	for xi,yi in lut_unoccupied[(x,y)]:
		floor_map[yi,xi,:] = color_map['unoccupied']
		drivable.add((xi,yi))

def get_frontiers(x, y):
    global frontier
    floor_map[frontier] = color_map['unoccupied']
    has_unknown = numpy.zeros(frontier.shape, dtype=bool)
    has_unknown[:,1:] = numpy.logical_or(has_unknown[:,1:], numpy.all(floor_map[:,:-1,:] == color_map['unknown'], axis=-1))
    has_unknown[:,:-1] = numpy.logical_or(has_unknown[:,:-1], numpy.all(floor_map[:,1:,:] == color_map['unknown'], axis=-1))
    has_unknown[1:,:] = numpy.logical_or(has_unknown[1:,:], numpy.all(floor_map[:-1,:,:] == color_map['unknown'], axis=-1))
    has_unknown[:-1,:] = numpy.logical_or(has_unknown[:-1,:], numpy.all(floor_map[1:,:,:] == color_map['unknown'], axis=-1))
    has_unoccupied = numpy.all(floor_map == color_map['unoccupied'], axis=-1)
    frontier = numpy.logical_and(has_unknown, has_unoccupied)
    floor_map[frontier] = color_map['frontier']
    fy, fx = numpy.nonzero(frontier)
    candidates = numpy.vstack((fx,fy)).transpose()
    return candidates

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
font = {'color':'red', 'size':16}
f = h5py.File(data_folder+'/'+raster_file+'.h5','r')
keys = f['keys']
c1 = 0
c2 = 0
for i in range(len(keys)):
	x,y = keys[i]
	lut_occupied[(x,y)] = f['pos1'][c1:c1+f['counts1'][i], :]
	c1 += f['counts1'][i]
	lut_unoccupied[(x,y)] = f['pos2'][c2:c2+f['counts2'][i], :]
	c2 += f['counts2'][i]
f.close()

for ep in range(len(path)):

	floor_map[:,:,:] = color_map['unknown']
	drivable.clear()
	frontier[:,:] = False
	trajectory_x = []
	trajectory_y = []
	source = path[ep]
	steps = 0
	i = 0
	inputs = numpy.zeros((1, max_seq_len, 50,50), dtype=numpy.float32)
	inputs2 = numpy.zeros((1, max_seq_len, 2), dtype=numpy.float32)
	inputs3 = numpy.zeros((1, max_seq_len, num_frontiers, 2), dtype=numpy.float32)
	collect_scan_lut(source[0], source[1])

	while True:
		candidates = get_frontiers(source[0], source[1])
		if len(candidates) > 0:
			snapshot = scipy.misc.imresize(floor_map[:,:,0], (context_size, context_size)) / 255.0
			robot_pos = [1.0*source[0]/raster.shape[1], 1.0*source[1]/raster.shape[0]]
			inputs[0,i,:,:] = snapshot
			inputs2[0,i,:] = [robot_pos[0], robot_pos[1]]
			inputs3[0,i,:len(candidates),:] = (1.0 * candidates / raster.shape[::-1])
			output_tensor = sess.run(net.output, { net.image_pl: inputs, net.pos_pl: inputs2, net.frontier_pl: inputs3, net.seq_pl: [i+1] })
			next_pos = numpy.array([int(numpy.round(output_tensor[0,i,0] * raster.shape[1])), int(numpy.round(output_tensor[0,i,1] * raster.shape[0]))])
			i += 1

			#navigate to frontier point closest to next_pos
			#optimistically assume that all unknown cells are drivable
			opt_drivable = set(drivable)
			uy,ux = numpy.nonzero(numpy.all(floor_map == color_map['unknown'], axis=-1))
			opt_drivable = opt_drivable.union(zip(ux,uy))
			opt_drivable = opt_drivable.union(next_pos)
			dist = batch_shortest_path(next_pos[0], next_pos[1], candidates, opt_drivable)
			if dist is not None:
				dist = dist[0]
			else:
				dist = numpy.sum((candidates - next_pos)**2, axis=1)
			target = candidates[numpy.argmin(dist)]
		else:
			target = None
		if target is None:
			print('Sequence %d: %.2f steps %.2f completeness'%(ep, steps, 1.0 * len(drivable) / len(navigable)))
			break
		current_path = shortest_path(source[0], source[1], target[0], target[1], drivable)
		for j in range(len(current_path)):
			x = current_path[j][0]
			y = current_path[j][1]
			trajectory_x.append(x)
			trajectory_y.append(y)
			t = time.time()
			if draw:
				ax.clear()
				plt.imshow(floor_map, interpolation='none')
				plt.scatter(x, y, color='r', s=20)
				plt.scatter(target[0], target[1], color='g', s=20)
				plt.scatter(next_pos[0], next_pos[1], color='y', s=20)
				plt.plot(trajectory_x, trajectory_y, 'b-')
				plt.pause(0.01)
			collect_scan_lut(x, y)
			if j==0:
				distance = numpy.sqrt((source[0] - x) * (source[0] - x) + (source[1] - y) * (source[1] - y))
				steps += distance
			else:
				distance = numpy.sqrt((current_path[j-1][0] - x) * (current_path[j-1][0] - x) + (current_path[j-1][1] - y) * (current_path[j-1][1] - y))
				steps += distance
		source = target
	
