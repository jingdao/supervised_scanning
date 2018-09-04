#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import sys
import math
import time
import heapq
import h5py
import scipy.misc

raster_file = 'map%s.npy' % sys.argv[1]
traj_file = 'traj_' + raster_file
draw = '--draw' in sys.argv
numpy.random.seed(0)

data_folder = 'data/'
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

scan_range = 100
def prepare_lut():
	sys.stderr.write('Preparing lookup-table. Please wait...\n')
	for x,y in navigable:
		lut_occupied[(x,y)] = []
		lut_unoccupied[(x,y)] = []
		for i in range(360):
			xi = x
			yi = y
			rx = numpy.cos(i * numpy.pi / 180)
			ry = numpy.sin(i * numpy.pi / 180)
			dx = numpy.abs(1 / rx) if rx!=0 else numpy.inf
			dy = numpy.abs(1 / ry) if ry!=0 else numpy.inf
			tx = dx
			ty = dy
			while True:
				if xi<0 or xi>=raster.shape[1] or yi<0 or yi>=raster.shape[0]: #out of bounds
					break
				elif tx!=numpy.inf and tx > scan_range or ty!=numpy.inf and ty > scan_range: #out of scan range
					break
				else:
					if raster[yi,xi]:
						lut_occupied[(x,y)].append((xi,yi))
						break
					else:
						lut_unoccupied[(x,y)].append((xi,yi))
				if tx==ty:
					tx += dx
					ty += dy
					xi += int(numpy.sign(rx))
					yi += int(numpy.sign(ry))
				elif tx < ty:
					tx += dx
					xi += int(numpy.sign(rx))
				else:
					ty += dy
					yi += int(numpy.sign(ry))
	sys.stderr.write('Prepared lookup-table (%d/%d elements)\n'%(len(lut_occupied),len(lut_unoccupied)))

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

def get_nearest_frontier(x, y):
	candidates = get_frontiers(x, y)
	if len(candidates) == 0:
		return None
	else:
		dist,_ = batch_shortest_path(x,y, candidates, drivable)
		nearest = candidates[numpy.argmin(dist), :]
		return nearest

if draw:
	fig = plt.figure()
	ax = fig.add_subplot(111, aspect='equal')
	font = {'color':'red', 'size':16}

try:
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
	sys.stderr.write('Loaded LUT file\n')
except IOError:
	prepare_lut()
	pos1 = []
	pos2 = []
	counts1 = []
	counts2 = []
	keys = []
	for key in lut_occupied:
		keys.append(key)
		counts1.append(len(lut_occupied[key]))
		counts2.append(len(lut_unoccupied[key]))
		pos1.extend(lut_occupied[key])
		pos2.extend(lut_unoccupied[key])
	f = h5py.File(data_folder+'/'+raster_file+'.h5','w')
	f.create_dataset( 'keys', data=numpy.array(keys), compression='gzip', compression_opts=4, dtype=numpy.int32)
	f.create_dataset( 'counts1', data=numpy.array(counts1), compression='gzip', compression_opts=4, dtype=numpy.int32)
	f.create_dataset( 'counts2', data=numpy.array(counts2), compression='gzip', compression_opts=4, dtype=numpy.int32)
	f.create_dataset( 'pos1', data=numpy.array(pos1), compression='gzip', compression_opts=4, dtype=numpy.int32)
	f.create_dataset( 'pos2', data=numpy.array(pos2), compression='gzip', compression_opts=4, dtype=numpy.int32)
	f.close()

for ep in range(len(path)):

	floor_map[:,:,:] = color_map['unknown']
	drivable.clear()
	frontier[:,:] = False
	trajectory_x = []
	trajectory_y = []
	source = path[ep % len(path)]
	steps = 0
	i = 0
	collect_scan_lut(source[0], source[1])

	while True:
		target = get_nearest_frontier(source[0], source[1])
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
		i += 1

