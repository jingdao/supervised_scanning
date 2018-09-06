Supervised Learning Method for Minimum-Time Robotic Scanning and Navigation
========

Supplementary material (map data and scan simulations) for the ICRA submission
*Supervised Learning Method for Minimum-Time Robotic Scanning and Navigation*.

Prerequisites
-----
1. numpy
2. scipy
3. matplotlib
4. tensorflow
5. h5py

Map Data
-----

Extract the zip file *maps.zip* under the data directory.
Map data consists of a thousand 50x50 numpy arrays stored in *.npy* format.
Each map has a corresponding 5x2 numpy array which stores 5 random starting positions for the robot (*traj_mapX.npy*)
The first time sim_nf.py is run on any map, a lookup table will be generated as *mapX.npy.h5*.

Usage
------

	#run scan simulation with nearest frontier on map ID 1
	python sim_nf.py 1 --draw

	#run scan simulation with RNN on map ID 1
	python sim_rnn.py 1 --draw

Benchmarking
-----

The numerical results in the paper may be reproduced by running the following commands:

	#run scan simulation on 1000 maps using nearest frontier method
	for i in `seq 1000`; do python -u sim_nf.py $i >> results/nf.txt;done

	#run scan simulation on 1000 maps using RNN method
	for i in `seq 1000`; do python -u sim_rnn.py $i >> results/rnn.txt;done

Result statistics may be computed by running `python compute_stats.py`. Note that
the best_policy count is not the same as the paper because the CNN results and
the results prior to fine-tuning are not included in the comparison.

Screenshots
-----

![screenshot1](results/screenshot1.png?raw=true)

![screenshot2](results/screenshot2.png?raw=true)
