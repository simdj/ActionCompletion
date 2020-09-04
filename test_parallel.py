
from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    info('function f')
    print('hello', name)

if __name__ == '__main__':
    info('main line')
    p = Process(target=f, args=('bob',))
    p.start()
    p.join()

     
# # import the necessary packages
# # from pyimagesearch.parallel_hashing import process_images
# # from pyimagesearch.parallel_hashing import chunk
# from multiprocessing import Pool
# from multiprocessing import cpu_count
# from imutils import paths
# import numpy as np
# import argparse
# import pickle
# import os



# import cv2
# def dhash(image, hashSize=8):
# 	# convert the image to grayscale
# 	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 	# resize the input image, adding a single column (width) so we
# 	# can compute the horizontal gradient
# 	resized = cv2.resize(gray, (hashSize + 1, hashSize))
# 	# compute the (relative) horizontal gradient between adjacent
# 	# column pixels
# 	diff = resized[:, 1:] > resized[:, :-1]
# 	# convert the difference image to a hash
# def convert_hash(h):
# 	# convert the hash to NumPy's 64-bit float and then back to
# 	# Python's built in int
# 	return int(np.array(h, dtype="float64"))

# def process_images(payload):
# 	# display the process ID for debugging and initialize the hashes
# 	# dictionary
# 	print("[INFO] starting process {}".format(payload["id"]))
# 	hashes = {}
# 	# loop over the image paths
# 	for imagePath in payload["input_paths"]:
# 		# load the input image, compute the hash, and conver it
# 		image = cv2.imread(imagePath)
# 		h = dhash(image)
# 		h = convert_hash(h)
# 		# update the hashes dictionary
# 		l = hashes.get(h, [])
# 		l.append(imagePath)
# 		hashes[h] = l
# 	# serialize the hashes dictionary to disk using the supplied
# 	# output path
# 	print("[INFO] process {} serializing hashes".format(payload["id"]))
# 	f = open(payload["output_path"], "wb")
# 	f.write(pickle.dumps(hashes))
# 	f.close()


# def chunk(l, n):
# 	# loop over the list in n-sized chunks
# 	for i in range(0, len(l), n):
# 		# yield the current n-sized chunk to the calling function
# 		yield l[i: i + n]

# # check to see if this is the main thread of execution
# if __name__ == "__main__":
# 	# construct the argument parser and parse the arguments
# 	ap = argparse.ArgumentParser()
# 	ap.add_argument("-i", "--images", required=True, type=str,
# 		help="path to input directory of images")
# 	ap.add_argument("-o", "--output", required=True, type=str,
# 		help="path to output directory to store intermediate files")
# 	ap.add_argument("-a", "--hashes", required=True, type=str,
# 		help="path to output hashes dictionary")
# 	ap.add_argument("-p", "--procs", type=int, default=-1,
# 		help="# of processes to spin up")
# 	args = vars(ap.parse_args())

# 	# determine the number of concurrent processes to launch when
# 	# distributing the load across the system, then create the list
# 	# of process IDs
# 	procs = args["procs"] if args["procs"] > 0 else cpu_count()
# 	procIDs = list(range(0, procs))
# 	# grab the paths to the input images, then determine the number
# 	# of images each process will handle
# 	print("[INFO] grabbing image paths...")
# 	allImagePaths = sorted(list(paths.list_images(args["images"])))

# 	numImagesPerProc = len(allImagePaths) / float(procs)
# 	numImagesPerProc = int(np.ceil(numImagesPerProc))
# 	# chunk the image paths into N (approximately) equal sets, one
# 	# set of image paths for each individual process
# 	chunkedPaths = list(chunk(allImagePaths, numImagesPerProc))

# 	# initialize the list of payloads
# 	payloads = []
# 	# loop over the set chunked image paths
# 	for (i, imagePaths) in enumerate(chunkedPaths):
# 		# construct the path to the output intermediary file for the
# 		# current process
# 		outputPath = os.path.sep.join([args["output"],
# 			"proc_{}.pickle".format(i)])
# 		# construct a dictionary of data for the payload, then add it
# 		# to the payloads list
# 		data = {
# 			"id": i,
# 			"input_paths": imagePaths,
# 			"output_path": outputPath
# 		}
# 		payloads.append(data)

# 	# construct and launch the processing pool
# 	print("[INFO] launching pool using {} processes...".format(procs))
# 	pool = Pool(processes=procs)
# 	pool.map(process_images, payloads)
# 	# close the pool and wait for all processes to finish
# 	print("[INFO] waiting for processes to finish...")
# 	pool.close()
# 	pool.join()
# 	print("[INFO] multiprocessing complete")




# 	# initialize our *combined* hashes dictionary (i.e., will combine
# 	# the results of each pickled/serialized dictionary into a
# 	# *single* dictionary
# 	print("[INFO] combining hashes...")
# 	hashes = {}
# 	# loop over all pickle files in the output directory
# 	for p in paths.list_files(args["output"], validExts=(".pickle"),):
# 		# load the contents of the dictionary
# 		data = pickle.loads(open(p, "rb").read())
# 		# loop over the hashes and image paths in the dictionary
# 		for (tempH, tempPaths) in data.items():
# 			# grab all image paths with the current hash, add in the
# 			# image paths for the current pickle file, and then
# 			# update our hashes dictionary
# 			imagePaths = hashes.get(tempH, [])
# 			imagePaths.extend(tempPaths)
# 			hashes[tempH] = imagePaths
# 	# serialize the hashes dictionary to disk
# 	print("[INFO] serializing hashes...")
# 	f = open(args["hashes"], "wb")
# 	f.write(pickle.dumps(hashes))
# 	f.close()