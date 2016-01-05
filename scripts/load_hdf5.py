
from imports import *


# helper function for sharing data
def make_shared(x,y):
	x_shared = theano.shared(asarray(x, dtype=theano.config.floatX),
		borrow=True)
	y_shared = theano.shared(asarray(y, dtype=theano.config.floatX),
		borrow=True)
	return(x_shared, T.cast(y_shared,'int32'))


# stores whale image data (detector) as a numpy array in hdf5 format
def store_detect_data(path_img,chunk_size=1000,w1=100,h1=100):
	
	print('... Reading Input Matrix ...')
	names_ok = glob(path_img+'select/right/*jpg')
	names_nope = glob(path_img+'select/wrong/*jpg')
	names_all = names_ok + names_nope
	n_ok, n_nope = len(names_ok), len(names_nope)
	print('Ok: '+str(n_ok)+' Nope: '+str(n_nope))
	rng = RandomState(290615)
	nrow, ncol = n_ok + n_nope, 100 * 100 * 3
	shuffled_idx = rng.permutation(range(nrow))
	
	t_start = time()
	data = h5py.File(path_img+'select/detect.hdf5','w')
	X = data.create_dataset(
		name='X',
		shape=(nrow,ncol),
		dtype='f64',
		chunks=(chunk_size,ncol))
	y = data.create_dataset(
		name='y',
		shape=(nrow,),
		dtype='i32',
		chunks=(chunk_size,))

	# read data based on shuffled_idx
	for i in range(nrow):
		si = shuffled_idx[i]
		im = Image.open(names_all[si])
		X[i,] = asarray(im).ravel()/255.
		y[i] = 1 if si < n_ok else 0 
		if i % chunk_size == 0:
			print('Reading row %i' % i)
	
	data.close()
	print('Saving to %s took %.1fs' % (path_img+'select/detect.hdf5', time()-t_start))



# stores whale image data (classifier) as a numpy array in hdf5 format
def store_classify_data(path_img,chunk_size=1000,w1=100,h1=100):

	train = pd.read_csv(path_img+'data/train.csv')
	freq = train.groupby('whaleID').size()
	fnames = glob(path_img+'thumbs/train/whale_*/*.jpg')
	nrow, ncol = len(fnames), w1 * h1 * 3
	rng = RandomState(290615)
	shuffled_idx = rng.permutation(range(nrow))
	data = h5py.File(path_img+'data/data.hdf5','w')

	t_start = time()
	X = data.create_dataset(
		name='X',
		shape=(nrow,ncol),
		dtype='f64',
		chunks=(chunk_size,ncol))
	y = data.create_dataset(
		name='y',
		shape=(nrow,),
		dtype='i16',
		chunks=(chunk_size,))

	# read data based on shuffled_idx
	for i in range(nrow):
		si = shuffled_idx[i]
		im = Image.open(fnames[si])
		f_ = re.search('train/(whale_[0-9]*)/',fnames[si]).group(1)
		X[i,] = asarray(im.resize((w1,h1))).ravel() / 255.
		y[i] = np.where(f_ == freq.index)[0][0]
		if i % chunk_size == 0:
			print('Reading row %i' % i)
	data.close()
	# with zipfile.ZipFile(path_img+'data/data_hdf5.zip','w') as z:
	# 	z.write(path_img+'data/data.hdf5')
	print('Saving to %s took %.1fs' % (path_img+'data/data.hdf5',time()-t_start))


