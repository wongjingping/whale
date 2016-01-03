
from imports import *

# stores whale image data as a numpy array in hdf5 format
def store_classify_data(chunk_size=1000,w1=100,h1=100):

	train = pd.read_csv(path_img+'data/train.csv')
	freq = train.groupby('whaleID').size()
	fnames = glob(path_img+'thumbs/train/whale_*/*.jpg')
	nrow, ncol, nclass = len(fnames), w1 * h1 * 3, len(freq)
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
		shape=(nrow,nclass),
		dtype='i16',
		chunks=(chunk_size,nclass),
		fillvalue=0)

	# read data based on shuffled_idx
	for i in range(nrow):
		si = shuffled_idx[i]
		im = Image.open(fnames[si])
		f_ = re.search('train/(whale_[0-9]*)/',fnames[si]).group(1)
		X[i,] = asarray(im.resize((w1,h1))).ravel() / 255.
		y[i,np.where(f_ == freq.index)[0][0]] = 1
		if i % chunk_size == 0:
			print('Reading row %i' % i)
	data.close()
	# with zipfile.ZipFile(path_img+'data/data_hdf5.zip','w') as z:
	# 	z.write(path_img+'data/data.hdf5')
	print('Storing took %.1fs' % (time()-t_start))


# helper function for sharing data
def make_shared(x,y):
	x_shared = theano.shared(asarray(x, dtype=theano.config.floatX),
		borrow=True)
	y_shared = theano.shared(asarray(y, dtype=theano.config.floatX),
		borrow=True)
	return(x_shared, T.cast(y_shared,'int32'))

# load data from hdf5 file
def load_classify_data(path_hdf5):
	print('Loading data from %s' % path_hdf5)
	data = h5py.File(path_hdf5,'r')
	return(data['X'], data['y'])

