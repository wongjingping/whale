
from imports import *


def store_classify_data(chunk_size=1000,p_test=0.1,w1=100,h1=100):

	train = pd.read_csv(path_img+'data/train.csv')
	freq = train.groupby('whaleID').size()
	fnames = glob(path_img+'thumbs/train/whale_*/*.jpg')
	nrow, ncol, nclass = len(fnames), w1 * h1 * 3, len(freq)
	rng = RandomState(290615)
	shuffled_idx = rng.permutation(range(nrow))
	data = h5py.File(path_img+'data/data.hdf5','w')

	# break_i = int(nrow * p_test)
	# X_train = data.create_dataset(
	# 	name='X_train',
	# 	shape=(break_i,ncol),
	# 	dtype='f64',
	# 	chunks=(chunk_size,ncol))
	# y_train = data.create_dataset(
	# 	name='y_train',
	# 	shape=(nrow-break_i,nclass),
	# 	dtype='i16',
	# 	chunks=(chunk_size,ncol),
	# 	fillvalue=0)
	# X_test = data.create_dataset(
	# 	name='X_test',
	# 	shape=(break_i,ncol),
	# 	dtype='f64',
	# 	chunks=(chunk_size,ncol))
	# y_test = data.create_dataset(
	# 	name='y_test',
	# 	shape=(break_i,nclass),
	# 	dtype='i16',
	# 	chunks=(chunk_size,ncol),
	# 	fillvalue=0)
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
		# if i < break_i:
		# 	X_test[i,] = asarray(im.resize((w1,h1))).ravel() / 255.
		# 	y_test[i,np.where(f_ == freq.index)[0][0]] = 1
		# else:
		# 	X_train[i,] = asarray(im.resize((w1,h1))).ravel() / 255.
		# 	y_train[i,np.where(f_ == freq.index)[0][0]] = 1
		if i % chunk_size == 0:
			print('Reading row %i' % i)
	data.close()
	print('Reading took %.1fs' % (time()-t_start))

