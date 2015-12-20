

path_img = '/Users/JP/Documents/whale/imgs/'
path_img = '/Users/yc/Downloads/whale/imgs/'

w1,h1 = 100,100

# reads in data from thumbs/train folder
# returns a list of matrices, chunk_size x 3 * w1 * h1
def load_data(chunk_size=20000,p_test=0.3,save=False):
	train = pd.read_csv(path_img+'data/train.csv')
	freq = train.groupby('whaleID').size()
	fnames = glob(path_img+'thumbs/train/whale_*/*.jpg')
	nrow, ncol, nclass = len(fnames), w1 * h1 * 3, len(freq)
	rng = RandomState(290615)
	shuffled_idx = rng.permutation(range(nrow))
	data = []
	t_start = time()
	# read data based on shuffled_idx
	print('Reading %i files in chunks of %i' % (nrow,chunk_size))
	for i in range(nrow):
		# allocate chunk of memory
		if i % chunk_size == 0:
			chunk_i = min(nrow-i,chunk_size)
			X = np.zeros((chunk_i,ncol),dtype=np.float64)
			y = np.zeros((chunk_i,nclass),dtype=np.int32)
		si = shuffled_idx[i]
		im = Image.open(fnames[si])
		ix = i % chunk_size
		X[ix,] = asarray(im.resize((w1,h1))).ravel() / 255.
		f_ = re.search('train/(whale_[0-9]*)/',fnames[si]).group(1)
		y[ix,np.where(f_ == freq.index)[0][0]] = 1
		# append chunk to data list
		if (i+1) % chunk_size == 0 or i == (nrow-1):
			break_i = int((1-p_test) * chunk_size)
			X_train = asarray(X[:break_i,])
			X_test = asarray(X[break_i:,])
			y_train = asarray(y[:break_i])
			y_test = asarray(y[break_i:])
			data.append(((X_train,y_train),(X_test,y_test)))
			print('Finished reading chunk %i' % ((i+1)/chunk_size))
		if i%1000 == 0:
			print('Reading row %i' % i)
	print('Reading took %.1fs' % (time()-t_start))
	if save:
		print('Writing data to '+path_img+'data/whale.pkl')
		t_start=time()
		with open(path_img+'data/whale.pkl','wb') as f:
			dump(data,f)
		print('Took %.1fs' % (time()-t))
	return data
