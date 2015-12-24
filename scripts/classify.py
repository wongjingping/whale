
from imports import *


# reads in data from thumbs/train folder
# returns a list of matrices, chunk_size x 3 * w1 * h1
def load_data(chunk_size=10000,p_test=0.3,w1=100,h1=100,save_chunks=True):
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
			chunk_nrow = min(nrow-i,chunk_size)
			X = np.zeros((chunk_nrow,ncol),dtype=np.float64)
			y = np.zeros((chunk_nrow,nclass),dtype=np.int32)
		si = shuffled_idx[i]
		im = Image.open(fnames[si])
		ix = i % chunk_size
		X[ix,] = asarray(im.resize((w1,h1))).ravel() / 255.
		f_ = re.search('train/(whale_[0-9]*)/',fnames[si]).group(1)
		y[ix,np.where(f_ == freq.index)[0][0]] = 1
		# append chunk to data list
		if (i+1) % chunk_size == 0 or i == (nrow-1):
			chunk_i = ((i+1)/chunk_size)
			break_i = int((1-p_test) * chunk_size)
			X_train = asarray(X[:break_i,])
			X_test = asarray(X[break_i:,])
			y_train = asarray(y[:break_i])
			y_test = asarray(y[break_i:])
			data_chunk = ((X_train,y_train),(X_test,y_test))
			if save_chunks:
				chunk_fname = path_img+'data/chunk_'+str(chunk_i)+'.pkl'
				print('Writing chunk %i to %s' % (chunk_i,chunk_fname))
				with open(chunk_fname,'wb') as f:
					dump(data_chunk,f)
			else:
				data.append(((X_train,y_train),(X_test,y_test)))
			print('Finished with chunk %i' % chunk_i)
		if i%1000 == 0:
			print('Reading row %i' % i)
	print('Reading took %.1fs' % (time()-t_start))
	if not save_chunks:
		return data

# helper function for sharing data
def make_shared(x,y):
		x_shared = theano.shared(asarray(x, dtype=theano.config.floatX),
			borrow=True)
		y_shared = theano.shared(asarray(y, dtype=theano.config.floatX),
			borrow=True)
		return (x_shared, T.cast(y_shared,'int32'))

# combine data in various chunks together



### ==================== Layer Classes ==================== ###


# builds and returns a convolution layer
class layer_conv(object):
	def __init__(self, x_in, W, shape, bound):
		if W is None:
			self.shape = shape
			self.W = theano.shared(
				value=asarray(
					a=rng.uniform(low=-bound, high=bound, size=self.shape),
					dtype=theano.config.floatX))
		else:
			self.W = theano.shared(value=W)
		self.x_out = conv.conv2d(input=x_in, filters=self.W)

# builds and returns a maxpool layer
class layer_maxpool(object):
	def __init__(self, x_in, poolsize, stride=None):
		stride = poolsize if stride is None else stride
		self.x_out = downsample.max_pool_2d(
			input=x_in,
			ds=(poolsize,poolsize), 
			st=(stride,stride),
			ignore_border=True)

# builds and returns a relu layer
class layer_relu(object):
	def __init__(self, x_in):
		self.x_out = x_in * (x_in > 0) # TODO update to theano's relu

# builds and returns a fully-connected hidden layer
class layer_fc(object):
	def __init__(self, x_in, W, b, shape, bound):
		if W is None:
			self.shape = shape
			self.W = theano.shared(
				value=asarray(
					rng.uniform(low=-bound, high=bound, size=shape),
					dtype=theano.config.floatX))
		else:
			self.W = theano.shared(value=W)
		if b is None:
			self.b = theano.shared(value=zeros((shape[1],), dtype=theano.config.floatX))
		else:
			self.b = theano.shared(value=b)
		self.x_out = T.dot(x_in,self.W) + self.b




# TODO re-write train_convnet to take in flexible architectures
def train_flex_convnet(
	arch,
	img_shape=(100,100,3),
	r=0.03, d=0.005, epochs=250, batch_size=100, 
	rng=RandomState(290615), srng=RandomStreams(seed=290615),
	save_par=True, print_freq=1, save_progress=True, final=False):

	n_batches = X_train.get_value().shape[0]/batch_size
	
	print('... Building Model ...')
	t_build = time()

	# symbolic variables
	e = T.scalar('e', dtype='int32')	
	b = T.scalar('b', dtype='int32')
	x = T.matrix('x')
	y = T.ivector('y')
	
	# batch_size, rgb channels, height, width
	x_in = x.reshape((x.shape[0],)+img_shape).dimshuffle(0,3,1,2)
	layers_train = []
	layers_test = []
	pars = []
	for i in range(len(arch)):
		if arch[i][0] == 'conv':
			n_filters, filter_size, channels = arch[i][1], arch[i][2], x_in.shape[1]
			bound = sqrt(6./((channels+n_filters)*(filter_size**2)))
			L = layer_conv(
				x_in=x_in,
				W=None,
				shape=(n_filters,channels,filter_size,filter_size),
				bound=bound)
			pars.append(L.W)
		elif arch[i][0] == 'maxpool':
			L = layer_maxpool(x_in=x_in,poolsize=arch[i][1],stride=arch[i][2])
		elif arch[i][0] == 'relu':
			L = layer_relu(x_in=x_in)
		elif arch[i][0] == 'fc':
			L = layer_fc(x_in=x_in,W=None)
		# mask for dropout
		mask = T.cast(srng.binomial(n=1,p=arch[i][3],size=(L.shape,)), \
		dtype=theano.config.floatX)
		layers_train.append(L * mask)
		layers_test.append(L * arch[i][3])
		x_in = L.x_out


	# TOCONTINUE

	# check if loading previous model's weights	
	if model_name is not None:
		with open(model_name) as f:
			model = load(f)
		model_pars, model_arch, model_tune = model
		arch, img_shape, field_size, maxpool_size = model_arch
		C1_W = theano.shared(value=model_pars[0],name='C1_W')
		C1_b = theano.shared(value=model_pars[1],name='C1_b')
		C1 = conv.conv2d(input=A0, filters=C1_W)
		M1 = downsample.max_pool_2d(
			input=C1,
			ds=(maxpool_size[0],maxpool_size[0]), 
			ignore_border=True)
		A1 = T.tanh(M1 + C1_b.dimshuffle('x',0,'x','x'))
		C2_W = theano.shared(value=model_pars[2],name='C2_W')
		C2_b = theano.shared(value=model_pars[3],name='C2_b')
		C2 = conv.conv2d(input=A1, filters=C2_W)
		M2 = downsample.max_pool_2d(
			input=C2,
			ds=(maxpool_size[0],maxpool_size[0]), 
			ignore_border=True)
		A2 = T.tanh(M2 + C2_b.dimshuffle('x',0,'x','x'))
		W3v, b3v, W4v, b4v = model_pars[4:8]

	else:
		# convolution layer 1
		C1_shape = (arch[1],arch[0],field_size[0],field_size[0])
		C1_bound = sqrt(6./((arch[0]+arch[1]/maxpool_size[0])*field_size[0]*field_size[0]))
		C1_W = theano.shared(
			value=asarray(
				a=rng.uniform(low=-C1_bound, high=C1_bound, size=C1_shape),
				dtype=theano.config.floatX),
			name='C1_W')
		C1_b = theano.shared(
			value=zeros(shape=(arch[1],), dtype=theano.config.floatX),
			name='C1_b')
		C1 = conv.conv2d(input=A0, filters=C1_W)
		
		# max pool layer 1
		M1 = downsample.max_pool_2d(
			input=C1,
			ds=(maxpool_size[0],maxpool_size[0]), 
			ignore_border=True)
		# tanh layer 1
		A1 = T.tanh(M1 + C1_b.dimshuffle('x',0,'x','x'))
		
		# convolution layer 2
		C2_shape = (arch[2],arch[1],field_size[1],field_size[1])
		C2_bound = sqrt(6./((arch[1]+arch[2]/maxpool_size[1])*field_size[1]*field_size[1]))
		C2_W = theano.shared(
			value=asarray(
				a=rng.uniform(low=-C2_bound, high=C2_bound, size=C2_shape),
				dtype=theano.config.floatX),
			name='C2_W')
		C2_b = theano.shared(
			value=zeros(shape=(arch[2],), dtype=theano.config.floatX),
			name='C2_b')
		C2 = conv.conv2d(input=A1, filters=C2_W)
		
		# max pool layer 2
		M2 = downsample.max_pool_2d(
			input=C2,
			ds=(maxpool_size[1],maxpool_size[1]), 
			ignore_border=True)
		# tanh layer 2
		A2 = T.tanh(M2 + C2_b.dimshuffle('x',0,'x','x'))
		
	# hidden layer 3
	n_in = arch[2]*(((img_shape[0]-field_size[0]+1)/2-field_size[1]+1)/2)**2
	W3_shape = (n_in,arch[3])
	W3_bound = sqrt(6./(n_in+arch[3]))
	if model_name is None:
		W3 = theano.shared(
			value=asarray(
				rng.uniform(low=-W3_bound, high=W3_bound, size=W3_shape),
				dtype=theano.config.floatX),
			name='W3', borrow=True)
		b3 = theano.shared(value=zeros((W3_shape[1],), dtype=theano.config.floatX),
						   name='b3', borrow=True)
	else:
		W3 = theano.shared(value=W3v, name='W3',borrow=True)
		b3 = theano.shared(value=b3v, name='b3',borrow=True)
	A3 = T.tanh(T.dot(A2.flatten(2),W3) + b3)
	
	# mask for dropout
	mask = T.cast(srng.binomial(n=1,p=p_dropout,size=(W3_shape[1],)), \
	dtype=theano.config.floatX)
	A3_train = A3 * mask
	A3_test = A3 * p_dropout
	
	# logistic layer 4
	W4_shape = (arch[3],arch[4])
	W4_bound = sqrt(6./(arch[3]+arch[4]))
	if model_name is None:
		W4 = theano.shared(
			value=asarray(
				rng.uniform(low=-W4_bound, high=W4_bound, size=W4_shape),
				dtype=theano.config.floatX),
			name='W4', borrow=True)
		b4 = theano.shared(value=zeros((W4_shape[1],), dtype=theano.config.floatX),
						   name='b4', borrow=True)
	else:
		W4 = theano.shared(value=W4v, name='W4',borrow=True)
		b4 = theano.shared(value=b4v, name='b4',borrow=True)
	
	# outputs, cost, gradients
	P_train = T.nnet.softmax(T.dot(A3_train, W4) + b4)
	P_test = T.nnet.softmax(T.dot(A3_test, W4) + b4)
	yhat_train = T.argmax(P_train, axis=1)
	yhat_test = T.argmax(P_test, axis=1)
	errors_train = T.mean(T.neq(yhat_train, y))
	errors_test = T.mean(T.neq(yhat_test, y))
	NLL_train = -T.mean(T.log(P_train)[T.arange(y.shape[0]), y])
	NLL_test = -T.mean(T.log(P_test)[T.arange(y.shape[0]), y])
	par_all = [C1_W,C1_b,C2_W,C2_b,W3,b3,W4,b4]
	g_all = T.grad(cost=NLL_train, wrt=par_all)
	lr = r/(d*e+1)
	updates = [(par, par - lr * gpar) for (par, gpar) in zip(par_all,g_all)]
	
	train_sgd = theano.function(
		inputs=[e,i],
		outputs=[NLL_train,errors_train],
		updates=updates,
		givens={
			x: X_train[i*batch_size:(i+1)*batch_size,],
			y: y_train[i*batch_size:(i+1)*batch_size]
		})
		
	if not final:
		test_model = theano.function(
			inputs=[],
			outputs=[NLL_test,errors_test],
			givens={x: X_test, y: y_test}
			)
	
	t_model = time()
	print('%.1fs elapsed' % (t_model-t_build))
	
	# train model
	print('... Training Model ...')
	e_min, it_min = 1., 0 
	progress = zeros((epochs,4))
	for it in range(epochs):
		t_e_start = time()
		NLL_train, e_train = zeros(n_batches), zeros(n_batches)
		for b in range(n_batches):
			NLL_train[b], e_train[b] = train_sgd(it, b)
#			NLL_train[b], e_train[b], A3_b, A3_train_b, A3_test_b = train_sgd(it, b)
		progress[it,0:2] = [mean(NLL_train),mean(e_train)]
		if not final:
			[NLL_test, e_test] = test_model()
			progress[it,2:4] = [NLL_test,e_test]
			if e_test < e_min:
				e_min, it_min = e_test, it
		if it % print_freq == 0:
			print('Epoch %i took %.1fs\n\tTrain NLL %.3f, error %.3f' % 
				(it,time()-t_e_start,mean(NLL_train), mean(e_train)))
			if not final:
				print('\tTest  NLL %.3f, error %.3f' % (NLL_test, e_test))
	t_train = time()
	print('%.1f min elapsed' % ((t_train-t_model)/60))
	if final:
		print('Best result at epoch %i: %.4f' % (it_min, e_min))

	# save model
	print('... Saving Parameters ...')	
	if save_par:
		model_pars = [p.get_value() for p in par_all]
		model_arch = (arch, img_shape, field_size, maxpool_size)
		model_tune = (r, d, epochs, batch_size, p_dropout, rng)
		model = (model_pars, model_arch, model_tune)
		fname = 'select/models/conv_'+'_'.join(str(a) for a in arch)+'_'+ \
			'_'.join(str(a) for a in model_tune[:5])+'.pkl'
		with open(path_img + fname,'wb') as f:
			dump(model,f)

	# save progress
	print('... Saving Progress ...')	
	if save_progress:		
		model_tune = (r, d, epochs, batch_size, p_dropout, rng)
		fname = 'select/models/conv_'+'_'.join(str(a) for a in arch)+'_'+ \
			'_'.join(str(a) for a in model_tune[:5])+'.csv'
		head = 'NLL_train,e_train,NLL_test,e_test'
		savetxt(path_img+fname, progress, delimiter=',', header=head)
	if not final:
		return((e_min, it_min))


if __name__ == 'main':

	path_img = '/Users/JP/Documents/whale/imgs/'
	path_img = '/Users/yc/Downloads/whale/imgs/'
	path_img = '/home/jp/whale/imgs/'

	load_data(chunk_size=10000,p_test=0.3,w1=100,h1=100,save_chunks=True)

	arch = [
	('conv',10,3,0.5),('maxpool',2,2,1),('relu',None,None,0.5),
	('conv',10,3,0.5),('maxpool',2,2,1),('relu',None,None,0.5),
	('fc')]




