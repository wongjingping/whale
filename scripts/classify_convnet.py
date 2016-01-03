
# this script builds a convnet for detecting if whale in thumb

from imports import *

model_name = None
arch = (3,5,5,500,447)
img_shape = (100,100)
field_size = (7,4)
maxpool_size = (2,2)
r = 0.03
d = 0.005
p_dropout = 1
epochs = 250
batch_size = 100
rng = RandomState(seed=290615)
srng = RandomStreams(seed=290615)
chunk_size=1000
i_train = 55000
save_par = True
print_freq = 1
save_progress = True
final = False




# trains a le-net model
def train_convnet(
	model_name=None,
	arch=(3,3,3,20,2), # input maps, C1 maps, C2 maps, MLP units, classes
	img_shape=(100,100), field_size=(7,4), maxpool_size=(2,2),
	r=0.03, d=0.005, p_dropout=0.5, epochs=250, batch_size=100, 
	rng=RandomState(290615), srng=RandomStreams(seed=290615),
	chunk_size=1000,i_train=55000,
	save_par=True, print_freq=1, save_progress=True, final=False):
	
	# shape input data
	nrow, ncol, nclass = X_.shape[0], X_.shape[1], y_.shape[1]

	print('... Building Model ...')
	t_build = time()

	# symbolic variables
	e = T.scalar('e', dtype='int32')
	i = T.scalar('i', dtype='int32')
	x = T.matrix('x', dtype='float64')
	y = T.imatrix('y')

	# batch_size, rgb channels, height, width
	A0 = x.reshape((x.shape[0],)+img_shape+(arch[0],)).dimshuffle(0,3,1,2)

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
	y_obs = T.argmax(y, axis=1)
	yhat_train = T.argmax(P_train, axis=1)
	yhat_test = T.argmax(P_test, axis=1)
	errors_train = T.mean(T.neq(yhat_train, y_obs))
	errors_test = T.mean(T.neq(yhat_test, y_obs))
	NLL_train = -T.mean(T.log(P_train)[T.arange(y.shape[0]), y_obs])
	NLL_test = -T.mean(T.log(P_test)[T.arange(y.shape[0]), y_obs])
	par_all = [C1_W,C1_b,C2_W,C2_b,W3,b3,W4,b4]
	g_all = T.grad(cost=NLL_train, wrt=par_all)
	lr = r/(d*e+1)
	updates = [(par, par - lr * gpar) for (par, gpar) in zip(par_all,g_all)]

	X_train = theano.shared(asarray(X_[:chunk_size,],dtype=theano.config.floatX), \
		borrow=True)
	y_train = T.cast(theano.shared(y_[:chunk_size,],borrow=True),'int32')

	train_sgd = theano.function(
		inputs=[e,i],
		outputs=[NLL_train,errors_train],
		updates=updates,
		givens={
			x: X_train[i*batch_size:(i+1)*batch_size,],
			y: y_train[i*batch_size:(i+1)*batch_size]
		})

	# not enough memory ><
	if not final:
		X_test = theano.shared(asarray(X_[i_train:,],dtype=theano.config.floatX), \
			borrow=True)
		y_test = T.cast(theano.shared(y_[i_train:,],borrow=True),'int32')
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
	n_batches = i_train / batch_size
	n_chunks = chunk_size / batch_size
	for it in range(epochs):
		t_e_start = time()
		NLL_train_v, e_train = zeros(n_batches), zeros(n_batches)
		for b in range(n_batches):
			if b % n_chunks == 0:
				X_train = theano.shared(asarray(
					X_[b*batch_size:b*batch_size+chunk_size,],
					dtype=theano.config.floatX), \
					borrow=True)
				y_train = T.cast(theano.shared(
					y_[b*batch_size:b*batch_size+chunk_size,],borrow=True),'int32')
			NLL_train_v[b], e_train[b] = train_sgd(it, b % n_chunks)
#			NLL_train[b], e_train[b], A3_b, A3_train_b, A3_test_b = train_sgd(it, b)
		progress[it,0:2] = [mean(NLL_train_v),mean(e_train)]
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
		
#	path_img = '/Users/JP/Documents/whale/imgs/'
#	path_img = '/Users/yc/Downloads/whale/imgs/'
	path_img = '/home/jp/whale/imgs/'
	
	# load data
	from load_hdf5 import *
	if os.path.isfile(path_img+'data/data.hdf5'):
		X_, y_ = load_classify_data(path_img+'data/data.hdf5')
	else:
		store_classify_data(chunk_size=1000,w1=100,h1=100)
		X_, y_ = load_classify_data(path_img+'data/data.hdf5')


	# test dropout
	p_drops = [1.0,0.7,0.5]
	res = []
	for p_drop in p_drops:
		(e_min, it_min) = train_convnet(arch=(3,3,3,40,2), print_freq=1, p_dropout=p_drop)
		res.append((e_min, it_min))
	
	arch = (3,3,3,20,2)
	train_convnet(arch=arch, print_freq=10, final=True)

