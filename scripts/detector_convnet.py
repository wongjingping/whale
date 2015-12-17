
# this script builds a convnet for detecting if whale in thumb


arch = (3,3,3,40,2)
img_shape = (100,100)
field_size = (7,4)
maxpool_size = (2,2)
r = 0.03
d = 0.005
p_dropout = 0.5
epochs = 250
batch_size = 100
rng = RandomState(seed=290615)
srng = RandomStreams(seed=290615)
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
	save_par=True, print_freq=1, save_progress=True, final=False):

	n_batches = X_train.get_value().shape[0]/batch_size
	
	print('... Building Model ...')
	t_build = time()

	# symbolic variables
	e = T.scalar('e', dtype='int32')	
	i = T.scalar('i', dtype='int32')
	x = T.matrix('x')
	y = T.ivector('y')
	
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
		A1 = T.tanh(M1 + C1_b.reshape((1,)+C1_b.shape+(1,1,)))
		C2_W = theano.shared(value=model_pars[2],name='C2_W')
		C2_b = theano.shared(value=model_pars[3],name='C2_b')
		C2 = conv.conv2d(input=A1, filters=C2_W)
		M2 = downsample.max_pool_2d(
			input=C2,
			ds=(maxpool_size[0],maxpool_size[0]), 
			ignore_border=True)
		A2 = T.tanh(M2 + C2_b.reshape((1,)+C2_b.shape+(1,1,)))
		W3, b3, W4, b4 = model_pars[4:8]
		A3 = T.tanh(T.dot(A2.flatten(2),W3) + b3)
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
		W3 = theano.shared(
			value=asarray(
				rng.uniform(low=-W3_bound, high=W3_bound, size=W3_shape),
				dtype=theano.config.floatX),
			name='W3', borrow=True)
		b3 = theano.shared(value=zeros((W3_shape[1],), dtype=theano.config.floatX),
						   name='b3', borrow=True)
		A3 = T.tanh(T.dot(A2.flatten(2),W3) + b3)
	
	# mask for dropout
	mask = T.cast(srng.binomial(n=1,p=p_dropout,size=(W3_shape[1],)), \
	dtype=theano.config.floatX)
	A3_train = A3 * mask
	A3_test = A3 * p_dropout
	
	# logistic layer 4
	W4_shape = (arch[3],arch[4])
	W4_bound = sqrt(6./(arch[3]+arch[4]))
	W4 = theano.shared(
		value=asarray(
			rng.uniform(low=-W4_bound, high=W4_bound, size=W4_shape),
			dtype=theano.config.floatX),
		name='W4', borrow=True)
	b4 = theano.shared(value=zeros((W4_shape[1],), dtype=theano.config.floatX),
					   name='b4', borrow=True)
	
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



def last_mile():
	train_sgd = theano.function(
		inputs=[e,i],
		outputs=[NLL,errors],
		updates=updates,
		givens={
			x: X_full[i*batch_size:(i+1)*batch_size,],
			y: y_full[i*batch_size:(i+1)*batch_size]
		})
		# train model
	print('... Training Model ...')
	e_min, it_min = 1., 0 
	for it in range(epochs):
		NLL_train, e_train = zeros(n_batches), zeros(n_batches)
		for b in range(n_batches):
			NLL_train[b], e_train[b] = train_sgd(it, b)
		if it % print_freq == 0:
			print('Epoch %i: \tTrain NLL %.3f, error %.3f' % 
				(it,mean(NLL_train), mean(e_train)))
	t_train = time()
	print('%.1f min elapsed' % ((t_train-t_model)/60))
	print('Best result at epoch %i: %.4f' % (it_min, e_min))

	# save model
	print('... Saving Parameters ...')	
	if save_par:
		model_pars = [p.get_value() for p in par_all]
		model_arch = (arch, img_shape, field_size, maxpool_size)
		model_tune = (r, d, epochs, batch_size, rng)
		model = (model_pars, model_arch, model_tune)
		fname = 'select/models/conv_'+'_'.join(str(a) for a in arch)+'_'+ \
			'_'.join(str(a) for a in model_tune[:4])+'.pkl'
		with open(path_img + fname,'wb') as f:
			dump(model,f)



# returns a predict function given model tuple defined in train_convnet
def build_model(model, viz=False):
	print('... Building Model ...')
	model_pars, model_arch, model_tune = model
	arch, img_shape, field_size, maxpool_size = model_arch
	
	x = T.matrix('x')
	A0 = x.reshape((x.shape[0],)+img_shape+(arch[0],)).dimshuffle(0,3,1,2)
	
	C1_W, C1_b = model_pars[0:2]
	C1 = conv.conv2d(input=A0, filters=C1_W)
	M1 = downsample.max_pool_2d(
		input=C1,
		ds=(maxpool_size[0],maxpool_size[0]), 
		ignore_border=True)
	A1 = T.tanh(M1 + C1_b.reshape((1,)+C1_b.shape+(1,1,)))

	C2_W, C2_b = model_pars[2:4]
	C2 = conv.conv2d(input=A1, filters=C2_W)
	M2 = downsample.max_pool_2d(
		input=C2,
		ds=(maxpool_size[0],maxpool_size[0]), 
		ignore_border=True)
	A2 = T.tanh(M2 + C2_b.reshape((1,)+C2_b.shape+(1,1,)))

	W3, b3, W4, b4 = model_pars[4:8]
	A3 = T.tanh(T.dot(A2.flatten(2),W3) + b3)
	P = T.nnet.softmax(T.dot(A3, W4) + b4)[:,1L]

	outs = [C1,M1,A1,C2,M2,A2,A3,P] if viz else P
	predict = theano.function(inputs=[x],outputs=outs)
	return(predict)




if __name__ == 'main':
		
	path_img = '/Users/JP/Documents/whale/imgs/'
#	path_img = '/Users/yc/Downloads/whale/imgs/'
	
	# load data
	from img_select import load_data
	t_start = time()
	(X_train,y_train),(X_test,y_test) = load_data()
	t_data = time()
	print('%.1fs elapsed' % (t_data-t_start))

	# test dropout
	p_drops = [1.0,0.7,0.5]
	res = []
	for p_drop in p_drops:
		(e_min, it_min) = train_convnet(arch=(3,3,3,40,2), print_freq=1, p_dropout=p_drop)
		res.append((e_min, it_min))
	
	arch = (3,3,3,20,2)
	train_convnet(arch=arch, print_freq=10, final=True)
