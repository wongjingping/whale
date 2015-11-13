

from cPickle import load
import pandas as pd
from PIL import Image
from time import time
from numpy import empty, asarray, where
from re import search
from matplotlib import pyplot as plt
import pylab

from detector_convnet import build_model

path_img = '/Users/JP/Documents/whale/imgs/'

# load model
print('... Loading Convolutional Network ...')
with open(path_img+'select/models/conv_3_3_3_30_2_0.03_0.005_150_100.pkl') as f:
	model = load(f)
predict = build_model(model)

# load training fnames
train = pd.read_csv(path_img+'data/train.csv')
fnames = path_img + 'raw/' + train['Image']

# sliding window parameters
w1,h1 = 600,400
w2,h2 = 100,100
window_len, x_step, y_step = 100, 10, 10
nx,ny = (w1-window_len)/x_step, (h1-window_len)/y_step

# takes a pic file name, a predict function, and plots probs
def plot_pred(fname, predict):

	t_open = time()

	# slide over each w2*h2 region in steps of x_step, y_step
	# and determine the probability of finding a whale. Take max
	im = Image.open(fname)
	im_sml = im.resize((w1,h1))
	X = empty((nx*ny,w2*h2*3))
	i = 0
	for y in range(0,h1-window_len,y_step):
		for x in range(0,w1-window_len,x_step):
			sub_img = im_sml.crop((x,y,x+window_len,y+window_len))
			X[i,] = asarray(sub_img).ravel()
			i += 1
	probs = predict(X/255.).reshape(ny,nx)
	x,y = (probs.argmax()%nx)*x_step, (probs.argmax()/nx)*y_step
	thumb = im_sml.crop((x,y,x+window_len,y+window_len))


	# plot
	fig = plt.figure(figsize=(12,4))
	p1 = fig.add_subplot(131)
	p1.set_title('Resized Raw Image (400X600)')
	plt.imshow(im_sml)
	p2 = fig.add_subplot(132)
	p2.set_title('Probability of Finding a Whale\'s Head')
	plt.imshow(probs)
	p3 = fig.add_subplot(133)
	p3.set_title('Final Cropped Image')
	plt.imshow(thumb)
	plt.suptitle(fname+' spotted at '+str(x)+','+str(y))
	# pylab.savefig(path_img+'thumbs/'+search('w_[0-9]*',fname).group()+'_diag.jpg')

	print('File %s took %.1fs' % (fname,time()-t_open))
	return(probs, x, y)

def save_wrong(fname, probs, x, y, threshold=0.5, w=5):
	x,y = x/10,y/10
	im = Image.open(fname)
	im_sml = im.resize((w1,h1))
	iy, ix = where(probs > threshold)
	ix2 = ix[~((x-w < ix) & (ix < x+w) & (y-w < iy) & (iy < y+w))]
	iy2 = iy[~((x-w < ix) & (ix < x+w) & (y-w < iy) & (iy < y+w))]
	f_ = search('(w_[0-9]+)',fname).group(1)
	for i in range(len(ix2)):
		xi, yi = ix2[i]*10, iy2[i]*10
		thumb = im_sml.crop((xi,yi,xi+window_len,yi+window_len))
		thumb.save(path_img+'select/wrong/'+f_+'_'+str(xi)+'_'+str(yi)+'.jpg')
	print('saved ' + str(len(ix2)) + ' images, p > ' + str(threshold))

# manually go over index i, adjust x,y where necessary
from img_select import savegood, plot_crop
i = 400
fname = fnames[i]
probs,x,y = plot_pred(fname, predict)
x,y = 260,190
plot_crop(x,y)
savegood(fname,x,y)
save_wrong(fname,probs,x,y,threshold=0.4,w=5)
train.loc[i,['x','y']] = x,y
i +=1
fname = fnames[i]
probs,x,y = plot_pred(fname, predict)

train.to_csv(path_img+'data/train.csv', index=False)



# run this first before viz_image, viz_filters
# if 'model' not in vars():
# 	print('loading model')
# 	with open(path_img+'select/models/conv_3_3_3_20_2_0.02_0.005_200_100.pkl') as f:
# 		model = load(f)
# if 'predict' not in vars():
# 	print('building model')
# 	predict = build_model(model, viz=True)

# visualizes 100*100 RGB image
def viz_image(fname):
	f_re = search('(w_[0-9]+)_([0-9]+)_([0-9]+).jpg',fname)
	[f_,pos_x,pos_y] = [f_re.group(i) for i in range(1,4)]
	im = Image.open(fname)
	x = asarray(im).ravel()/255.
	x = x.reshape((1,x.shape[0]))
	[C1,M1,A1,C2,M2,A2,A3,P] = predict(x)
	fig = plt.figure(figsize=(14,4))
	p = fig.add_subplot(3,9,10)
	plt.imshow(im)
	plt.axis('off')
	p.set_title(f_, fontsize=16)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+2)
		Ci = C1[0,i,:,:]
		plt.imshow(Ci, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Convolution Layer 1, Map %i' % (i+1), fontsize=8)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+3)
		Mi = M1[0,i,:,:]
		plt.imshow(Mi, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Max Pool Layer 1, Map %i' % (i+1), fontsize=8)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+4)
		Ai = A1[0,i,:,:]
		plt.imshow(Ai, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Activation Layer 1, Map %i' % (i+1), fontsize=8)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+5)
		Ci = C2[0,i,:,:]
		plt.imshow(Ci, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Convolution Layer 2, Map %i' % (i+1), fontsize=8)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+6)
		Mi = M2[0,i,:,:]
		plt.imshow(Mi, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Max Pool Layer 2, Map %i' % (i+1), fontsize=8)
	for i in range(3):
		p = fig.add_subplot(3,9,i*9+7)
		Ai = A2[0,i,:,:]
		plt.imshow(Ai, cmap=plt.cm.gray)
		plt.axis('off')
		p.set_title('Activation Layer 2, Map %i' % (i+1), fontsize=8)
	p = fig.add_subplot(1,9,8)
	plt.imshow(A3.T)
	plt.xticks([])
	p.set_title('Activation Layer 3', fontsize=8)
	p = fig.add_subplot(1,9,9)
	p.set_title('Final Layer\'s Weights', fontsize=8)
	plt.imshow(model[0][6])
	plt.xticks([])
	plt.suptitle(f_+' spotted at '+pos_x+','+pos_y+\
		' with prob %.3f' % P, fontsize=24)
	plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.8,wspace=.2,hspace=.2)
	fig.savefig(path_img+'plots/convlayers_'+f_+'_'+pos_x+'_'+pos_y+'.jpg',
		transparent=True)

# visualizes filters of trained convnet
def viz_filters():
	f_ = 'select/models/conv_3_3_3_20_2_0.02_0.005_200_100.pkl'
	with open(path_img+f_) as f:
		model = load(f)
	model_pars, model_arch, model_tune = model
	arch, img_shape, field_size, maxpool_size = model_arch
	
	C1_W, C1_b, C2_W, C2_b, W3, b3, W4, b4 = model_pars
	nc, nr, n_in = 7, arch[3]/5, ((img_shape[0]-field_size[0]+1)/2-field_size[1]+1)/2
	fig = plt.figure(figsize=(14,4))
	plt.suptitle('Visualizing Filters')
	plt.subplots_adjust(left=.05,right=.95,bottom=.05,top=.8,wspace=.2,hspace=.2)
	for i in range(arch[1]):
		p = fig.add_subplot(arch[1],nc,i*nc+1)
		C1_i = asarray(255*C1_W[i,:,:,:].transpose(1,2,0), dtype='uint8')
		plt.imshow(C1_i)
		plt.axis('off')
		p.set_title('Convolution Layer 1, Filter %i' % (i+1), fontsize=8)
	for i in range(arch[2]):
		p = fig.add_subplot(arch[1],nc,i*nc+2)
		C2_i = asarray(255*C2_W[i,:,:,:].transpose(1,2,0), dtype='uint8')
		plt.imshow(C2_i)
		plt.axis('off')
		p.set_title('Convolution Layer 2, Filter %i' % (i+1), fontsize=8)
	i = 0
	for ix in range(2,nc):
		for iy in range(nr):
			p = fig.add_subplot(nr,nc,iy*nc+ix+1)
			W_i = asarray(255*W3[:,i].reshape((arch[2],n_in,n_in)).mean(axis=0),
				dtype='uint8')
			plt.imshow(W_i,cmap=plt.cm.gray)
			plt.axis('off')
			p.set_title('Activation Layer 3, Filter %i' % (i+1),fontsize=8)
			i += 1
	fig.savefig(path_img+'plots/convfilters_'+f_+'.jpg',
		transparent=True)




### =================== deprecated stuff =================== ###

# to recover previous cases which I didn't save into train yet:
train = pd.read_csv(path_img+'data/train.csv')
for i in range(110):
	fname = train['Image'][i]
	fnames = pd.Series(glob(path_img+'select/right/'+fname[:-4]+'*.jpg'))
	coords = fnames.str.extract('w_[0-9]+_([0-9]+)_([0-9]+).jpg').astype(int)
	x,y = coords.median()
	x,y = round(x,-1), round(y,-1)
	train.loc[i,['x','y']] = x,y


