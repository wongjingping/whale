
# this script prepares a dataset of 100x100x3 thumbnails
# containing whales / not. 

from glob import glob
from os import remove
from cPickle import dump
from PIL import Image
import matplotlib.pyplot as plt
from re import search
from random import sample, random
from numpy import empty, asarray, ones, zeros, concatenate, int32
from numpy.random import RandomState
import theano
import theano.tensor as T

path_img = '/Users/JP/Documents/whale/imgs/'
path_img = '/Users/yc/Downloads/whale/imgs/'
fnames = glob(path_img+'raw/*.jpg')

# example image
if 0:
	w0 = Image.open(path_img+'raw/w_0.jpg')
	plt.imshow(w0)

# get distribution of aspect ratio ~ 4 mins
def get_aspect_ratio():
	ar = empty(len(fnames))
	for i in range(len(fnames)):
		a = asarray(Image.open(fnames[i]))
		ar[i] = 1. * a.shape[1] / a.shape[0]
	plt.hist(ar) # all 3:2

# read jpeg files and sample smaller rectangles
def generate_windows():
	for fname in sample(fnames,50):
		im = Image.open(fname)
		len_x, len_y = im.size
		window_len = 512
		for x in range(0,len_x-window_len,200):
			for y in range(0,len_y-window_len,200):
				if random() > 0.1:
					continue
				fname_out = path_img+'select/'+search('w_[0-9]*',fname).group()+\
					'_'+str(x)+'_'+str(y)+'.jpg'
				sub_img = im.crop((x,y,x+window_len,y+window_len))
				sub_img.save(fname_out)
		print('Done with '+fname)


# manually filter from data/proc/select folder into /ok or /nope
# due to insufficient ok pics, need to manually crop nice pics to teach classifier

# filter for converting 3D rgb im to 2D grayscale array
def rgb2gray(im, red = 1.5, green = -0.5, blue = 0.):
	rgb = asarray(im)
	r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
	gray = red * r + green * g + blue * b
	return gray

def plot_im(fname=None):
	if not fname:
		fname = sample(fnames,1)[0]
	im = Image.open(fname)
	print('original res: ' + str(asarray(im).shape))
	im_sml = im.resize((600,400))
	plt.subplot(2,2,1)
	plt.imshow(im)
	plt.subplot(2,2,2)
	plt.imshow(im_sml)
	plt.subplot(2,2,3)
	plt.imshow(rgb2gray(im_sml,1.5,-0.5,0),cmap=plt.cm.gray)
	plt.suptitle(fname)
	return im,fname

def plot_crop(fname, x, y, window_len = 100):
	im = Image.open(fname)
	sub_img = im.resize((600,400)).crop((x,y,x+window_len,y+window_len))
	plt.imshow(sub_img)

# a variable window length means we need to resize all images
# before training whale detector

def savegood(fname, x, y, bnw = False, window_len = 100):
	fname_out = path_img+'select/right/'+search('w_[0-9]*',fname).group()
	# nudge image: jittered images for each image
	im = Image.open(fname)
	im_sml = im.resize((600,400))
	shift_x = shift_y = range(-3,4,3) # -3,0,+3
	nx, ny = len(shift_x), len(shift_y)
	for i in range(nx):
		for j in range(ny):
			x_, y_ = shift_x[i], shift_y[j]
			sub_img = im_sml.crop((x+x_,y+y_,x+x_+window_len,y+y_+window_len))
			if bnw:
				sub_img = Image.fromarray(rgb2gray(sub_img,1.5,-0.5,0)).convert('RGB')
			sub_img.save(fname_out+'_'+str(x+x_)+'_'+str(y+y_)+'.jpg')
			plt.subplot(nx,ny,i*nx+j+1)
			if bnw:
				plt.imshow(sub_img,cmap=plt.cm.gray)
			else:
				plt.imshow(sub_img)


def savenope(x, y, window_len = 100):
	fname_out = path_img+'select/nope/'+search('w_[0-9]*',fname).group()+\
	'_'+str(x)+'_'+str(y)+'_'
	# nudge image : 5*5 jittered image for each image
	im_sml = im.resize((600,400))
	shift_x = shift_y = range(-20,21,10) # -20,-10,0,+10,+20
	nx, ny = len(shift_x), len(shift_y)
	for i in range(nx):
		for j in range(ny):
			x_, y_ = shift_x[i], shift_y[j]
			sub_img = im_sml.crop((x+x_,y+y_,x+x_+window_len,y+y_+window_len))
			bnw_img = Image.fromarray(rgb2gray(sub_img,1.5,-0.5,0)).convert('RGB')
			bnw_img.save(fname_out+str(i*nx+j)+'.jpg')
			plt.subplot(nx,ny,i*nx+j+1)
			plt.imshow(bnw_img,cmap=plt.cm.gray)

# run these 3 lines of code to inspect images randomly
if 0:
	im, fname = plot_im()
	f = fname
	plot_crop(320,120) # change x,y manually
	savegood(270,250) # change x,y manually
	savenope(250,250)

# read in and format training data for whale classifier
def load_data(p_test=0.3):
	print('... Reading Input Matrix ...')
	names_ok = glob(path_img+'select/right/*jpg')
	names_nope = glob(path_img+'select/wrong/*jpg')
	names_all = names_ok + names_nope
	n_ok, n_nope = len(names_ok), len(names_nope)
	print('Ok: '+str(n_ok)+' Nope: '+str(n_nope))
	rng = RandomState(290615)
	nrow = n_ok + n_nope
	ncol = 100 * 100 * 3
	shuffled_idx = rng.permutation(range(nrow))
	X = zeros((nrow,ncol))
	y = zeros((nrow,),dtype=int32)
	y_all = asarray(concatenate([ones(n_ok),zeros(n_nope)]),dtype=int32)

	# read data based on shuffled_idx
	for i in range(nrow):
		si = shuffled_idx[i]
		im = Image.open(names_all[si])
		X[i,] = asarray(im).ravel()/255.
		y[i] = y_all[si]
		if i%1000 == 0:
			print('Reading row %i' % i)

	break_i = int((1-p_test)*nrow/100)*100
	X_train = asarray(X)[:break_i,]
	X_test = asarray(X)[break_i:,]
	y_train = asarray(y)[:break_i]
	y_test = asarray(y)[break_i:]

	def make_shared(x,y):
		x_shared = theano.shared(asarray(x, dtype=theano.config.floatX),
			borrow=True)
		y_shared = theano.shared(asarray(y, dtype=theano.config.floatX),
			borrow=True)
		return (x_shared, T.cast(y_shared,'int32'))
	return (make_shared(X_train,y_train), make_shared(X_test,y_test))



# saves variants of pictures that were manually identified to be correct
def save_right(shifts=2,window_len=100):
	fnames = glob(path_img+'select/right/*jpg')
	for fname in fnames:
		im = Image.open(fname)
		fname_out = path_img+'select/ok/'+search('w_[0-9]*',fname).group()+'_'
		# nudge image : 2 jittered images for each image
		im_sml = im.resize((window_len+shifts,window_len+shifts))
		for i in range(shifts):
			sub_img = im_sml.crop((i,i,i+window_len,i+window_len))
			bnw_img = Image.fromarray(rgb2gray(sub_img,1.5,-0.5,0)).convert('RGB')
			bnw_img.save(fname_out+str(i)+'.jpg')
			# plt.subplot(1,shifts,i+1)
			# plt.imshow(bnw_img,cmap=plt.cm.gray)

def save_wrong(shifts=3,window_len=100):
	fnames = glob(path_img+'select/wrong/*jpg')
	for fname in fnames:
		im = Image.open(fname)
		fname_out = path_img+'select/nope/'+search('w_[0-9]*',fname).group()+'_'
		# nudge image : 2 jittered images for each image
		im_sml = im.resize((window_len+shifts,window_len+shifts))
		for i in range(shifts):
			sub_img = im_sml.crop((i,i,i+window_len,i+window_len))
			bnw_img = Image.fromarray(rgb2gray(sub_img,1.5,-0.5,0)).convert('RGB')
			bnw_img.save(fname_out+str(i)+'.jpg')




### =================== deprecated methods =================== ###

# save data ~ 22s (not worth saving. faster to read from jpeg)
def save_data(X_train,y_train,X_test,y_test):
	print('Saving Input Matrix')
	whale = ((X_train,y_train),(X_test,y_test))
	with open(path_img+'select/whale.pkl','wb') as f:
		dump(whale,f)



# check filter for reducing rgb to gray
def check_filter(fnames):
	im_rgb = Image.open(sample(fnames,1)[0])
	im_gray = rgb2gray(asarray(im_rgb))
	plt.imshow(im_gray, cmap=plt.cm.gray)


# change old color images of various sizes into 100*100 bnw
def rgb2bnw100x100(fnames):
	for f in fnames:
		im = Image.open(f)
		im_sml = im.resize((100,100))
		bnw_img = Image.fromarray(rgb2gray(im_sml,1.5,-0.5,0)).convert('RGB')
		bnw_img.save(f[:-4]+'_fromcol.jpg')
		remove(f)

if(0):
	rgb2bnw100x100(glob(path_img+'select/ok/*jpg'))
	rgb2bnw100x100(glob(path_img+'select/nope/*jpg'))
	
	