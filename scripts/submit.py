#!/usr/bin/env python

# this script takes in 2 models as arguments and generates a submission file

from imports import *





if __name__ == '__main__':

	path_img = '/Users/JP/Documents/whale/imgs/'
	# path_img = '/Users/yc/Downloads/whale/imgs/'
	# path_img = '/home/jp/whale/imgs/'

	[path1, path2] = sys.argv[1:3]

	# build models
	from detector_convnet import build_model
	with open(path_img + path1) as f:
		model1 = load(f)
	detector = build_model(model1, viz=False)

	from classify_convnet import build_model
	with open(path_img + path2) as f:
		model2 = load(f)
	classifier = build_model(model2, viz=False)

	# read in data
	submit = pd.read_csv(path_img + 'data/submit.csv')

	# sliding window parameters
	w1,h1 = 600,400
	w2,h2,d2 = 100,100,3
	window_len, x_step, y_step = 100, 10, 10
	nx,ny = (w1-window_len)/x_step, (h1-window_len)/y_step

	for idx in submit.index:
		t_i = time()
		w_ = submit['Image'][idx]
		im = Image.open(path_img + 'raw/' + w_)
		im_sml = im.resize((w1,h1))
		X = empty((nx*ny,w2*h2*d2))
		i = 0
		for y in xrange(0,h1-window_len,y_step):
			for x in xrange(0,w1-window_len,x_step):
				sub_img = im_sml.crop((x,y,x+window_len,y+window_len))
				X[i,] = asarray(sub_img).ravel() / 255.
				i += 1
		probs1 = detector(X).reshape(ny,nx)
		x,y = (probs1.argmax()%nx)*x_step, (probs1.argmax()/nx)*y_step
		im_crop = im_sml.crop((x,y,x+window_len,y+window_len))
		w = re.search('(w_[0-9]*).jpg',w_).group(1)
		im_crop.save('%sthumbs/test/%s_%i_%i.jpg' % (path_img,w,x,y))
		x_crop = asarray(im_crop).ravel() / 255.
		probs2 = classifier(x_crop.reshape((1,x_crop.shape[0])))
		submit.iloc[idx,1:] = probs2.ravel()
		print('%i %s took %.1fs' % (idx, w_, time()-t_i))
		# save every ten images processed
		if idx % 10 == 0:
			submit.to_csv(path_img + 'data/submit.csv', index=False)



