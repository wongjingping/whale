
# this script extracts the thumbnails given the specified 
# x and y coordinates from train.csv

from imports import *

path_img = '/Users/JP/Documents/whale/imgs/'
path_img = '/Users/yc/Downloads/whale/imgs/'

w1,h1 = 600,400 # resize all img to 600 x 400
w2,h2 = 100,100 # size of thumbnail
w2p,h2p = 3,3 # size of padding for 'extending image'

# load training fnames
train = pd.read_csv(path_img+'data/train.csv')
fnames = path_img + 'raw/' + train['Image']
whales = pd.DataFrame(data={
	'whale':pd.unique(train['whaleID']),
	'n_total':-1,
	'n_label':-1},
	columns=['whale','n_total','n_label'])

t_start = time()
for iw in whales.index:
	wID = whales.ix[iw]['whale']
	ss = train[train['whaleID'] == wID]
	ssl = ss[(~ss['x'].isnull()) & (~ss['y'].isnull())]
	whales.loc[iw,['n_total','n_label']] = len(ss), len(ssl)
	print(wID + '\tn_total:' + str(len(ss)) + '\tn_label:' + str(len(ssl)))
	for i in ssl.index:
		# load image and create folder for each wID 
		fname, f_ = re.search('(w_[0-9]*).jpg',train.ix[i]['Image']).group(0,1)
		im = Image.open(path_img+'raw/'+fname).resize((w1,h1))
		dir_i = path_img + 'thumbs/train/' + wID
		if not os.path.exists(dir_i):
			os.makedirs(dir_i)
		# set jitter ranges
		x, y = int(ssl.ix[i]['x']), int(ssl.ix[i]['y'])
		x_start = max(0, x-w2p)
		y_start = max(0, y-h2p)
		for x_ in range(x_start,x+1,w2p):
			for y_ in range(y_start,y+1,h2p):
				# save with 4 rotation settings
				thumb_i = im.crop((x_,y_,x_+w2+w2p,y_+h2+h2p))
				thumb_i = thumb_i.resize((w2,h2))
				thumb_i.save(dir_i+'/'+f_+'_0_'+str(x_)+'_'+str(y_)+'.jpg')
				thumb_i2 = thumb_i.transpose(Image.ROTATE_90)
				thumb_i2.save(dir_i+'/'+f_+'_90_'+str(x_)+'_'+str(y_)+'.jpg')
				thumb_i3 = thumb_i.transpose(Image.ROTATE_180)
				thumb_i3.save(dir_i+'/'+f_+'_180_'+str(x_)+'_'+str(y_)+'.jpg')
				thumb_i4 = thumb_i.transpose(Image.ROTATE_270)
				thumb_i4.save(dir_i+'/'+f_+'_270_'+str(x_)+'_'+str(y_)+'.jpg')
print('Took %.1fs to export %i thumbs' % \
	(time() - t_start, sum(~train.x.isnull())))
whales.to_csv(path_img+'data/whales.csv', index=False)

# annotate those with < 1 label
ss2 = whales[((whales.n_label < 2) & (whales.n_total > 1)) | \
	((whales.n_label < 1) & (whales.n_total > 0))]
fcheck = []
for iw in ss2.index:
	wID = ss2.ix[iw]['whale']
	fnames_i = train[train['whaleID']==wID]['Image'].values
	fcheck.extend(fnames_i.tolist())

# manually go over index i, adjust x,y where necessary
from img_select import savegood, plot_crop
from build_dataset import plot_pred
i = 0
fname = path_img+'raw/'+fcheck[i]
probs,x,y = plot_pred(fname, predict)
x,y = 230,180
plot_crop(fname,x,y)
savegood(fname,x,y)
save_wrong(fname,probs,x,y,threshold=0.4,w=6)
train.loc[train.Image==fcheck[i],['x','y']] = x,y
i +=1
fname = path_img+'raw/'+fcheck[i]
probs,x,y = plot_pred(fname, predict)

train.to_csv(path_img+'data/train.csv', index=False)


