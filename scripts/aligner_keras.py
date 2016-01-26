# -*- coding: utf-8 -*-
"""
Stores resized full images and their corresponding annotated points.
Builds a convnet to predict the bonnet, blowhole points

@author: jingpingw
"""

from imports import *
from keras.models import Sequential, model_from_json
from keras.layers.core import Reshape, Permute, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.regularizers import l2



# store resized images in hdf5
def store_data(nrow,ncol,ny,batch_size):
    t_start = time()
    rng = RandomState(290615)
    print('Storing data into hdf5')
    data = h5py.File(path_img+'data/xy.hdf5','w')
    X = data.create_dataset(
        name='X',
        shape=(nrow,ncol),
        dtype=np.float64,
        chunks=(batch_size,ncol))
    Y = data.create_dataset(
        name='Y',
        shape=(nrow,ny),
        dtype=np.float64,
        chunks=(batch_size,ny))
    si = rng.permutation(len(p1))
    for i in range(len(p1)):
        i_ = si[i]
        assert(p1[i_]['filename'] == p2[i_]['filename'])
        f_ = path_img + 'raw/' + p1[i_]['filename']
        im = Image.open(f_)
        im_sml = im.resize((w,h))
        X[i,:] = asarray(im_sml).ravel()/128.-1 # scale inputs [-1,+1]
        x1 = p1[i_]['annotations'][0]['x'] / im.size[0] * w
        y1 = p1[i_]['annotations'][0]['y'] / im.size[1] * h
        x2 = p2[i_]['annotations'][0]['x'] / im.size[0] * w
        y2 = p2[i_]['annotations'][0]['y'] / im.size[1] * h
        assert(x1 >= 0) & (x1 <= w) & (x2 >= 0) & (x2 <= w) & \
            ((y1 >= 0) & (y1 <= h) & (y2 >= 0) & (y2 <= h))
        Y[i,:] = asarray([x1,y1,x2,y2]) # scaled coords
        if i % 200 == 0:
            print('Processing %ith image' % i)
    data.close()
    print('Storing into hdf5 took %.0fs' % (time()-t_start))


# generator for image augmentation
def image_augmentor(X_,Y_,i_max):
    bi,b_list = 0,rng.permutation(i_max/batch_size)
    while True:
        b = b_list[bi]
        # t_aug = time()
        X_batch = X_[b*batch_size:(b+1)*batch_size,]
        Y_batch = Y_[b*batch_size:(b+1)*batch_size,]
        for ix in range(batch_size):
            # reshape, rescale pixel intensities to 0,1
            xi = (X_batch[ix,].reshape((h,w,ch))+1)/2
            x1,y1,x2,y2 = Y_batch[ix,]
            # flip coords horizontally
            if rng.rand(1)[0] > 0.5:
                xi,x1,x2 = xi[:,::-1,:],w-x1,w-x2
            # flip coords vertically
            if rng.rand(1)[0] > 0.5:
                xi,y1,y2 = xi[::-1,:,:],h-y1,h-y2
            # rescale slightly within a random range
            if rng.rand(1)[0] > 0.2:
                xleft,xright = min(x1,x2,30),min(w-x1,w-x2,30)
                yup,ydown = min(y1,y2,30),min(h-y1,h-y2,30)
                dx,dy = rng.uniform(-xleft,xright,1),rng.uniform(-yup,ydown,1)
                xl,xr = max(0,dx),min(255,255+dx)
                yu,yd = max(0,dy),min(255,255+dy)
                pts1 = np.float32([[xl,yu],[xr,yu],[xl,yd],[xr,yd]])
                pts2 = np.float32([[0,0],[255,0],[0,255],[255,255]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                xi = cv2.warpPerspective(xi,M,(w,h))
                # transform y
                if xl > 0:
                    x1,x2 = w-((w-x1)/(w-xl)*w),w-((w-x2)/(w-xl)*w)
                if xr < 255:
                    x1,x2 = x1/xr*w,x2/xr*w
                if yu > 0:
                    y1,y2 = h-((h-y1)/(h-yu)*h),h-((h-y2)/(h-yu)*h)
                if yd < 255:
                    y1,y2 = y1/yd*h,y2/yd*h
            # save back to X_batch
            X_batch[ix,] = (xi.ravel()*2)-1
            Y_batch[ix,] = x1,y1,x2,y2
        # print('Batch %i \nAugmentation took %.1fs' % (b,time()-t_aug))
        yield(X_batch,Y_batch)
        if bi < len(b_list)-1:
            bi += 1
        else:
            bi,b_list = 0,rng.permutation(i_max/batch_size)


# train a predictor for the keypoints (x1,y1,x2,y2)
def train_xy(epochs=50,batch_size=32,h=256,w=256,ch=3,
             train_p=0.8, valid_p=0.1):
    print('Compiling Model')
    t_comp = time()
    model = Sequential()
    # reshape input to ch, h, w (no sample axis)
    model.add(Reshape(dims=(h,w,ch),input_shape=(ch*h*w,)))
    model.add(Permute((3,1,2)))
    # add conv layers
    model.add(Convolution2D(16,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(32,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Convolution2D(64,3,3,init='glorot_uniform',activation='relu',
                            subsample=(1,1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(output_dim=2000,init='glorot_uniform',activation='relu',
                    W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=2000,init='glorot_uniform',activation='relu',
                    W_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim=4,init='glorot_uniform',activation='relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    t_train = time()
    print('Took %.1fs' % (t_train-t_comp))
    # split dataset
    i_test = int(train_p*nrow)/batch_size*batch_size
    i_valid = int(i_test*(1-valid_p))/batch_size*batch_size
    X_train, Y_train = X_[:i_valid,], Y_[:i_valid,]
    X_valid, Y_valid = X_[i_valid:i_test,], Y_[i_valid:i_test,]
    
    # naive fitting to lower rmse faster
    hist = model.fit(X_train,Y_train, batch_size=batch_size, nb_epoch=10, 
              verbose=1, validation_split=0.1)
    print(hist)
    # fit by batch using generator!
    img_aug = image_augmentor(X_,Y_,i_valid)
    hist = model.fit_generator(generator=img_aug,samples_per_epoch=i_valid,
                               nb_epoch=5000,verbose=1,
                        validation_data=(X_valid,Y_valid),nb_worker=1)
    rmse_test = model.evaluate(X_[i_test:,],Y_[i_test:,])
    print('Test RMSE: %.4f' % rmse_test)
    
    # save model
    model_json = model.to_json()
    open(path_img+'locate/model_116.json', 'w').write(model_json)
    model.save_weights(path_img+'locate/model_116_weights.h5')



if __name__ == '__main__':

    path_img = '/Users/jingpingw/Documents/whale/imgs/'
    
    # read in annotated points
    with open(path_img+'data/p1.json') as j1:
        p1 = json.load(j1)
    with open(path_img+'data/p2.json') as j2:
        p2 = json.load(j2)
    assert(len(p1)==len(p2))
    w,h,ch = 256,256,3
    nrow,ncol,ny,batch_size = len(p1),w*h*3,4,32

    # read in data from image files
    if not os.path.isfile(path_img+'data/xy.hdf5'):
        store_data(nrow,ncol,ny,batch_size)
    data = h5py.File(path_img+'data/xy.hdf5','r')
    X_, Y_ = data['X'], data['Y']

    # train    
    train_xy(epochs=50,batch_size=32,h=h,w=w,ch=3,train_p=0.8,valid_p=0.1)
    
    # predict
    model_best = model_from_json(open(path_img+'locate/model_116.json').read())
    model_best.load_weights(path_img+'locate/model_116_weights.h5')

    fnames = os.listdir(path_img+'raw/')
    fnames_train = [p1[i]['filename'] for i in range(len(p1))]
    fnames_test = list(set(fnames) - set(fnames_train))
    
    t = time()
    for i in range(10):
        f_ = path_img + 'raw/' + fnames_test[i]
        im = Image.open(f_)
        im_sml = im.resize((w,h))
        xi = asarray(im_sml,dtype=np.float32).reshape((1,h*w*ch))/128-1
        yi = model_best.predict(xi)
        x1,y1,x2,y2 = yi[0,]
        # diagnostic plot
        plt.imshow(im_sml)
        plt.scatter(x=[x1,x2],y=[y1,y2],c=['r','b'])
        plt.savefig(path_img+'crop/test/'+fnames_test[i])
        plt.close()
    time()-t