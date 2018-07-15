from __future__ import print_function
import os
import sys
import pickle
from time import time
import argparse

import numpy as np

from keras.utils import Sequence
# https://keras.io/utils/#sequence

# For python2/3 compatibility when calling isinstance(x,basestring)
# From: https://stackoverflow.com/questions/11301138/how-to-check-if-variable-is-string-with-python-2-and-3-compatibility
try:
  basestring
except NameError:
  basestring = str

def normalize_images( images, default=True ):
    if default:
        rmean = 92.93206363205326
        gmean = 85.80540021330793
        bmean = 54.14884297660608
        rstd = 57.696159704394354
        gstd = 53.739380109203445
        bstd = 47.66536771313241

        #print( "Default normalization" )
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd
    else:
        rmean = np.mean(images[:,:,:,0])
        gmean= np.mean(images[:,:,:,1])
        bmean= np.mean(images[:,:,:,2])
        rstd = np.std(images[:,:,:,0])
        gstd = np.std(images[:,:,:,1])
        bstd = np.std(images[:,:,:,2])
        print( "Image means: {}/{}/{}".format( rmean, gmean, bmean ) )
        print( "Image stds: {}/{}/{}".format( rstd, gstd, bstd ) )

# should only do this for the training data, not val/test, but I'm not sure how to do that when Keras makes the train/val split
        images[:,:,:,0] -= rmean
        images[:,:,:,1] -= gmean
        images[:,:,:,2] -= bmean
        images[:,:,:,0] /= rstd
        images[:,:,:,1] /= gstd
        images[:,:,:,2] /= bstd

class VaeDataGenerator(Sequence):
    """ Loads data for training the VAE step of WorldModels
        From: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html"""
    def __init__(self, filelist, data_dir="record", batch_size=32, shuffle=True, max_load=30000 ):
        """ Input a list of files and a directory.
            Pre-load each to count number of samples.
            load one file and use it to generate batches until we run out.
            load the next file, repeat
            Re-shuffle on each epoch end
        """
        'Initialization'
        self.files = filelist
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_load = max_load
        self.image_norm = False
        self.next_dir_index = 0
        self.images = []
        self.current_start = 0
        self.count = self.__count()
        self.on_epoch_end()

    def __len__(self):
        'The number of batches per epoch'
        return int(np.floor(self.count / self.batch_size))

    def __getitem__(self, index):
        sample_beg = index * self.batch_size
        sample_beg -= self.current_start
        sample_end = sample_beg + self.batch_size
        #print( "getitem {} {}:{}".format( index, sample_beg, sample_end ) )

        if sample_end <= len(self.images):
            images = self.images[sample_beg:sample_end]
            return images

        if sample_beg <= len(self.images):
            images = self.images[sample_beg:]
            sample_end = len(self.images) - sample_beg
            self.images = self.__load_next_max()
            i2 = self.images[0:sample_end]
            images = np.append(images,i2,axis=0)
            return images

    def __load_next_max(self):

        self.current_start += len(self.images)

        images = []

        while len(images) <= self.max_load and self.next_dir_index < len(self.files):
            fname = os.path.join( self.data_dir, self.files[self.next_dir_index] )
            dimages = np.load(fname)['obs']
            if self.shuffle == True:
                np.random.shuffle(dimages)
            images.extend(dimages)
            self.next_dir_index += 1

        images = np.array(images)

        if self.image_norm:
            normalize_images(images)

        return images

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.files)
        self.next_dir_index = 0
        self.current_start = 0
        self.images = self.__load_next_max()

    def __count(self):
        count = 0
        for onefile in self.files:
            fname = os.path.join( self.data_dir, onefile )
            raw_data = np.load(fname)['obs']
            count += len(raw_data)
        return count

def runTests(args):
    pass

def getOptions():

    parser = argparse.ArgumentParser(description='Test data loader.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('directory', nargs=1, metavar="Directory", help='A directory containing recorded data')
    parser.add_argument('--test_only', action="store_true", default=False, help='run tests, then exit')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getOptions()

    if args.test_only:
        runTests(args)
        exit()

    filelist = os.listdir(args.directory[0])
    filelist.sort()


    gen = VaeDataGenerator( filelist, batch_size=32, shuffle=True, max_load=2000 )
    print( "# samples: {}".format( gen.count ) )
    print( "# batches: {}".format( len(gen) ) )
    for i in range(len(gen)):
        print( "Batch {}: {}".format( i, gen[i].shape ), end='\r' )
        sys.stdout.flush()
    print("")
