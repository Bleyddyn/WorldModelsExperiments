'''
Train VAE model on data created using extract.py
final model saved into tf_vae/vae.json
'''

import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0" # can just override for multi-gpu systems

import tensorflow as tf
import random
import numpy as np
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

from keras.callbacks import EarlyStopping

from vae.vae import ConvVAE, reset_graph
from load_drives import DriveDataGenerator

def main( dirs, z_size=32, batch_size=100, learning_rate=0.0001, kl_tolerance=0.5, epochs=100, save_model=False, verbose=True, optimizer="Adam" ):

    if save_model:
        model_save_path = "tf_vae"
        if not os.path.exists(model_save_path):
          os.makedirs(model_save_path)

    gen = DriveDataGenerator(dirs, image_size=(64,64), batch_size=batch_size, shuffle=True, max_load=10000, images_only=True )
        
    num_batches = len(gen)

    reset_graph()

    vae = ConvVAE(z_size=z_size,
                  batch_size=batch_size,
                  learning_rate=learning_rate,
                  kl_tolerance=kl_tolerance,
                  is_training=True,
                  reuse=False,
                  gpu_mode=True,
                  optimizer=optimizer)

    early = EarlyStopping(monitor='loss', min_delta=0.1, patience=5, verbose=verbose, mode='auto')
    early.set_model(vae)
    early.on_train_begin()

    best_loss = sys.maxsize

    if verbose:
        print("epoch\tstep\tloss\trecon_loss\tkl_loss")
    for epoch in range(epochs):
        for idx in range(num_batches):
            batch = gen[idx]

            obs = batch.astype(np.float)/255.0

            feed = {vae.x: obs,}

            (train_loss, r_loss, kl_loss, train_step, _) = vae.sess.run([
              vae.loss, vae.r_loss, vae.kl_loss, vae.global_step, vae.train_op
            ], feed)
            
            if train_loss < best_loss:
                best_loss = train_loss

            if save_model:
                if ((train_step+1) % 5000 == 0):
                  vae.save_json("tf_vae/vae.json")
        if verbose:
            print("{} of {}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format( epoch, epochs, (train_step+1), train_loss, r_loss, kl_loss) )
        gen.on_epoch_end()
        early.on_epoch_end(epoch, logs={"loss": train_loss})
        if vae.stop_training:
            break
    early.on_train_end()


# finished, final model:
    if save_model:
        vae.save_json("tf_vae/vae.json")

    return best_loss

if __name__ == "__main__":

    import argparse
    import malpiOptions

    parser = argparse.ArgumentParser(description='Test data loader.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--z_size', type=int, default=32, help='Latent space size')
    parser.add_argument('--batch_size', type=int, default=100, help='Mini-batch size')
    parser.add_argument('--kl_tolerance', type=float, default=0.5, help='Max KL tolerance')
    parser.add_argument('--epochs', type=int, default=100, help='Mini-batch size')

    malpiOptions.addMalpiOptions( parser )
    args = parser.parse_args()
    malpiOptions.preprocessOptions(args)

    if args.test_only:
        runTests(args)
        exit()

    main( args.dirs, z_size=args.z_size, batch_size=args.batch_size, learning_rate=args.learning_rate, kl_tolerance=args.kl_tolerance, epochs=args.epochs, save_model=False, verbose=True )
