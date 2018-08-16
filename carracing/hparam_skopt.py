import pickle
import datetime
from skopt.space import Real, Integer, Categorical, Dimension, Identity
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import dump, load

from vae_train import main

class Holder:
    def __init__(self, dirs, log_to=None):
        self.dirs = dirs
        self.log_to = log_to
        self.vals = []
        self.run = 0

#    @use_named_args(space)
    def __call__(self, args):
        #print( args )
        learning_rate, batch_size, kl_tolerance = args
        arg_dict = { 'learning_rate': learning_rate,
                     'batch_size': batch_size,
                     'kl_tolerance': kl_tolerance }

        self.run += 1
        print( "Run {}".format( self.run ) )
        print( "   Args {}".format( args ) )
        loss = main( self.dirs, verbose=False, **arg_dict )
        self.vals.append(loss)
        print( "   Loss {}".format( loss ) )
        if self.log_to is not None:
            with open( self.log_to, 'a' ) as f:
                f.write( "{}\n".format( loss ) )
        return loss

def runParamTests(args):
    pass

if __name__ == "__main__":
    import argparse
    import malpiOptions

    parser = argparse.ArgumentParser(description='Hyperparameter Optimizer.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n', '--number', type=int, default=100, help='Number of test runs')

    malpiOptions.addMalpiOptions( parser )
    args = parser.parse_args()
    malpiOptions.preprocessOptions(args)

    #K.set_learning_phase(True)
    #setCPUCores( 4 )

    if args.test_only:
        runParamTests(args)
        exit()

    logging = True
    log_file = "skopt.txt"

    holder = Holder(args.dirs, log_to=log_file)

    max_batch = 256

    space  = [
              Real(10**-9, 10**-4, "log-uniform", name='learning_rate'),
              Integer(5, max_batch, name='batch_size'),
              #Integer(32, 32, name='z_size'),
              Real(0.1, 0.9, "uniform", name='kl_tolerance'),
              #Categorical(["RMSProp", "Adagrad", "Adadelta", "Adam"], name='optimizer'),
              ]

    if logging:
        with open( log_file, 'a' ) as f:
            f.write( "#{}\n".format( datetime.datetime.now() ) )

    res_gp = gp_minimize(holder, space, n_calls=args.number)
 
    print( "Best: {}".format( res_gp.fun ) )
    print("""Best parameters:
    - learning_rate=%.6f
    - batch_size=%d
    - z_size=%d
    - kl_tolerance=%.6f""" % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3] ))

    if logging:
        n = datetime.datetime.now()
        fname = n.strftime('hparam_skopt_%Y%m%d_%H%M%S.pkl')
        dump( res_gp, fname, store_objective=False )
