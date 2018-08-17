import pickle
import datetime
from skopt.space import Real, Integer, Categorical, Dimension, Identity
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt import dump, load
from skopt.callbacks import CheckpointSaver

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

def initialValues():
    x0 = []
    y0 = []
    x0.append( [5.657923701050716e-07, 88, 0.6313774502819828] )
    y0.append( 1091.47998046875 )
    x0.append( [1.2198035331185924e-09, 32, 0.2890843291817985] )
    y0.append( 1001.132080078125 )
    x0.append( [2.0025058389169895e-06, 96, 0.8774315789992737] )
    y0.append( 358.79248046875 )
    x0.append( [1.6767382135850956e-06, 213, 0.7180939581314286] )
    y0.append( 1114.944580078125 )
    x0.append( [4.4046697348276916e-08, 24, 0.1651150961445776] )
    y0.append( 971.1505737304688 )
    return (x0, y0)

def runParamTests(args):
    x0, y0 = initialValues()
    print( x0 )

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

    x0, y0 = initialValues()
    checkpoint_saver = CheckpointSaver("./hparam_checkpoint.pkl", compress=9)

    res_gp = gp_minimize(holder, space, x0=x0, y0=y0, n_calls=args.number, callback=[checkpoint_saver])
 
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
