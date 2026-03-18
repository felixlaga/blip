import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from chainconsumer.chain import Chain
from chainconsumer.plotting.config import PlotConfig
from chainconsumer.statistics import SummaryStatistic
from chainconsumer.truth import Truth
#import healpy as hp
#from healpy import Alm
import pickle, argparse
#import logging
from src.hierarchical import postprocess
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def format_bound_title(parameter, bound):
    '''
    Format a posterior summary label for diagonal corner-plot axes.
    '''
    label_base = parameter[:-1] if parameter.endswith('$') else parameter
    label_suffix = '$' if parameter.endswith('$') else ''

    if bound.lower is None or bound.upper is None:
        center = bound.center
        if np.abs(center) <= 1e-3:
            center_form = '{0:.3e}'.format(center)
        else:
            center_form = '{0:.3f}'.format(center)
        return label_base + ' = ' + center_form + label_suffix

    err = [bound.upper - bound.center, bound.center - bound.lower]

    if np.abs(bound.center) <= 1e-3:
        mean_def = '{0:.3e}'.format(bound.center)
        eidx = mean_def.find('e')
        base = float(mean_def[0:eidx])
        exponent = int(mean_def[eidx+1:])
        mean_form = str(base) + ' \\times ' + '10^{' + str(exponent) + '}'
    else:
        mean_form = '{0:.3f}'.format(bound.center)

    if np.abs(err[0]) <= 1e-2:
        err[0] = '{0:.4f}'.format(err[0])
    else:
        err[0] = '{0:.2f}'.format(err[0])

    if np.abs(err[1]) <= 1e-2:
        err[1] = '{0:.4f}'.format(err[1])
    else:
        err[1] = '{0:.2f}'.format(err[1])

    return label_base + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}' + label_suffix

if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='postproc', usage='%(prog)s [options] rundir', description='run hierarchical postprocessing')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory.')
    parser.add_argument('--outdir', metavar='outdir', type=str, help='The path to the output directory Defaults to rundir.',default=None)
    parser.add_argument('--model', metavar='model', type=str, help='Parameterized spatial model to use.', default='breivik2020')
    parser.add_argument('--Nwalkers', metavar='Nwalkers', type=int, help='Number of walkers.', default=50)
    parser.add_argument('--Nsamples', metavar='Nsamples', type=int, help='Number of desired samples.', default=10000)
    parser.add_argument('--Nburn', metavar='Nburn', type=int, help='Number of desired burn-in samples.', default=1000)
    parser.add_argument('--seed', metavar='seed', type=int, help='Desired seed for the rng.', default=None)
    parser.add_argument('--Nthread', metavar='Nthread', type=int, help='Number of desired cores for multiprocessing.', default=1)
    # execute parser
    args = parser.parse_args()


    paramfile = open(args.rundir + '/config.pickle', 'rb')
    ## things are loaded from the pickle file in the same order they are put in
    params = pickle.load(paramfile)
    inj = pickle.load(paramfile)
    parameters = pickle.load(paramfile)
    ## initualize the postprocessing class
    postprocessor = postprocess(args.rundir,params,inj,parameters)
    ## run the sampler
    sampler = postprocessor.hierarchical_sampler(model=args.model,Nwalkers=args.Nwalkers,Nsamples=args.Nsamples,Nburn=args.Nburn,rng=args.seed,Nthread=args.Nthread)
    ## plot
    chain = sampler.flatchain
    ## model use cases
    knowTrue = False
    if args.model=='breivik2020':
        npar=2
        post_parameters = ['$r_h$','$z_h$']
        ## deal with older config files and assign true values if known
        if 'fg_type' in inj.keys():
            if inj['fg_type'] == 'breivik2020':
                knowTrue = True
                truevals = [inj['rh'],inj['zh']]
    else:
        raise TypeError("Unknown model. Currently supported models: 'breivik2020'.")
    cc = ChainConsumer()
    cc.add_chain(Chain(samples=pd.DataFrame(chain, columns=post_parameters), name='Posterior',
                       smooth=False, kde=False, statistics=SummaryStatistic.MAX_CENTRAL,
                       summary_area=0.95, sigmas=[1, 2], plot_cloud=False, bins=40))
    cc.set_plot_config(PlotConfig(max_ticks=2, label_font_size=18, tick_font_size=18,
                                  spacing=2, summarise=False, dpi=150))
    if knowTrue:
        cc.add_truth(Truth(location=dict(zip(post_parameters, truevals)), color='g',
                           line_style='--', alpha=0.7))

    fig = cc.plotter.plot(figsize=(16, 16))

    ## make axis labels to be parameter summaries
    sum_data = cc.analysis.get_summary()['Posterior']
    axes = np.array(fig.axes).reshape((npar, npar))

    # Adjust axis labels
    for ii in range(npar):
        ax = axes[ii, ii]

        # get the right summary for the parameter ii
        label = format_bound_title(post_parameters[ii], sum_data[post_parameters[ii]])
        ax.set_title(label, {'fontsize':18}, loc='left')

    ## save
    if args.outdir is None:
        fig.savefig(args.rundir  + '/postproc_corners.png', dpi=150)
        print("Posteriors plots printed in " + args.rundir + "/postproc_corners.png")
        plt.close(fig)
        np.savetxt(args.rundir+'/postprocessing_samples.txt',chain)
    else:
        fig.savefig(args.outdir  + '/postproc_corners.png', dpi=150)
        print("Posteriors plots printed in " + args.outdir + "/postproc_corners.png")
        plt.close(fig)
        np.savetxt(args.outdir+'/postprocessing_samples.txt',chain)
    






