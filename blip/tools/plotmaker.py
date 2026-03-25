import sys, os
sys.path.append(os.getcwd()) ## this lets python find src
import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
try:
    from chainconsumer import ChainConsumer
    from chainconsumer.chain import Chain
    from chainconsumer.plotting.config import PlotConfig
    from chainconsumer.statistics import SummaryStatistic
    from chainconsumer.truth import Truth
except ImportError:
    ChainConsumer = None
    Chain = None
    PlotConfig = None
    SummaryStatistic = None
    Truth = None
try:
    import corner as corner_plot
except ImportError:
    corner_plot = None
import healpy as hp
from healpy import Alm
from astropy import units as u
import pickle, argparse
import logging
matplotlib.rcParams.update(matplotlib.rcParamsDefault)


def simple_interval(samples):
    '''
    Return the 95% central interval and median for a 1D posterior sample.
    '''
    return np.quantile(samples,0.025), np.median(samples), np.quantile(samples,0.975)


def build_chainconsumer_chain(post, parameters, name='Posterior'):
    '''
    Build a ChainConsumer chain from posterior samples using BLIP's default
    styling and a 95% max-central summary interval.
    '''
    samples = pd.DataFrame(post, columns=parameters)
    return Chain(samples=samples, name=name, smooth=False, kde=False,
                 statistics=SummaryStatistic.MAX_CENTRAL, summary_area=0.95,
                 sigmas=[1, 2], plot_cloud=False, bins=40)


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
        mean_form = str(base)
        exp_form = ' \\times ' + '10^{' + str(exponent) + '}'
    else:
        mean_form = '{0:.3f}'.format(bound.center)
        exp_form = ''

    if np.abs(err[0]) <= 1e-2:
        err[0] = '{0:.4f}'.format(err[0])
    else:
        err[0] = '{0:.2f}'.format(err[0])

    if np.abs(err[1]) <= 1e-2:
        err[1] = '{0:.4f}'.format(err[1])
    else:
        err[1] = '{0:.2f}'.format(err[1])

    return label_base + ' = ' + mean_form + '^{+' + err[0] + '}_{-' + err[1] + '}' + exp_form + label_suffix


def mapmaker(post, params, parameters, Model, saveto=None, coord=None, cmap=None, post_map_kwargs={}, med_map_kwargs={}):
    '''
    Function to create skymaps from the anisotropic search posteriors.
    
    Arguments
    ---------------
    post (array) : posterior samples
    params (dict) : params dictionary
    inj (dict) : injection params dictionary
    Model (Model object) : Combined Model used for the analysis.
    
    saveto (str) : /path/to/save/skymaps/ (Defaults to params['rundir']).
    coord (str) : Healpy coordinate choice. Defaults to 'E'.
    cmap (matplolib colormap) : Colormap to use for the skymaps.
    post_map_kwargs (dict) : kwargs to be passed to the marginalized posterior skymap mollview plot.
    med_map_kwargs (dict) : kwargs to be passed to the median posterior skymap mollview plot.
    
    '''
    
    
    
    
    sph_models = []
    hierarchical_models = []
    for submodel_name in Model.submodel_names:
        ## spatial type will be the latter part of the name
        ## also catch duplicates (with -N appended to them_)
        spatial_name = submodel_name.split('_')[-1].split('-')[0]
        if spatial_name == 'sph':
            sph_models.append(submodel_name)
        elif spatial_name == 'hierarchical':
            hierarchical_models.append(submodel_name)
    if (len(sph_models)==0 ) and (len(hierarchical_models)==0):
        print("Called mapmaker but none of the recovery models have a non-isotropic spatial model. Skipping...")
        return
    
    
    ## handle projection, kwargs
    # setting coord back to E, if parameter isn't specified
    if coord is None:
        if 'projection' in params.keys():
            coord = ['E',params['projection']]
        else:
            coord = 'E'
    else:
        coord = ['E',coord]
    # handling titles, units
    post_base_kwargs = {'title':'Marginalized posterior skymap of $\\Omega(f= 1mHz)$','unit':"$\\Omega(f= 1mHz)$"}
    med_base_kwargs = {'title':'Median skymap of $\\Omega(f= 1mHz)$','unit':"$\\Omega(f= 1mHz)$"}
    for kwarg_dict, base_dict in zip([post_map_kwargs,med_map_kwargs],[post_base_kwargs,med_base_kwargs]):
        kwarg_dict = base_dict | kwarg_dict
    
    nside = params['nside']

    npix = hp.nside2npix(nside)

        
    start_idx = 0   
    for submodel_name in Model.submodel_names:
        ## grab submodel
        sm = Model.submodels[submodel_name]
        
        # Initialize power skymap
        omega_map = np.zeros(npix)
        
        ## only make a map if there's a map to make (this is also good life advice)
        if submodel_name in sph_models+hierarchical_models:
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger = logging.getLogger()
            logger.setLevel(logging.ERROR)
            
            ## select relevant posterior columns
            post_i = post[:,start_idx:(start_idx+sm.Npar)]
            
            print("Computing marginalized posterior skymap for submodel: {}...".format(submodel_name))
            
            for ii in range(post.shape[0]):
                
                ## get Omega(f=1mHz)
                Omega_1mHz = sm.omegaf(1e-3,*post_i[ii,:sm.blm_start])
                
                ## convert blm params to full blms
                blm_vals = sm.blm_params_2_blms(post_i[ii,sm.blm_start:])
                
                ## normalize, convert to map, and sum
                norm = np.sum(blm_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_vals[(sm.lmax + 1):])**2)
                
                prob_map  = (1.0/norm) * (hp.alm2map(blm_vals, nside))**2
                
                omega_map = omega_map + Omega_1mHz * prob_map

            omega_map = omega_map/post.shape[0]
            
            
            # generating skymap
            hp.mollview(omega_map, coord=coord, cmap=cmap, **post_map_kwargs)
            hp.graticule()
            
            ## switch logging level back to normal so we get our own status updates
            logger.setLevel(logging.INFO)
            
            if saveto is not None:
                plt.savefig(saveto + '/{}_post_skymap.png'.format(submodel_name), dpi=150)
                logger.info('Saving posterior skymap at ' +  saveto + '/{}_post_skymap.png'.format(submodel_name))
        
            else:
                plt.savefig(params['out_dir'] + '/{}_post_skymap.png'.format(submodel_name), dpi=150)
                logger.info('Saving posterior skymap at ' +  params['out_dir'] + '/{}_post_skymap.png'.format(submodel_name))
            plt.close()
            
            ## now do the median skymap
            print("Computing median posterior skymap for submodel {}...".format(submodel_name))
            
            ## HEALpy is really, REALLY noisy sometimes. This stops that.
            logger.setLevel(logging.ERROR)
            
            # median values of the posteriors
            med_vals = np.median(post_i, axis=0)
            
            # Omega(f=1mHz)
            Omega_1mHz_median = sm.omegaf(1e-3,*med_vals[:sm.blm_start])
            ## blms.
            blms_median = np.append([1], med_vals[sm.blm_start:])
            
            blm_median_vals = sm.blm_params_2_blms(blms_median)
        
            norm = np.sum(blm_median_vals[0:(sm.lmax + 1)]**2) + np.sum(2*np.abs(blm_median_vals[(sm.lmax + 1):])**2)

            Omega_median_map  =  Omega_1mHz_median * (1.0/norm) * (hp.alm2map(blm_median_vals, nside))**2
            
            hp.mollview(Omega_median_map, coord=coord, cmap=cmap, **med_map_kwargs)
            
            hp.graticule()
            
            ## switch logging level back to normal so we get our own status updates
            logger.setLevel(logging.INFO)
            
            if saveto is not None:
                plt.savefig(saveto + '/post_median_skymap.png', dpi=150)
                logger.info('Saving injected skymap at ' +  saveto + '/post_median_skymap.png')
        
            else:
                plt.savefig(params['out_dir'] + '/post_median_skymap.png', dpi=150)
                logger.info('Saving injected skymap at ' +  params['out_dir'] + '/post_median_skymap.png')
        
            plt.close()
        
            
        
        ## increment start regardless of if we made a map
        start_idx += sm.Npar
    
    
    return
    
    
    


def fitmaker(post,params,parameters,inj,Model,Injection=None,saveto=None,plot_convolved=True,astro_kwargs={},det_kwargs={}):
    
    '''
    Make a plot of the spectral fit from the samples generated by the mcmc/nested sampling algorithm.

    Parameters
    -----------

    post : array
        Posterior samples
    
    params : dictionary
        Dictionary of config params

    parameters: string
        Array or list of strings with names of the parameters

    inj : dictionary
        Dictionary of injection params
        
    Model : Model object
        The federated Model used for the analysis
    
    Injection : Injection object
        The federated Injection used to create the data.
    
    *_kwargs : dict
        Keyword argument dictionaries for tweaking the astrophysical/detector plots. Limited number of attributes are supported.
        Supported attributes: figsize, dpi, color_dict, title, title_fontsize, xlabel, xlabel_fontsize, ylabel, ylabel_fontsize, xmin, xmax, ymin, ymax.
        Most of the above are the associated matplotlib argument. The exception is 'color_dict', which should be of the form {'submodel_name':'colorname'}
            and can be used to specify the desired plotting color for specific submodels.
    '''
    
    ## check that an injection was specified if we're not using external data
    if not params['load_data']:
        if Injection is None:
            print("Warning: Not using externally generated data, but no Injection object has been provided to the fitmaker. Returning without making plots...")
            return
    
    ## build the default plot kwargs
    default_kwargs = {'figsize':None,'dpi':150,'color_dict':{},'title':None,'title_fontsize':None,
                      'xlabel':'Frequency [Hz]','xlabel_fontsize':None,'ylabel':'PSD [1/Hz]','ylabel_fontsize':None,
                      'xmin':None,'xmax':None,'ymin':None,'ymax':None}
    ## update astro kwargs
    astro_kwargs = {'title':"Fit vs. Injection (Astrophysical)"} | astro_kwargs
    astro_kwargs = default_kwargs | astro_kwargs
    ## update det kwargs
    det_kwargs = {'title':"Fit vs. Injection (in Detector)"} | det_kwargs
    det_kwargs = default_kwargs | det_kwargs

    if getattr(Model, 'fixedL_channel_mode', False):
        print("Saving fixed-L channel diagnostics...")
        destination = saveto if saveto is not None else params['out_dir']
        Model.save_fixedL_channel_diagnostics(post, destination)
        return
    
    print("Computing spectral fit median and 95% CI...")
    ## get samples
    
    ## the population injection looks funky with a dashed line, but we still need to make it clear that it's an injection.
    ## this makes the Notation Legend "Injection" label be a split dashed/solid line
    
    if params['load_data']:
        notation_legend_elements = [Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Median Fit','$95\%$ C.I.']
        notation_handler_map = {}
        notation_handlelength = None
    elif 'population' in Injection.component_names:
        notation_legend_elements = [(Line2D([0], [0], color='k', ls='--'),Line2D([0], [0], color=Injection.components['population'].color,ls='-',lw=0.75,alpha=0.8)),
                                    Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Injection','Median Fit','$95\%$ C.I.']
        notation_handler_map = {tuple: HandlerTuple(ndivide=None)}
        notation_handlelength = 3
    else:
        notation_legend_elements = [Line2D([0], [0], color='k', ls='--'),
                                    Line2D([0], [0], color='k', ls='-'),
                                    Patch(color='k',alpha=0.25)]
        notation_legend_labels = ['Injection','Median Fit','$95\%$ C.I.']
        notation_handler_map = {}
        notation_handlelength = None
    
    ## get frequencies
    frange = Model.fs
    ffilt = np.logical_and(frange >= params['fmin'], frange <= params['fmax'])
#    ffilt = (frange>params['fmin'])*(frange<params['fmax'])
    ## commenting for testing version
#    fs = frange[ffilt][::10]
    fs = frange[ffilt]
    fs = fs.reshape(-1,1)

    
    ## make the deconvolved spectral fit plot
    plt.figure(figsize=astro_kwargs['figsize'])
    
    ## plot our recovered spectra
    if 'noise' in Model.submodel_names:
        start_idx = 2
    else:
        start_idx = 0
    
    model_legend_elements = []
    ymins = []
    ## loop over submodels
    signal_model_names = [sm_name for sm_name in Model.submodel_names if sm_name!='noise']
    if len(signal_model_names) > 0:
        signal_aliases = [Model.submodels[sm_name].alias for sm_name in signal_model_names if hasattr(Model.submodels[sm_name],"alias")]
        for i, sm_name in enumerate(signal_model_names):
            sm = Model.submodels[sm_name]
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            ## this grabs the relevant bits of the posterior vector for each model
            ## will need to fix this for the anisotropic case later...
            post_sm = [post[:,idx] for idx in range(start_idx,start_idx+sm.Npar)]
            ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
            if hasattr(sm,"blm_start"):
                post_sm = post_sm[:sm.blm_start]
            start_idx += sm.Npar
            ## the spectrum of every sample
            Sgw = sm.compute_Sgw(fs,post_sm)
            ## get summary statistics
            ## median and 95% C.I.
            Sgw_median = np.median(Sgw,axis=1)
            Sgw_upper95 = np.quantile(Sgw,0.975,axis=1)
            Sgw_lower95 = np.quantile(Sgw,0.025,axis=1)
            ymins.append(Sgw_median.min())
            ymins.append(Sgw_lower95.min())
            ## plot
            plt.loglog(fs,Sgw_median,color=sm.color)
            plt.fill_between(fs.flatten(),Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)

        if not params['load_data']:
            ## plot the injected spectra, if known
            for component_name in Injection.component_names:
                if component_name != 'noise':
                    ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                    kwargs = {'ls':'--','color':Injection.components[component_name].color,
                              **Injection.components[component_name].plot_kwargs}
                    ## overwrite color if specified in the the high-level kwargs
                    if component_name in astro_kwargs['color_dict'].keys():
                        kwargs['color'] = astro_kwargs['color_dict'][component_name]
                    Injection.plot_injected_spectra(component_name,fs_new=fs,legend=False,ymins=ymins,**kwargs)
                    if component_name not in Model.submodel_names and component_name not in signal_aliases:
                        model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))

        ## avoid plot squishing due to signal spectra with cutoffs, etc.
        if astro_kwargs['ymin'] is None:
            ymin = np.min(ymins)
            if ymin < 1e-43:
                plt.ylim(bottom=1e-43)
        else:
            plt.ylim(bottom=astro_kwargs['ymin'])
        plt.ylim(top=astro_kwargs['ymax'])

        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                    handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)

        plt.title(astro_kwargs['title'],fontsize=astro_kwargs['title_fontsize'])
        plt.xlabel(astro_kwargs['xlabel'],fontsize=astro_kwargs['xlabel_fontsize'])
        plt.ylabel(astro_kwargs['ylabel'],fontsize=astro_kwargs['ylabel_fontsize'])
        if saveto is not None:
            plt.savefig(saveto + '/spectral_fit_astro.png', dpi=astro_kwargs['dpi'])
        else:
            plt.savefig(params['out_dir'] + '/spectral_fit_astro.png', dpi=astro_kwargs['dpi'])
        print("Astrophysical spectral fit plot saved to " + params['out_dir'] + "spectral_fit_astro.png")
        plt.close()
    
    ## plot our recovered convolved spectra if desired
    if plot_convolved:
        model_legend_elements = []
        ymins = []
        plt.figure(figsize=det_kwargs['figsize'])

        start_idx = 0
        ## loop over submodels
        for sm_name in Model.submodel_names:
            sm = Model.submodels[sm_name]
            
            model_legend_elements.append(Line2D([0],[0],color=sm.color,lw=3,label=sm.fancyname))
            
            fdata = sm.fs
#            filt = (fdata>params['fmin'])*(fdata<params['fmax'])
            filt = np.logical_and(frange >= params['fmin'], frange <= params['fmax'])
            fdata = fdata[filt]
            f0 = sm.f0[filt]

            ## the spectrum of every sample
            ## for memory's sake, this needs to be a for loop
            Sgw = np.zeros((post.shape[0],len(fdata)))
            for jj in range(post.shape[0]):
                post_sm = post[jj,start_idx:start_idx+sm.Npar]
                ## handle noise and gw differently, but they all ended up named Sgw. Oh well.
                if sm_name == 'noise':
                    Np = 10**post_sm[0]
                    Na = 10**post_sm[1]
                    Sgw_j = sm.instr_noise_spectrum(fdata,f0,Np=Np,Na=Na)[2,2,:]
                ## handle any additional spatial variables (will need to fix this when I introduce hierarchical models)
                elif hasattr(sm,"blm_start"):
                    post_sm_sph = post_sm[sm.blm_start:]
                    post_sm = post_sm[:sm.blm_start]
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.compute_summed_response(sm.compute_skymap_alms(post_sm_sph))[0,0,filt,:],axis=1)
                else:
                    Sgw_j = np.mean(sm.compute_Sgw(fdata,post_sm)[:,None] * sm.response_mat[0,0,filt,:],axis=1)
                
                Sgw[jj,:] = np.real(Sgw_j)
            start_idx += sm.Npar
            ## get summary statistics
            ## median and 95% C.I.
            Sgw_median = np.median(Sgw,axis=0)
            Sgw_upper95 = np.quantile(Sgw,0.975,axis=0)
            Sgw_lower95 = np.quantile(Sgw,0.025,axis=0)
            ymins.append(Sgw_median.min())
            ymins.append(Sgw_lower95.min())
            ## plot
            plt.loglog(fdata,Sgw_median,color=sm.color)
            plt.fill_between(fdata,Sgw_lower95,Sgw_upper95,alpha=0.25,color=sm.color)
            
            
        ## now make the convolved spectral fit
        
        if not params['load_data']:
            ## plot the injected spectra, if known
            for component_name in Injection.component_names:
                ## this will overwrite the default linestyle if 'ls' is given in cm.plot_kwargs
                kwargs = {'ls':'--','color':Injection.components[component_name].color,
                          **Injection.components[component_name].plot_kwargs}
                ## overwrite color if specified in the the high-level kwargs
                if component_name in det_kwargs['color_dict'].keys():
                    kwargs['color'] = det_kwargs['color_dict'][component_name]
                if component_name == 'noise':
                    Injection.plot_injected_spectra(component_name,channels='22',ymins=ymins,**kwargs)
                else:
                    Injection.plot_injected_spectra(component_name,fs_new=fdata,convolved=True,ymins=ymins,**kwargs)
                    if component_name not in Model.submodel_names and component_name not in signal_aliases:
                        model_legend_elements.append(Line2D([0],[0],color=Injection.components[component_name].color,lw=3,label=Injection.components[component_name].fancyname))
        
        ## avoid plot squishing due to signal spectra with cutoffs, etc.
        if det_kwargs['ymin'] is None:
            if len(ymins) > 0:
                ymin = np.min(ymins)
                if ymin < 1e-43:
                    plt.ylim(bottom=1e-43)
        else:
            plt.ylim(bottom=det_kwargs['ymin'])
        plt.ylim(top=det_kwargs['ymax'])
        
        ax = plt.gca()
        model_legend = ax.legend(handles=model_legend_elements,loc='upper right')
        ax.add_artist(model_legend)
        N_models = len(model_legend_elements)
        notation_legend = ax.legend(handles=notation_legend_elements,labels=notation_legend_labels,handler_map=notation_handler_map,
                                    handlelength=notation_handlelength,loc='upper right',bbox_to_anchor=(1,0.9825-0.056*N_models))
        ax.add_artist(notation_legend)
        
        plt.title(det_kwargs['title'],fontsize=det_kwargs['title_fontsize'])
        plt.xlabel(det_kwargs['xlabel'],fontsize=det_kwargs['xlabel_fontsize'])
        plt.ylabel(det_kwargs['ylabel'],fontsize=det_kwargs['ylabel_fontsize'])
        if saveto is not None:
            plt.savefig(saveto + '/spectral_fit_detector.png', dpi=det_kwargs['dpi'])
        else:
            plt.savefig(params['out_dir'] + '/spectral_fit_detector.png', dpi=det_kwargs['dpi'])
        print("Detector spectral fit plot saved to " + params['out_dir'] + "spectral_fit_detector.png")
        plt.close()
 
    
    return
    

  
def plotmaker(post, params,parameters, inj, Model, Injection=None,saveto=None):

    '''
    Make posterior plots from the samples generated by tge mcmc/nested sampling algorithm.

    Parameters
    -----------
    post : array
        Collection of posterior samples.

    params : dictionary
        Dictionary of config params

    parameters: string or dict
        Dictionary or list of strings with names of the parameters

    npar : int
        Dimensionality of the parameter space
    '''

    all_parameters = Model.parameters['all']
    
    ## get truevals if not using an external injection
    if not params['load_data']:
        if Injection is None:
            print("Warning: Not using externally generated data, but no Injection object has been provided to the corner plotmaker. Returning without making plots...")
            return
        
        inj_truevals = Injection.truevals
        
        truevals = {}
        for smn in Model.submodel_names:
            for cmn in Injection.component_names:
                if smn == cmn or (hasattr(Model.submodels[smn],"alias") and Model.submodels[smn].alias == cmn):
                    truevals |= {param:inj_truevals[cmn][param] for param in Model.submodels[smn].parameters if param in inj_truevals[cmn].keys()}
                    
        if len(truevals) > 0:
            knowTrue = 1 ## Bit for whether we know the true vals or not
        else:
            knowTrue = 0
    else:
        knowTrue = 0
    
    npar = Model.Npar

    if params['out_dir'][-1] != '/':
        params['out_dir'] = params['out_dir'] + '/'
    
    if ChainConsumer is None:
        if corner_plot is not None:
            print("ChainConsumer is not installed. Falling back to the 'corner' library for posterior plots.")
            truths = None
            if knowTrue:
                truths = [truevals.get(parameter, None) for parameter in all_parameters]

            fig = corner_plot.corner(
                post,
                labels=all_parameters,
                truths=truths,
                bins=40,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=False,
                title_fmt='.3g',
                plot_datapoints=False,
                fill_contours=True,
                color='slateblue',
                levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)),
                hist_kwargs={'color': 'slateblue', 'alpha': 0.85},
            )

            axes = np.array(fig.axes).reshape((npar, npar))
            for ii in range(npar):
                qlo, qmed, qhi = np.quantile(post[:, ii], [0.16, 0.5, 0.84])
                err_hi = qhi - qmed
                err_lo = qmed - qlo
                label = all_parameters[ii][:-1] if all_parameters[ii].endswith('$') else all_parameters[ii]
                suffix = '$' if all_parameters[ii].endswith('$') else ''
                axes[ii, ii].set_title(
                    label + ' = {:.3g}^{{+{:.2g}}}_{{-{:.2g}}}'.format(qmed, err_hi, err_lo) + suffix,
                    fontsize=12,
                    loc='left'
                )

            if saveto is not None:
                fig.savefig(saveto + 'corners.png', dpi=200)
            else:
                fig.savefig(params['out_dir'] + 'corners.png', dpi=200)
            print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
            plt.close(fig)

            if not params['load_data']:
                print("ChainConsumer is not installed. Skipping walker plot generation.")
            return

        print("ChainConsumer is not installed. Falling back to a basic matplotlib corner plot.")
        fig, axes = plt.subplots(npar, npar, figsize=(4*npar, 4*npar))
        axes = np.atleast_2d(axes)
        
        for ii in range(npar):
            for jj in range(npar):
                ax = axes[ii, jj]
                
                if jj > ii:
                    ax.axis('off')
                    continue
                
                if ii == jj:
                    ax.hist(post[:, ii], bins=40, color='slateblue', alpha=0.7)
                    if knowTrue and all_parameters[ii] in truevals:
                        ax.axvline(truevals[all_parameters[ii]], color='g', ls='--', alpha=0.7)
                    qlo, qmed, qhi = simple_interval(post[:, ii])
                    err_hi = qhi - qmed
                    err_lo = qmed - qlo
                    ax.set_title(all_parameters[ii][:-1] + ' = {:.3g}^{{+{:.2g}}}_{{-{:.2g}}}$'.format(qmed, err_hi, err_lo),
                                 fontsize=12, loc='left')
                else:
                    ax.scatter(post[:, jj], post[:, ii], s=2, alpha=0.08, color='slateblue', rasterized=True)
                    if knowTrue:
                        if all_parameters[jj] in truevals:
                            ax.axvline(truevals[all_parameters[jj]], color='g', ls='--', alpha=0.4)
                        if all_parameters[ii] in truevals:
                            ax.axhline(truevals[all_parameters[ii]], color='g', ls='--', alpha=0.4)
                
                if ii == npar - 1:
                    ax.set_xlabel(all_parameters[jj], fontsize=12)
                else:
                    ax.set_xticklabels([])
                
                if jj == 0 and ii > 0:
                    ax.set_ylabel(all_parameters[ii], fontsize=12)
                elif jj > 0:
                    ax.set_yticklabels([])
        
        plt.tight_layout()
        if saveto is not None:
            plt.savefig(saveto + 'corners.png', dpi=200)
        else:
            plt.savefig(params['out_dir'] + 'corners.png', dpi=200)
        print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
        plt.close()
        
        if not params['load_data']:
            print("ChainConsumer is not installed. Skipping walker plot generation.")
        return
        
    ## Make chainconsumer corner plots
    cc = ChainConsumer()
    cc.add_chain(build_chainconsumer_chain(post, all_parameters))
    cc.set_plot_config(PlotConfig(max_ticks=2, label_font_size=18, tick_font_size=18,
                                  spacing=2, summarise=False, dpi=200))
    if knowTrue:
        cc.add_truth(Truth(location=truevals, color='g', line_style='--', alpha=0.7))

    fig = cc.plotter.plot(figsize=(16, 16))

    ## make axis labels to be parameter summaries
    sum_data = cc.analysis.get_summary()['Posterior']
    axes = np.array(fig.axes).reshape((npar, npar))

    # Adjust axis labels
    for ii in range(npar):
        ax = axes[ii, ii]

        # get the right summary for the parameter ii
        label = format_bound_title(all_parameters[ii], sum_data[all_parameters[ii]])
        ax.set_title(label, {'fontsize':18}, loc='left')


    ## Save posterior
    if saveto is not None:
        fig.savefig(saveto + 'corners.png', dpi=200)
    else:
        fig.savefig(params['out_dir'] + 'corners.png', dpi=200)
    print("Posteriors plots printed in " + params['out_dir'] + "corners.png")
    plt.close(fig)
    
    if not params['load_data']:    
        # plot walkers
        fig = cc.plotter.plot_walks(convolve=10)
        fig.savefig(params['out_dir'] + 'plotwalks.png', dpi=200)
        plt.close(fig)


if __name__ == '__main__':

    # Create parser
    parser = argparse.ArgumentParser(prog='plotmaker', usage='%(prog)s [options] rundir', description='run plotmaker')

    # Add arguments
    parser.add_argument('rundir', metavar='rundir', type=str, help='The path to the run directory')

    parser.add_argument('--nofit', action='store_true', help="Disable spectral fit reconstruction plots.")
    parser.add_argument('--nomap', action='store_true', help="Disable skymaps.")
    parser.add_argument('--nocorner', action='store_true', help="Disable corner plots.")
    
    # execute parser
    args = parser.parse_args()

    with open(args.rundir + '/config.pickle', 'rb') as paramfile:
        ## things are loaded from the pickle file in the same order they are put in
        params = pickle.load(paramfile)
        inj = pickle.load(paramfile)
        parameters = pickle.load(paramfile)
    
    ## grab the model and injection
    with open(args.rundir + '/model.pickle', 'rb') as modelfile:
        Model = pickle.load(modelfile)
    if not params['load_data']:
        with open(args.rundir + '/injection.pickle', 'rb') as injectionfile:
            Injection = pickle.load(injectionfile)
    else:
        Injection = None
    
    
    post = np.loadtxt(params['out_dir'] + "/post_samples.txt")
    
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    
    if not args.nocorner:
        plotmaker(post, params, parameters, inj, Model, Injection)    
    if not args.nofit:
        fitmaker(post, params, parameters, inj, Model, Injection)
    if not args.nomap:
        if 'healpy_proj' in params.keys():
            mapmaker(post, params, parameters, Model, coord=params['healpy_proj'])
        else:
            mapmaker(post, params, parameters, Model)
