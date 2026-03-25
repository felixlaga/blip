import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import healpy as hp
import logging
from blip.src.utils import log_manager, catch_duplicates, gen_suffixes, catch_color_duplicates
from blip.src.geometry import geometry
from blip.src.sph_geometry import sph_geometry
from blip.src.clebschGordan import clebschGordan
from blip.src.instrNoise import instrNoise
try:
    from blip.src.astro import Population
    import blip.src.astro as astro
except ImportError:
    Population = None
    astro = None



class submodel(geometry,sph_geometry,clebschGordan,instrNoise):
    '''
    Modular class that can represent either an injection or an analysis model. Will have different attributes depending on use case.
    
    Includes all information required to generate an injection or a likelihood/prior.
    
    New models (injection or analysis) should be added here.
    
    '''
    def __init__(self,params,inj,submodel_name,fs,f0,tsegmid,injection=False,suffix=''):
        '''
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "powerlaw_isgwb" defines a submodel with an isotropic spatial distribution and a power law spectrum.
        
        Resulting objects has different attributes depending on if it is to be used as an Injection component or part of our unified multi-signal Model.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        submodel_name (str) : submodel name, defined as "[spectral]_[spatial]" (or just "noise")
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        injection (bool)    : If True, generate the submodel as an injection component, rather than a Model submodel.
        suffix (str)        : String to append to parameter names, etc., to differentiate between duplicate submodels.
        
        Returns
        ------------
        submodel (object) : submodel with all needed attributes to serve as an Injection component or Model submodel as desired.
        
        '''
        
        ## preliminaries
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters
        self.fs = fs
        self.f0= f0
        self.tsegmid = tsegmid
        self.time_dim = tsegmid.size
        self.name = submodel_name
        self.injection = injection
        self._prior_debug_seen = set()
        geometry.__init__(self)
        
        ## remove the duplicate identifier if needed (powerlaw_isgwb-3 -> powerlaw_isgwb)
        submodel_full_name = submodel_name
        submodel_split = submodel_full_name.split('-')
        submodel_name = submodel_split[0]
        if len(submodel_split) == 1:
            submodel_count = ''
        elif len(submodel_split) == 2:
            submodel_count = ' ({})'.format(submodel_split[1])
        else:
            raise ValueError("'{}' is not a valid submodel/component specfication.".format(submodel_full_name))
        
        if submodel_full_name in params['alias'].keys():
            self.alias = params['alias'][submodel_full_name]
        
        if injection:
            self.truevals = {}
            ## for ease of use, assign the trueval dict to a variable
            if submodel_full_name in self.inj['truevals'].keys():
                self.injvals = self.inj['truevals'][submodel_full_name]
        
        ## plot kwargs dict to allow for case-by-case exceptions to our usual plotting approach
        ## e.g., the population spectra look real weird as dotted lines.
        self.plot_kwargs = {}
            
        ## handle & return noise case in bespoke fashion, as it is quite different from the signal models
        if submodel_name == 'noise':
            self.spectral_parameters = [r'$\log_{10} (Np)$'+suffix, r'$\log_{10} (Na)$'+suffix]
            self.spatial_parameters = []
            self.parameters = self.spectral_parameters
            self.Npar = 2
            ## for plotting
            self.fancyname = "Instrumental Noise"
            self.color = 'dimgrey'
            self.has_map = False
            # Figure out which instrumental noise spectra to use
            if self.params['tdi_lev']=='aet':
                self.instr_noise_spectrum = self.aet_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_aet_noise
            elif self.params['tdi_lev']=='xyz':
                self.instr_noise_spectrum = self.xyz_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_xyz_noise
            elif self.params['tdi_lev']=='michelson':
                self.instr_noise_spectrum = self.mich_noise_spectrum
                if injection:
                    self.gen_noise_spectrum = self.gen_michelson_noise
            else:
                raise ValueError("Unknown specification of 'tdi_lev'; can be 'michelson', 'xyz', or 'aet'.")
            if not injection:
                ## prior transform
                self.prior = self.instr_noise_prior
                ## covariance calculation
                self.cov = self.compute_cov_noise
            else:
                ## truevals
                self.truevals[r'$\log_{10} (Np)$'] = self.injvals['log_Np']
                self.truevals[r'$\log_{10} (Na)$'] = self.injvals['log_Na']
                ## save the frozen noise spectra
                self.frozen_spectra = self.instr_noise_spectrum(self.fs,self.f0,Np=10**self.injvals['log_Np'],Na=10**self.injvals['log_Na'])
            
            return

        else:
            self.parameters = []
            self.spectral_parameters = []
            self.spatial_parameters = []
            ## for convenience, so there's no need to specify e.g., "population_population"
            if submodel_name == 'population':
                self.spectral_model_name = self.spatial_model_name = submodel_name
            else:
                self.spectral_model_name, self.spatial_model_name = submodel_name.split('_')
            
            
        
        ###################################################
        ###            BUILD NEW MODELS HERE            ###
        ###################################################

        ## assignment of spectrum
        if self.spectral_model_name == 'powerlaw':
            if self.spatial_model_name == 'multipole':
                amplitude_parameter = self.get_single_multipole_amplitude_parameter()
                self.powerlaw_amplitude_prior_key = 'log_A_L'
                self.powerlaw_amplitude_trueval_key = 'log_A_L'
            elif self.spatial_model_name == 'fixedLchannels':
                amplitude_parameter = r'$\log_{10} (A_c)$'
                self.powerlaw_amplitude_prior_key = 'log10_Ac'
                self.powerlaw_amplitude_trueval_key = 'log10_Ac'
            else:
                amplitude_parameter = r'$\log_{10} (\Omega_0)$'
                self.powerlaw_amplitude_prior_key = 'log_omega0'
                self.powerlaw_amplitude_trueval_key = 'log_omega0'
            if self.spatial_model_name == 'fixedLchannels':
                self.spectral_parameters = self.spectral_parameters + [amplitude_parameter, r'$\alpha$']
                self.omegaf = self.fixedL_powerlaw_signal
            else:
                self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', amplitude_parameter]
                self.omegaf = self.powerlaw_spectrum
            self.fancyname = "Power Law"+submodel_count
            if not injection:
                if self.spatial_model_name == 'fixedLchannels':
                    self.spectral_prior = self.fixedL_powerlaw_prior
                else:
                    self.spectral_prior = self.powerlaw_prior
            else:
                self.truevals[amplitude_parameter] = self.injvals[self.powerlaw_amplitude_trueval_key]
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
        elif self.spectral_model_name == 'brokenpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha_1$',r'$\log_{10} (\Omega_0)$',r'$\alpha_2$',r'$\log_{10} (f_{break})$']
            self.omegaf = self.broken_powerlaw_spectrum
            self.fancyname = "Broken Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.broken_powerlaw_prior
            else:
                self.truevals[r'$\alpha_1$'] = self.injvals['alpha1']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\alpha_2$'] = self.injvals['alpha2']
                self.truevals[r'$\log_{10} (f_{break})$'] = self.injvals['log_fbreak']
        
        elif self.spectral_model_name == 'truncatedpowerlaw':
            self.spectral_parameters = self.spectral_parameters + [r'$\alpha$', r'$\log_{10} (\Omega_0)$', r'$\log_{10} (f_{\mathrm{cut}})$',r'$\log_{10} (f_{\mathrm{scale}})$']
            self.omegaf = self.truncated_powerlaw_spectrum
            self.fancyname = "Truncated Power Law"+submodel_count
            if not injection:
                self.spectral_prior = self.truncated_powerlaw_prior
            else:
                self.truevals[r'$\alpha$'] = self.injvals['alpha']
                self.truevals[r'$\log_{10} (\Omega_0)$'] = self.injvals['log_omega0']
                self.truevals[r'$\log_{10} (f_{\mathrm{cut}})$'] = self.injvals['log_fcut']
                self.truevals[r'$\log_{10} (f_{\mathrm{scale}})$'] = self.injvals['log_fscale']
        elif self.spectral_model_name == 'population':
            if not injection:
                raise ValueError("Populations are injection-only.")
            if Population is None:
                raise ImportError("Population injections require the optional astrophysical dependencies (including legwork).")
            self.fancyname = "DWD Population"+submodel_count
            self.population = Population(self.params,self.inj,self.fs)
            self.compute_Sgw = self.population.Sgw_wrapper
            self.omegaf = self.population.omegaf_wrapper
            self.ispop = True
            self.plot_kwargs |= {'ls':'-','lw':0.75,'alpha':0.6}
        
        else:
            ValueError("Unsupported spectrum type. Check your spelling or add a new spectrum model!")
        
        ## assignment of response and spatial methods
        response_kwargs = {}
        
        ## This is the isotropic spatial model, and has no additional parameters.
        if self.spatial_model_name == 'isgwb':
            if self.params['tdi_lev'] == 'michelson':
                self.response = self.isgwb_mich_response
            elif self.params['tdi_lev'] == 'xyz':
                self.response = self.isgwb_xyz_response
            elif self.params['tdi_lev'] == 'aet':
                self.response = self.isgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## compute response matrix
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## plotting stuff
            self.fancyname = "Isotropic "+self.fancyname
            self.subscript = "_{\mathrm{I}}"
            self.color='darkorange'
            self.has_map = False

            if not injection:
                ## prior transform
                self.prior = self.isotropic_prior
                self.cov = self.compute_cov_isgwb
            else:
                ## create a wrapper b/c isotropic and anisotropic injection responses are different
                self.inj_response_mat = self.response_mat
        
        ## This is the spherical harmonic spatial model. It is the workhorse of the spherical harmonic anisotropic analysis.
        ## It can also be used to perform arbitrary injections in the spherical harmonic basis via direct specification of the blms.
        elif self.spatial_model_name == 'sph':
            
            if injection:
                self.lmax = self.inj['inj_lmax']
            else:
                self.lmax = self.params['lmax']
            
            ## almax is twice the blmax
            self.almax = 2*self.lmax
            response_kwargs['set_almax'] = self.almax
            
            if self.params['tdi_lev']=='michelson':
                self.response = self.asgwb_mich_response
            elif self.params['tdi_lev']=='xyz':
                self.response = self.asgwb_xyz_response
            elif self.params['tdi_lev']=='aet':
                self.response = self.asgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## compute response matrix
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## plotting stuff
            self.fancyname = "Anisotropic "+self.fancyname
            self.subscript = "_{\mathrm{A}}"
            self.color = 'teal'
            self.has_map = True
            
            # add the blms
            blm_parameters = gen_blm_parameters(self.lmax)
            
            ## save the blm start index for the prior, then add the blms to the parameter list
            self.blm_start = len(self.spectral_parameters)
            self.spatial_parameters = self.spatial_parameters + blm_parameters
            
            if not injection:
                self.prior = self.sph_prior
                self.cov = self.compute_cov_asgwb
            else:
                ## get blm truevals
                val_list = self.blms_2_blm_params(inj['blms'])
                
                for param, val in zip(blm_parameters,val_list):
                    self.truevals[param] = val
                
                ## get alms
                self.alms_inj = self.compute_skymap_alms(inj['blms'])
                ## get sph basis skymap
                self.sph_skymap =  hp.alm2map(self.alms_inj[0:hp.Alm.getsize(self.almax)],self.params['nside'])
                ## get response integrated over the Ylms
                self.summ_response_mat = self.compute_summed_response(self.alms_inj)
                ## create a wrapper b/c isotropic and anisotropic injection responses are different
                self.inj_response_mat = self.summ_response_mat
        
        ## This mode keeps one chosen multipole L and marginalizes over its m-modes internally.
        ## It infers only Np, Na, alpha, and log10(A_L), with no free b_lm coefficients and no sky reconstruction.
        elif self.spatial_model_name == 'multipole':
            self.multipole_l = self.get_selected_multipole_l()
            response_kwargs['set_almax'] = self.multipole_l
            
            if self.params['tdi_lev']=='michelson':
                self.response = self.asgwb_mich_response
            elif self.params['tdi_lev']=='xyz':
                self.response = self.asgwb_xyz_response
            elif self.params['tdi_lev']=='aet':
                self.response = self.asgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## build one effective fixed-L response by summing over m in quadrature
            multipole_response_basis = self.response(f0,tsegmid,**response_kwargs)
            self.response_mat = self.compute_single_multipole_response(multipole_response_basis,self.multipole_l)
            
            ## plotting stuff
            self.fancyname = "Single Multipole $L={}$ ".format(self.multipole_l) + self.fancyname
            self.subscript = "_{\mathrm{L=" + str(self.multipole_l) + "}}"
            self.color = 'royalblue'
            self.has_map = False
            
            if not injection:
                self.prior = self.isotropic_prior
                self.cov = self.compute_cov_isgwb
            else:
                self.inj_response_mat = self.response_mat

        elif self.spatial_model_name == 'fixedLchannels':
            if self.params['tdi_lev'] != 'aet':
                raise ValueError("The fixed-L channel likelihood currently supports only tdi_lev='aet'.")

            self.fixedL = self.get_selected_fixedL()
            response_kwargs['set_almax'] = self.fixedL
            self.response = self.asgwb_aet_response
            self.response_basis_mat = self.response(f0, tsegmid, **response_kwargs)
            self.fixedL_channel_pairs = self.get_fixedL_channel_pairs()
            self.fixedL_channel_response_segments = self.compute_fixedL_channel_response_segments(
                self.response_basis_mat,
                self.fixedL
            )
            self.fixedL_channel_response_matrix = self.assemble_fixedL_channel_response_matrix(
                self.fixedL_channel_response_segments
            )
            self.fixedL_channel_response = {
                pair: np.mean(response_segments, axis=1)
                for pair, response_segments in self.fixedL_channel_response_segments.items()
            }

            self.fancyname = "Fixed-L Channels $L={}$ ".format(self.fixedL) + self.fancyname
            self.subscript = "_{\mathrm{fixedL=" + str(self.fixedL) + "}}"
            self.color = 'royalblue'
            self.has_map = False

            if not injection:
                self.prior = self.isotropic_prior
                self.compute_Sgw = self.fixedL_analysis_guard_compute_Sgw
                self.cov = self.fixedL_analysis_guard_covariance
            else:
                ## Use the same direct per-pair fixed-L response object as the channel likelihood.
                ## This keeps the injection conventions aligned with the analysis mode.
                self.inj_response_mat = self.fixedL_channel_response_matrix
                self.compute_injected_sgw = self.compute_fixedL_injected_sgw

        ## Handle all the astrophysical spatial distributions together due to their similarities
        elif self.spatial_model_name in ['galaxy','dwarfgalaxy','lmc','pointsource','twopoints','population']:
            
            ## the astrophysical spatial models are generally injection-only
            if not injection:
                raise ValueError("This model is injection-only.")
            if astro is None:
                raise ImportError("Astrophysical sky injections require the optional astrophysical dependencies (including legwork).")
            
            self.has_map = True
            
            ## almax is twice the blmax
            self.lmax = self.inj['inj_lmax']
            self.almax = 2*self.lmax
            response_kwargs['set_almax'] = self.almax
            
            if self.params['tdi_lev']=='michelson':
                self.response = self.asgwb_mich_response
            elif self.params['tdi_lev']=='xyz':
                self.response = self.asgwb_xyz_response
            elif self.params['tdi_lev']=='aet':
                self.response = self.asgwb_aet_response
            else:
                raise ValueError("Invalid specification of tdi_lev. Can be 'michelson', 'xyz', or 'aet'.")
            
            ## compute response matrix
            self.response_mat = self.response(f0,tsegmid,**response_kwargs)
            
            ## model-specific quantities
            if self.spatial_model_name == 'galaxy':
                ## store the high-level MW truevals for the hierarchical analysis
                self.truevals[r'$r_{\mathrm{h}}$'] = inj['rh']
                self.truevals[r'$z_{\mathrm{h}}$'] = inj['zh']
                ## plotting stuff
                self.fancyname = "Galactic Foreground"
                self.subscript = "_{\mathrm{G}}"
                self.color = 'mediumorchid'
                ## generate skymap
                self.skymap = astro.generate_galactic_foreground(self.injvals['rh'],self.injvals['zh'],self.params['nside'])
            elif self.spatial_model_name == 'lmc':
                ## plotting stuff
                self.fancyname = "LMC"
                self.subscript = "_{\mathrm{LMC}}"
                self.color = 'darkmagenta'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside']) ## sdg defaults are for the LMC
            elif self.spatial_model_name == 'dwarfgalaxy':
                ## plotting stuff
                self.fancyname = "Dwarf Galaxy"+submodel_count
                self.subscript = "_{\mathrm{DG}}"
                self.color = 'maroon'
                ## generate skymap
                self.skymap = astro.generate_sdg(self.params['nside'],ra=self.injvals['sdg_RA'], dec=self.injvals['sdg_DEC'], D=self.injvals['sdg_dist'], r=self.injvals['sdg_rad'], N=self.injvals['sdg_N'])
            elif self.spatial_model_name == 'pointsource':
                ## plotting stuff
                self.fancyname = "Point Source"+submodel_count
                self.subscript = "_{\mathrm{1P}}"
                self.color = 'forestgreen'
                ## generate skymap
                self.skymap = astro.generate_point_source(self.injvals['theta'],self.injvals['phi'],self.params['nside'])
            elif self.spatial_model_name == 'twopoints':
                ## revisit this when I have duplicates sorted, maybe unnecessary (could just have 2x point source injection components)
                ## plotting stuff
                self.fancyname = "Two Point Sources"+submodel_count
                self.subscript = "_{\mathrm{2P}}"
                self.color = 'gold'
                ## generate skymap
                self.skymap = astro.generate_two_point_source(self.injvals['theta_1'],self.injvals['phi_1'],self.injvals['theta_2'],self.injvals['phi_2'],self.params['nside'])
            elif self.spatial_model_name == 'population':
                ## flag the fact that we have a population skymap
                self.skypop = True
                ## plotting stuff
                self.fancyname = "DWD Population"+submodel_count
                self.subscript = "_{\mathrm{P}}"
                self.color = 'midnightblue'
                if self.spectral_model_name != 'population':
                    ## generate population if still needed
                    self.population = Population(self.params,self.inj,self.fs)
                self.skymap = self.population.skymap
            
            else:
                raise ValueError("Astrophysical submodel type not found. Did you add a new model to the list at the top of this section?")
            
            self.process_astro_skymap(self.skymap)
            

        elif self.spatial_model_name == 'hierarchical':
            pass
        else:
            raise ValueError("Invalid specification of spatial model name ('{}'). Can be 'isgwb', 'sph', 'galaxy', or 'hierarchical'.".format(self.spatial_model_name))
        
        
        ## store final parameter list and count
        self.parameters = self.parameters + self.spectral_parameters + self.spatial_parameters
        if not injection:               
            self.Npar = len(self.parameters)
        ## store response kwargs for use elsewhere as needed
        self.response_kwargs = response_kwargs
        ## add suffix to parameter names and trueval keys, if desired
        ## (we need this in the multi-model or duplicate model case)
        if suffix != '':
            if injection:
                updated_truevals = {parameter+suffix:self.truevals[parameter] for parameter in self.parameters}
                self.truevals = updated_truevals
            updated_spectral_parameters = [parameter+suffix for parameter in self.spectral_parameters]
            updated_spatial_parameters = [parameter+suffix for parameter in self.spatial_parameters]
            updated_parameters = updated_spectral_parameters+updated_spatial_parameters
            if len(updated_parameters) != len(self.parameters):
                raise ValueError("If you've added a new variety of parameters above, you'll need to update this bit of code too!")
            self.spectral_parameters = updated_spectral_parameters
            self.spatial_parameters = updated_spatial_parameters
            self.parameters = updated_parameters
            
        
        return
    

    #############################
    ##    Spectral Functions   ##
    #############################
    def powerlaw_spectrum(self,fs,alpha,log_omega0):
        '''
        Function to calculate a simple power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha (float)        : slope of the power law
        log_omega0 (float)   : power law amplitude in units of log dimensionless GW energy density at f_ref
        
        Returns
        -----------
        spectrum (array of floats) : the resulting power law spectrum
        
        '''
        return 10**(log_omega0)*(fs/self.params['fref'])**alpha
    
    
    def broken_powerlaw_spectrum(self,fs,alpha_1,log_omega0,alpha_2,log_fbreak):
        '''
        Function to calculate a broken power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha_1 (float)      : slope of the first power law
        log_omega0 (float)   : power law amplitude of the first power law in units of log dimensionless GW energy density at f_ref
        alpha_2 (float)      : slope of the second power law
        log_fbreak (float)   : log of the break frequency ("knee") in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting broken power law spectrum
        
        '''
        delta = 0.1
        fbreak = 10**log_fbreak
        norm = (fbreak/self.params['fref'])**alpha_1 / 1.25989194 ## this normalizes the broken powerlaw such that its first leg matches the equivalent standard power law
        return norm * (10**log_omega0)*(fs/fbreak)**(alpha_1) * (0.5*(1+(fs/fbreak)**(1/delta)))**((alpha_1-alpha_2)*delta)
    
    def truncated_powerlaw_spectrum(self,fs,alpha,log_omega0,log_fcut,log_fscale):
        '''
        Function to calculate a tanh-truncated power law spectrum.
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        alpha (float)        : slope of the power law
        log_omega0 (float)   : power law amplitude of the power law in units of log dimensionless GW energy density at f_ref (if left un-truncated)
        log_fcut (float)     : log of the cut frequency ("knee") in Hz
        log_fscale           : log of the cutoff scale factor in Hz
        
        Returns
        -----------
        spectrum (array of floats) : the resulting truncated power law spectrum
        
        '''
        fcut = 10**log_fcut
        fscale = 10**log_fscale
        return 0.5 * (10**log_omega0)*(fs/self.params['fref'])**(alpha) * (1+np.tanh((fcut-fs)/fscale))
    
    def compute_Sgw(self,fs,omegaf_args):
        '''
        Wrapper function to generically calculate the associated stochastic gravitational wave PSD (S_gw)
            for a spectral model given in terms of the dimensionless GW energy density Omega(f)
        
        Arguments
        -----------
        fs (array of floats) : frequencies at which to evaluate the spectrum
        omegaf_args (list)   : list of arguments for the relevant Omega(f) function
        
        Returns
        -----------
        Sgw (array of floats) : the resulting GW PSD
        
        '''
        H0 = 2.2*10**(-18)
        Omegaf = self.omegaf(fs,*omegaf_args)
        Sgw = Omegaf*(3/(4*fs**3))*(H0/np.pi)**2
        return Sgw

    def omega_to_sgw_factor(self, fs):
        '''
        Conversion factor from Omega(f) h^2 units to SGWB PSD units, using BLIP's H0 convention.
        '''
        H0 = 2.2*10**(-18)
        return (3/(4*fs**3))*(H0/np.pi)**2

    def sgw_to_omega_factor(self, fs):
        '''
        Conversion factor from SGWB PSD units back to Omega(f) h^2 units.
        '''
        return 1.0 / self.omega_to_sgw_factor(fs)

    def fixedL_powerlaw_signal(self, fs, log10_Ac, alpha, fc=None):
        '''
        Fixed-L SGWB power-law model in Omega_GW^L(f) h^2 units.

        The fixed multipole is selected in configuration via fixedL, so this helper returns only the
        spectral part, Omega_GW^L(f) h^2 = 10^(log10_Ac) * (f / f_c)^alpha.
        '''
        if fc is None:
            fc = self.params.get('fixedL_fc', 2.5e-3)
        return 10**(log10_Ac) * (fs / fc)**alpha

    def d_fixedL_d_log10_Ac(self, fs, log10_Ac, alpha, fc=None):
        '''
        Diagnostic derivative of the fixed-L signal with respect to log10_Ac.
        '''
        signal = self.fixedL_powerlaw_signal(fs, log10_Ac, alpha, fc=fc)
        return np.log(10.0) * signal

    def d_fixedL_d_alpha(self, fs, log10_Ac, alpha, fc=None):
        '''
        Diagnostic derivative of the fixed-L signal with respect to alpha.
        '''
        if fc is None:
            fc = self.params.get('fixedL_fc', 2.5e-3)
        signal = self.fixedL_powerlaw_signal(fs, log10_Ac, alpha, fc=fc)
        return np.log(fs / fc) * signal

    def compute_fixedL_injected_sgw(self, fs, signal_args):
        '''
        Injection-only helper returning SGWB PSD units for the fixed-L power-law signal.
        '''
        log10_Ac, alpha = signal_args
        omega_signal = self.fixedL_powerlaw_signal(fs, log10_Ac, alpha)
        return omega_signal * self.omega_to_sgw_factor(fs)

    def fixedL_analysis_guard_compute_Sgw(self, *args, **kwargs):
        '''
        Guard against routing the fixed-L channel likelihood through the legacy compute_Sgw path.
        '''
        raise RuntimeError("The fixed-L channel likelihood must not call compute_Sgw(...).")

    def fixedL_analysis_guard_covariance(self, *args, **kwargs):
        '''
        Guard against routing the fixed-L channel likelihood through covariance assembly.
        '''
        raise RuntimeError("The fixed-L channel likelihood must not assemble a signal covariance matrix.")
    
    def get_prior_bounds(self,prior_key,default_bounds):
        '''
        Return the configured [min, max] bounds for a named prior, falling back to legacy defaults.
        '''
        has_prior_bounds = 'prior_bounds' in self.params
        prior_bounds = self.params.get('prior_bounds',{})
        bounds = prior_bounds.get(prior_key,default_bounds)
        bounds = np.asarray(bounds,dtype=float)
        if bounds.shape != (2,):
            raise ValueError("Prior '{}' must have exactly two bounds.".format(prior_key))
        bounds.sort()
        if self.params.get('debug_priors',0) and prior_key not in self._prior_debug_seen:
            print("[BLIP prior lookup] key={} default={} has_prior_bounds={} available_keys={} chosen={}".format(
                prior_key,
                list(default_bounds),
                has_prior_bounds,
                sorted(list(prior_bounds.keys())) if isinstance(prior_bounds,dict) else type(prior_bounds).__name__,
                bounds.tolist()
            ))
            self._prior_debug_seen.add(prior_key)
        return bounds
    
    def rescale_uniform_prior(self,unit_theta,prior_key,default_bounds):
        '''
        Rescale unit-cube draws onto a configured uniform prior interval.
        '''
        lower, upper = self.get_prior_bounds(prior_key,default_bounds)
        return lower + (upper-lower)*np.asarray(unit_theta)
    
    def get_selected_multipole_l(self):
        '''
        Return the fixed multipole L for the single-multipole total-power mode.
        '''
        if self.injection:
            multipole_l = self.inj.get('multipole_l', self.params.get('multipole_l', -1))
        else:
            multipole_l = self.params.get('multipole_l', -1)
        
        multipole_l = int(multipole_l)
        if multipole_l < 1:
            raise ValueError("The single-multipole total-power mode requires 'multipole_l' >= 1 in the ini file.")
        
        return multipole_l

    def get_selected_fixedL(self):
        '''
        Return the fixed multipole L for the channel-resolved fixed-L mode.
        '''
        if self.injection:
            fixedL = self.inj.get('fixedL', self.params.get('fixedL', -1))
        else:
            fixedL = self.params.get('fixedL', -1)

        fixedL = int(fixedL)
        if fixedL < 1:
            raise ValueError("The fixed-L channel mode requires 'fixedL' >= 1 in the ini file.")

        return fixedL
    
    def get_single_multipole_amplitude_parameter(self):
        '''
        Parameter label for the single-multipole total-power amplitude.
        '''
        return r'$\log_{10} (A_{' + str(self.get_selected_multipole_l()) + '})$'
    
    def get_single_multipole_indices(self,multipole_l):
        '''
        Return the response-basis indices corresponding to the chosen multipole L and all of its m-modes.
        '''
        return [ii for ii in range((multipole_l + 1)**2) if self.idxtoalm(multipole_l, ii)[0] == multipole_l]

    def get_fixedL_channel_pairs(self):
        '''
        Return the unique unordered AET channel pairs used in the fixed-L likelihood.
        '''
        return {
            'AA': (0, 0),
            'EE': (1, 1),
            'TT': (2, 2),
            'AE': (0, 1),
            'AT': (0, 2),
            'ET': (1, 2),
        }

    def compute_fixedL_channel_response_segments(self, response_basis_mat, fixedL):
        '''
        Build the per-segment fixed-L response for each unique channel pair from BLIP's direct pair
        anisotropic response coefficients R_{OO',lm}(f,t).

        This is the channel-resolved response used by the new fixed-L likelihood, not the legacy
        covariance-template kernel. We compress the selected 2L+1 m-modes within each channel pair
        into a real power-like response via mean_m |R_{OO',Lm}(f,t)|^2.
        '''
        fixedL_indices = self.get_single_multipole_indices(fixedL)
        response_segments = {}
        for pair, (ii, jj) in self.get_fixedL_channel_pairs().items():
            pair_coefficients = np.take(response_basis_mat[ii, jj], fixedL_indices, axis=-1)
            response_segments[pair] = np.mean(np.abs(pair_coefficients)**2, axis=-1)

        return response_segments

    def assemble_fixedL_channel_response_matrix(self, response_segments):
        '''
        Assemble the Hermitian 3x3 per-segment fixed-L response matrix from the unique unordered
        AET channel pairs used by the channel-resolved likelihood.

        The matrix elements are built directly from the per-pair fixed-L responses rather than
        from the legacy hidden-index contraction used by the covariance-template mode.
        '''
        response_matrix = np.zeros((3, 3, self.fs.size, self.tsegmid.size), dtype='complex')
        for pair, (ii, jj) in self.get_fixedL_channel_pairs().items():
            response_matrix[ii, jj] = response_segments[pair]
            if ii != jj:
                response_matrix[jj, ii] = np.conj(response_segments[pair])

        return response_matrix

    def compute_fixedL_stochastic_power_response_matrix(self, response_basis_mat, fixedL):
        '''
        Legacy hidden-index fixed-L response contraction retained only for backwards comparison.

        The new fixed-L channel likelihood must not use this helper. It contracts over the hidden
        detector index to build a covariance-template kernel, which is the behavior replaced by the
        direct per-pair fixed-L response matrix assembled in assemble_fixedL_channel_response_matrix.
        '''
        fixedL_indices = self.get_single_multipole_indices(fixedL)
        multipole_response = np.take(response_basis_mat, fixedL_indices, axis=-1)
        power_response = np.einsum('ikftm,jkftm->ijft', multipole_response, np.conj(multipole_response))
        power_response = power_response / len(fixedL_indices)
        power_response = 0.5 * (power_response + np.swapaxes(np.conj(power_response), 0, 1))
        return power_response

    def compute_fixedL_isotropic_normalization(self, fs=None, f0=None, tsegmid=None):
        '''
        Cross-check that the l=0,m=0 anisotropic response reproduces the isotropic response up to sqrt(4*pi).
        '''
        if self.params['tdi_lev'] != 'aet':
            raise ValueError("The fixed-L isotropic-normalization cross-check is only implemented for AET.")

        if fs is None:
            fs = self.fs
        if f0 is None:
            f0 = self.f0
        if tsegmid is None:
            tsegmid = self.tsegmid

        anisotropic_l0 = self.asgwb_aet_response(f0, tsegmid, set_almax=0)
        isotropic = self.isgwb_aet_response(f0, tsegmid)
        ratio = anisotropic_l0[:, :, :, :, 0] / isotropic

        return {
            'expected_ratio': np.sqrt(4*np.pi),
            'ratio': ratio,
            'max_abs_deviation': np.nanmax(np.abs(ratio - np.sqrt(4*np.pi))),
        }
    
    def compute_single_multipole_response(self,response_basis_mat,multipole_l):
        '''
        Build a statistically isotropic covariance response for the total power in one fixed multipole L.
        
        This first-pass mode handles the m-modes analytically by assuming an isotropic power distribution.
        It computes the rotationally-invariant, effectively unpolarized Gram matrix sum_m R_m R_m^dagger 
        normalized by 2L+1. Because R_m is dimensionless, this returns a valid positive semi-definite 
        dimensionless response matrix ready to be scaled by the total power amplitude A_L.
        '''
        multipole_indices = self.get_single_multipole_indices(multipole_l)
        multipole_response = np.take(response_basis_mat, multipole_indices, axis=-1)
        effective_response_sq = np.einsum('ikftm,jkftm->ijft', multipole_response, np.conj(multipole_response))
        effective_response_sq = effective_response_sq / len(multipole_indices)
        
        ## Ensure Hermitian symmetry explicitly
        effective_response_sq = 0.5 * (effective_response_sq + np.swapaxes(np.conj(effective_response_sq), 0, 1))
        
        return effective_response_sq
    
    #############################
    ##          Priors         ##
    #############################
    def isotropic_prior(self,theta):
        '''
        Isotropic prior transform. Just serves as a wrapper for the spectral prior, as no additional foofaraw is necessary.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for the spectral parameters.
            
        '''
        return self.spectral_prior(theta)
    
    def sph_prior(self,theta):
        '''
        Spherical harmonic anisotropic prior transform. Combines a generic spectral prior function with the spherical harmonic priors for the desired lmax.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for both the spectral and spatial parameters.
        '''
        
        ## spectral prior takes everything up to 
        spectral_theta = self.spectral_prior(theta[:self.blm_start])
        
        # Calculate lmax from the size of theta blm arrays. The shape is
        # given by size = (lmax + 1)**2 - 1. The '-1' is because b00 is
        # an independent parameter
        lmax = np.sqrt( len(theta[self.blm_start:]) + 1 ) - 1

        if lmax.is_integer():
            lmax = int(lmax)
        else:
            raise ValueError('Illegitimate theta size passed to the spherical harmonic prior')

        # The rest of the priors define the blm parameter space
        sph_theta = []

        ## counter for the rest of theta
        cnt = self.blm_start

        for lval in range(1, lmax + 1):
            for mval in range(lval + 1):

                if mval == 0:
                    sph_theta.append(6*theta[cnt] - 3)
                    cnt = cnt + 1
                else:
                    ## prior on amplitude, phase
                    sph_theta.append(3* theta[cnt])
                    sph_theta.append(2*np.pi*theta[cnt+1] - np.pi)
                    cnt = cnt + 2

        return spectral_theta+sph_theta
    
    def hierarchical_prior(self,theta):
        '''
        Hierarchical anisotropic prior transform. Combines a generic spectral prior function with the hierarchical astrophysical prior.
        
        Arguments
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled for both the spectral and spatial parameters.
        '''
        pass
        
        
    def instr_noise_prior(self,theta):


        '''
        Prior function for only instrumental noise

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, omega_ref, Np and Na

        '''


        # Unpack: Theta is defined in the unit cube
        log_Np, log_Na = theta

        # Transform to actual priors
        log_Np = self.rescale_uniform_prior(log_Np,'log_Np',[-44,-39])
        log_Na = self.rescale_uniform_prior(log_Na,'log_Na',[-51,-46])

        return [log_Np, log_Na]
    
    def powerlaw_prior(self,theta):


        '''
        Prior function for an isotropic stochastic backgound analysis.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha and log(Omega0)

        '''


        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = self.rescale_uniform_prior(theta[0],'alpha',[-5,5])
        log_amplitude = self.rescale_uniform_prior(theta[1],getattr(self,'powerlaw_amplitude_prior_key','log_omega0'),[-14,8])
        
        return [alpha, log_amplitude]

    def fixedL_powerlaw_prior(self, theta):
        '''
        Prior transform for the fixed-L channel-resolved power-law signal.

        Returns parameters in the dedicated order [log10_Ac, alpha].
        '''
        log10_Ac = self.rescale_uniform_prior(theta[0], 'log10_Ac', [-14, -8])
        alpha = self.rescale_uniform_prior(theta[1], 'alpha', [-5, 5])
        return [log10_Ac, alpha]
    
    def broken_powerlaw_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a broken power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha_1, log(Omega_0), alpha_2, and log(f_break).

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha_1 = self.rescale_uniform_prior(theta[0],'alpha1',[-4,6])
        log_omega0 = self.rescale_uniform_prior(theta[1],'log_omega0',[-14,8])
        alpha_2 = self.rescale_uniform_prior(theta[2],'alpha2',[0,40])
        log_fbreak = self.rescale_uniform_prior(theta[3],'log_fbreak',[-4,-2])

        return [alpha_1, log_omega0, alpha_2, log_fbreak]
    
    def truncated_powerlaw_prior(self,theta):


        '''
        Prior function for a stochastic signal search with a truncated power law spectral model.

        Parameters
        -----------

        theta   : float
            A list or numpy array containing samples from a unit cube.

        Returns
        ---------

        theta   :   float
            theta with each element rescaled. The elements are  interpreted as alpha, log(Omega_0), log(f_cut), and log(f_scale)

        '''

        # Unpack: Theta is defined in the unit cube
        # Transform to actual priors
        alpha = self.rescale_uniform_prior(theta[0],'alpha',[-5,5])
        log_omega0 = self.rescale_uniform_prior(theta[1],'log_omega0',[-14,8])
        log_fcut = self.rescale_uniform_prior(theta[2],'log_fcut',[-4,-2])
        log_fscale = self.rescale_uniform_prior(theta[3],'log_fscale',[-4,-2])
        

        return [alpha, log_omega0, log_fcut, log_fscale]

    def compute_fixedL_channel_noise_omega(self, fs, f0, log_Np, log_Na):
        '''
        Channel-resolved AET instrumental noise written in Omega h^2 units for the fixed-L likelihood.

        Cross-channel AET noise terms are set to zero after confirming that BLIP's aet_noise_spectrum
        returns only zero or negligible floating-point roundoff for AE, AT, and ET in this basis.
        '''
        if self.params['tdi_lev'] != 'aet':
            raise ValueError("The fixed-L channel likelihood currently supports only tdi_lev='aet'.")

        Np = 10**log_Np
        Na = 10**log_Na
        noise_cov = self.aet_noise_spectrum(fs, f0, Np=Np, Na=Na)
        omega_factor = self.omega_to_sgw_factor(fs)
        noise_omega = {}
        for pair, (ii, jj) in self.get_fixedL_channel_pairs().items():
            if pair in ['AE', 'AT', 'ET']:
                noise_omega[pair] = np.zeros_like(fs)
            else:
                noise_omega[pair] = np.real(noise_cov[ii, jj, :]) / omega_factor

        return noise_omega

    
    #############################
    ## Covariance Calculations ##
    #############################
    def compute_cov_noise(self,theta):
        '''
        Computes the noise covariance for a given draw of log_Np, log_Na
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_noise (array) : The corresponding 3 x 3 x frequency x time covariance matrix for the detector noise submodel.
        
        '''
        ## unpack priors
        log_Np, log_Na = theta

        Np, Na =  10**(log_Np), 10**(log_Na)

        ## Modelled Noise PSD
        cov_noise = self.instr_noise_spectrum(self.fs,self.f0, Np, Na)

        ## repeat C_Noise to have the same time-dimension as everything else
        cov_noise = np.repeat(cov_noise[:, :, :, np.newaxis], self.time_dim, axis=3)
        
        return cov_noise
    
    def compute_cov_isgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a generic isotropic stochastic GW signal.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an isotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta)

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*self.response_mat
        
        return cov_sgwb
    
    def compute_cov_asgwb(self,theta):
        '''
        Computes the covariance matrix contribution from a generic anisotropic stochastic GW signal.
        
        Arguments
        ----------
        theta (float)   :  A list or numpy array containing samples from a unit cube.
        
        Returns
        ----------
        cov_sgwb (array) : The corresponding 3 x 3 x frequency x time covariance matrix for an anisotropic SGWB submodel.
        
        '''
        ## Signal PSD
        Sgw = self.compute_Sgw(self.fs,theta[:self.blm_start])
        
        ## get skymap and integrate over alms
        summ_response_mat = self.compute_summed_response(self.compute_skymap_alms(theta[self.blm_start:]))

        ## The noise spectrum of the GW signal. Written down here as a full
        ## covariance matrix axross all the channels.
        cov_sgwb = Sgw[None, None, :, None]*summ_response_mat
        
        return cov_sgwb

       
    ##########################################
    ##   Skymap and Response Calculations   ##
    ##########################################
    
    def compute_skymap_alms(self,blm_params):
        '''
        Function to compute the anisotropic skymap a_lms from the blm parameters.
        
        Arguments
        ----------
        blm_params (array of complex floats) : the blm parameters
        
        Returns
        ----------
        alm_vals (array of complex floats) : the corresponding alms
        
        '''
        ## Spatial distribution
        blm_vals = self.blm_params_2_blms(blm_params)
        alm_vals = self.blm_2_alm(blm_vals)

        ## normalize and return
        return alm_vals/(alm_vals[0] * np.sqrt(4*np.pi))
    
    def compute_summed_response(self,alms):
        '''
        Function to compute the integrated, skymap-convolved anisotropic response
        
        Arguments
        ----------
        alms (array of complex floats) : the spherical harmonic alms
        
        Returns
        ----------
        summ_response_mat (array) : the sky/alm-integrated response (3 x 3 x frequency x time)
        
        '''
        return np.einsum('ijklm,m', self.response_mat, alms)
    
    def process_astro_skymap(self,skymap):
        '''
        
        Function that takes in an astrophysical pixel skymap and:
            - calculates all associated sph quantities
            - computes corresponding blm parameter truevals
            - convolves with response
            
        Arguments
        -----------
        skymap (healpy array) : pixel-basis astrophysical skymap
        
        '''
        if astro is None:
            raise ImportError("Astrophysical sky injections require the optional astrophysical dependencies (including legwork).")
        ## transform to blms
        self.astro_blms = astro.skymap_pix2sph(skymap,self.lmax)
        ## get corresponding truevals
        inj_blms = self.blms_2_blm_params(self.astro_blms)
        blm_parameters = gen_blm_parameters(self.lmax)
        for param, val in zip(blm_parameters,inj_blms):
            self.truevals[param] = val
        
        self.alms_inj = self.blm_2_alm(self.astro_blms)
        self.alms_inj = self.alms_inj/(self.alms_inj[0] * np.sqrt(4*np.pi))
        self.sph_skymap = hp.alm2map(self.alms_inj[0:hp.Alm.getsize(self.almax)],self.params['nside'])
        ## get response integrated over the Ylms
        self.summ_response_mat = self.compute_summed_response(self.alms_inj)
        ## create a wrapper b/c isotropic and anisotropic injection responses are different
        self.inj_response_mat = self.summ_response_mat
        
        return
    
    
    def recompute_response(self,f0=None,tsegmid=None):
        '''
        Function to recompute the LISA response matrices if needed.
        
        When we save the Injection object, we delete the LISA response of each injection, as to do otherwise takes up egregious amounts of disk space.
        This allows us to recompute them identically as desired.
        
        Arguments
        -------------
        f0 (array)      : LISA-characteristic-frequency-scaled frequency array at which to compute the response (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints at which to compute the response
        
        Returns
        --------------
        response_mat (array) : The associated response for this submodel. 
        '''
        ## allow for respecification of frequency/time grid, but avoid needless computation of extant response matrices
        fsame = True
        tsame = True
        if f0 is not None:
            if f0.shape != self.f0.shape:
                fsame = False
            elif not np.all(f0==self.f0):
                fsame = False
        else:
            f0 = self.f0
        if tsegmid is not None:
            if tsegmid.shape != self.tsegmid.shape:
                tsame = False
            elif not np.all(tsegmid==self.tsegmid):
                tsame = False
        else:
            tsegmid = self.tsegmid
        
        tf_same = tsame and fsame
        
        ## if we're using the same frequencies and times, first check to see if there's already a response connected to the submodel:
        if tf_same and hasattr(self,'response_mat'):
            print("Attempted to recompute response matrix, but there is already an attached response matrix at these times and frequencies. Returning the original...")
            return self.response_mat
        else:
            return self.response(f0,tsegmid,**self.response_kwargs)



###################################################
###      UNIFIED MODEL PRIOR & LIKELIHOOD       ###
###################################################


class Model():
    '''
    Class to house all model attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid,rmat):
        
        '''
        Model() parses a Model string from the params file. This is of the form of an arbitrary number of "+"-delimited submodel types.
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "noise+powerlaw_isgwb+truncated-powerlaw_sph" defines a model with noise, an isotropic SGWB with a power law spectrum,
            and a (spherical harmonic model for) an anisotropic SGWB with a truncated power law spectrum.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        rmat (array)        : the data correllation matrix for all LISA arms
        
        Returns
        ------------
        Model (object) : Unified Model comprised of an arbitrary number of noise/signal submodels, with a corresponding unified prior and likelihood.
        
        '''
        
        self.fs = fs
        self.params = params
        
        ## separate into submodels
        self.submodel_names = params['model'].split('+')
        
        ## separate into submodels
        base_component_names = params['model'].split('+')
        
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate submodel names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
        self.submodel_names = catch_duplicates(base_component_names)
        suffixes = gen_suffixes(base_component_names)
        
        ## initialize submodels
        self.submodels = {}
        self.Npar = 0
        self.parameters = {}
        all_parameters = []
        spectral_parameters = []
        spatial_parameters = []
        for submodel_name, suffix in zip(self.submodel_names,suffixes):
            sm = submodel(params,inj,submodel_name,fs,f0,tsegmid,suffix=suffix)
            self.submodels[submodel_name] = sm
            self.Npar += sm.Npar
            self.parameters[submodel_name] = sm.parameters
            spectral_parameters += sm.spectral_parameters
            spatial_parameters += sm.spatial_parameters
            all_parameters += sm.parameters
        self.parameters['spectral'] = spectral_parameters
        self.parameters['spatial'] = spatial_parameters
        self.parameters['all'] = all_parameters
        
        ## update colors as needed
        catch_color_duplicates(self)
        
        ## assign reference to data for use in likelihood
        self.rmat = rmat
        self.f0 = f0
        self.tsegmid = tsegmid
        self.fixedL_channel_mode = np.any([
            getattr(self.submodels[sm_name], 'spatial_model_name', None) == 'fixedLchannels'
            for sm_name in self.submodel_names
        ])
        if self.fixedL_channel_mode:
            self.configure_fixedL_channel_mode()

    def configure_fixedL_channel_mode(self):
        '''
        Finalize the dedicated fixed-L channel-likelihood configuration.
        '''
        fixedL_submodels = [
            sm_name for sm_name in self.submodel_names
            if getattr(self.submodels[sm_name], 'spatial_model_name', None) == 'fixedLchannels'
        ]
        noise_submodels = [sm_name for sm_name in self.submodel_names if sm_name == 'noise']

        if len(fixedL_submodels) != 1 or len(noise_submodels) != 1 or len(self.submodel_names) != 2:
            raise ValueError(
                "The fixed-L channel likelihood currently supports exactly one 'noise' submodel "
                "and one 'powerlaw_fixedLchannels' submodel."
            )

        self.fixedL_signal_submodel_name = fixedL_submodels[0]
        self.fixedL_noise_submodel_name = noise_submodels[0]
        signal_submodel = self.submodels[self.fixedL_signal_submodel_name]
        noise_submodel = self.submodels[self.fixedL_noise_submodel_name]

        ## Use the dedicated parameter order theta = (log10_Ac, alpha, log_Np, log_Na).
        self.parameters['spectral'] = signal_submodel.parameters + noise_submodel.parameters
        self.parameters['spatial'] = []
        self.parameters['all'] = signal_submodel.parameters + noise_submodel.parameters
        self.Npar = len(self.parameters['all'])

        self.fixedL_channel_pair_indices = signal_submodel.fixedL_channel_pairs
        self.fixedL_channel_pairs = list(self.fixedL_channel_pair_indices.keys())
        self.fixedL_channel_response_segments = signal_submodel.fixedL_channel_response_segments
        self.fixedL_channel_response = signal_submodel.fixedL_channel_response
        self.fixedL_Nc = self.rmat.shape[1]
        (
            self.fixedL_channel_data_segments,
            self.fixedL_channel_data,
            self.fixedL_channel_data_imag_segments,
            self.fixedL_channel_data_imag,
        ) = self.build_fixedL_channel_data_product()

    def build_fixedL_channel_data_product(self):
        '''
        Build the real-valued channel-resolved fixed-L data product in Omega h^2 units.
        '''
        signal_submodel = self.submodels[self.fixedL_signal_submodel_name]
        omega_factor = signal_submodel.omega_to_sgw_factor(self.fs)
        data_segments = {}
        data_average = {}
        imag_segments = {}
        imag_average = {}

        for pair, (ii, jj) in self.fixedL_channel_pair_indices.items():
            pair_csd_omega = self.rmat[:, :, ii, jj] / omega_factor[:, None]
            data_segments[pair] = np.real(pair_csd_omega)
            data_average[pair] = np.mean(data_segments[pair], axis=1)
            imag_segments[pair] = np.imag(pair_csd_omega)
            imag_average[pair] = np.mean(imag_segments[pair], axis=1)

        return data_segments, data_average, imag_segments, imag_average

    def fixedL_channel_prior(self, unit_theta):
        '''
        Dedicated prior ordering for the fixed-L channel likelihood:
        (log10_Ac, alpha, log_Np, log_Na).
        '''
        signal_submodel = self.submodels[self.fixedL_signal_submodel_name]
        noise_submodel = self.submodels[self.fixedL_noise_submodel_name]
        theta = signal_submodel.prior(unit_theta[:signal_submodel.Npar])
        theta += noise_submodel.prior(unit_theta[signal_submodel.Npar:(signal_submodel.Npar + noise_submodel.Npar)])

        if len(theta) != len(unit_theta):
            raise ValueError("Fixed-L prior transform changed the dimensionality of theta.")

        return theta

    def compute_fixedL_channel_theory(self, theta):
        '''
        Compute the fixed-L signal, noise, and channel-resolved theory curves in Omega h^2 units.
        '''
        signal_submodel = self.submodels[self.fixedL_signal_submodel_name]
        log10_Ac, alpha, log_Np, log_Na = theta
        signal_omega = signal_submodel.fixedL_powerlaw_signal(self.fs, log10_Ac, alpha)
        noise_omega = signal_submodel.compute_fixedL_channel_noise_omega(self.fs, self.f0, log_Np, log_Na)
        theory = {
            pair: self.fixedL_channel_response[pair] * signal_omega + noise_omega[pair]
            for pair in self.fixedL_channel_pairs
        }

        return signal_omega, noise_omega, theory

    def likelihood_fixedL_channels(self, theta):
        '''
        Channel-resolved Gaussian likelihood for the fixed-L power-law SGWB mode.
        '''
        if self.params.get('fixedL_variance_model', 'theory_square') != 'theory_square':
            raise ValueError("Unsupported fixedL_variance_model '{}'; only 'theory_square' is implemented.".format(
                self.params.get('fixedL_variance_model')
            ))

        _, _, theory = self.compute_fixedL_channel_theory(theta)
        loglike = 0.0
        tiny = np.finfo(float).tiny
        for pair in self.fixedL_channel_pairs:
            residual = self.fixedL_channel_data[pair] - theory[pair]
            sigma2 = np.maximum(theory[pair]**2, tiny)
            loglike -= 0.5 * self.fixedL_Nc * np.sum(residual**2 / sigma2)

        return float(np.real(loglike))

    def save_fixedL_channel_diagnostics(self, post, out_dir):
        '''
        Save fixed-L channel diagnostics and minimal plots for the dedicated channel likelihood mode.
        '''
        if not self.fixedL_channel_mode:
            return

        if out_dir[-1] != '/':
            out_dir = out_dir + '/'

        median_theta = np.median(post, axis=0)
        signal_omega, noise_omega, theory = self.compute_fixedL_channel_theory(median_theta)
        signal_submodel = self.submodels[self.fixedL_signal_submodel_name]

        np.savez(
            out_dir + 'fixedL_signal_omega.npz',
            fs=self.fs,
            fixedL=signal_submodel.fixedL,
            fixedL_fc=self.params.get('fixedL_fc', 2.5e-3),
            log10_Ac=median_theta[0],
            alpha=median_theta[1],
            signal_omega=signal_omega,
            d_log10_Ac=signal_submodel.d_fixedL_d_log10_Ac(self.fs, median_theta[0], median_theta[1]),
            d_alpha=signal_submodel.d_fixedL_d_alpha(self.fs, median_theta[0], median_theta[1]),
        )
        np.savez(
            out_dir + 'fixedL_channel_response.npz',
            fs=self.fs,
            **{pair: self.fixedL_channel_response[pair] for pair in self.fixedL_channel_pairs}
        )
        np.savez(
            out_dir + 'fixedL_channel_noise_omega.npz',
            fs=self.fs,
            log_Np=median_theta[2],
            log_Na=median_theta[3],
            **{pair: noise_omega[pair] for pair in self.fixedL_channel_pairs}
        )
        np.savez(
            out_dir + 'fixedL_channel_data.npz',
            fs=self.fs,
            Nc=self.fixedL_Nc,
            **{pair: self.fixedL_channel_data[pair] for pair in self.fixedL_channel_pairs},
            **{pair + '_imag': self.fixedL_channel_data_imag[pair] for pair in self.fixedL_channel_pairs},
            **{pair + '_segments': self.fixedL_channel_data_segments[pair] for pair in self.fixedL_channel_pairs},
            **{pair + '_imag_segments': self.fixedL_channel_data_imag_segments[pair] for pair in self.fixedL_channel_pairs},
        )
        np.savez(
            out_dir + 'fixedL_channel_theory.npz',
            fs=self.fs,
            **{pair: theory[pair] for pair in self.fixedL_channel_pairs}
        )

        plt.close()
        plt.loglog(self.fs, signal_omega, color='royalblue')
        plt.xlabel('$f$ in Hz')
        plt.ylabel(r'$\Omega_{\mathrm{GW}}^L(f) h^2$')
        plt.title('Posterior-Median Fixed-L Signal')
        plt.savefig(out_dir + 'fixedL_signal_omega.png', dpi=200)
        plt.close()

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, pair in zip(axes.flatten(), self.fixedL_channel_pairs):
            ax.loglog(self.fs, np.maximum(self.fixedL_channel_response[pair], np.finfo(float).tiny), color='slateblue')
            ax.set_title(pair)
            ax.set_xlabel('$f$ [Hz]')
            ax.set_ylabel(r'$\tilde{R}_{OO^\prime,L}$')
        plt.tight_layout()
        plt.savefig(out_dir + 'fixedL_channel_response.png', dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for ax, pair in zip(axes.flatten(), self.fixedL_channel_pairs):
            ax.semilogx(self.fs, self.fixedL_channel_data[pair], label='Data', color='slategrey', alpha=0.8)
            ax.semilogx(self.fs, theory[pair], label='Theory', color='royalblue')
            ax.semilogx(self.fs, noise_omega[pair], label='Noise', color='dimgrey', ls='--')
            ax.set_title(pair)
            ax.set_xlabel('$f$ [Hz]')
            ax.set_ylabel(r'$\Omega h^2$')
        axes.flatten()[0].legend(loc='best', fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir + 'fixedL_channel_data_theory.png', dpi=200)
        plt.close(fig)
    

    def prior(self,unit_theta):
        '''
        Unified prior function to interatively perform prior draws for each submodel in the proper order
        
        Arguments
        ----------------
        unit_theta (array) : draws from the unit cube
        
        Returns
        ----------------
        theta (list) : transformed prior draws for all submodels in sequence
        '''
        if self.fixedL_channel_mode:
            return self.fixedL_channel_prior(unit_theta)

        theta = []
        start_idx = 0
        
        for sm_name in self.submodel_names:
            sm = self.submodels[sm_name]
            theta += sm.prior(unit_theta[start_idx:(start_idx+sm.Npar)])
            start_idx += sm.Npar
        
        if len(theta) != len(unit_theta):
            raise ValueError("Input theta does not have same length as output theta, something has gone wrong!")
        
        return theta
    
    
    def likelihood(self,theta):
        '''
        Unified likelihood function to compare the combined covariance contributions of a generic set of noise/SGWB models to the data.
        
        Arguments
        ----------------
        theta (list) : transformed prior draws for all submodels in sequence
        
        Returns
        ----------------
        loglike (float) : resulting joint log likelihood
        '''
        if self.fixedL_channel_mode:
            return self.likelihood_fixedL_channels(theta)

        start_idx = 0
        for i, sm_name in enumerate(self.submodel_names):
            sm = self.submodels[sm_name]
            theta_i = theta[start_idx:(start_idx+sm.Npar)]
            start_idx += sm.Npar
            if i==0:
                cov_mat = sm.cov(theta_i)
            else:
                cov_mat = cov_mat + sm.cov(theta_i)

        ## change axis order to make taking an inverse easier
        cov_mat = np.moveaxis(cov_mat, [-2, -1], [0, 1])

        ## take inverse and determinant
        inv_cov, det_cov = bespoke_inv(cov_mat)

        logL = -np.einsum('ijkl,ijkl', inv_cov, self.rmat) - np.einsum('ij->', np.log(np.pi * self.params['seglen'] * np.abs(det_cov)))


        loglike = np.real(logL)

        return loglike
    

###################################################
###       UNIFIED INJECTION INFRASTRUCTURE      ###
################################################### 

    
class Injection():#geometry,sph_geometry):
    '''
    Class to house all injection attributes in a modular fashion.
    '''
    def __init__(self,params,inj,fs,f0,tsegmid):
        '''
        Injection() parses a Injection string from the params file. This is of the form of an arbitrary number of "+"-delimited submodel types.
        Each submodel should be defined as "[spectral]_[spatial]", save for the noise model, which is just "noise".
        
        e.g., "noise+powerlaw_isgwb+truncated-powerlaw_sph" defines an injection with noise, an isotropic SGWB with a power law spectrum,
            and a (spherical harmonic description of) an anisotropic SGWB with a truncated power law spectrum.
        
        Arguments
        ------------
        params, inj (dict)  : params and inj config dictionaries as generated in run_blip.py
        fs, f0 (array)      : frequency array and its LISA-characteristic-frequency-scaled counterpart (f0=fs/(2*fstar))
        tsegmid (array)     : array of time segment midpoints
        
        Returns
        ------------
        Injection (object)  : Unified Injection comprised of an arbitrary number of noise/signal injection components, with a variety of helper functions to aid in the BLIP injection procedure.
        
        '''
        self.params = params
        self.inj = inj
        
        self.frange = fs
        self.f0 = f0
        self.tsegmid = tsegmid
        
        ## separate into components
        self.component_names = inj['injection'].split('+')
        
        ### commenting this out because we're switching to active specification of duplicates in the params file
        ## check for and differentiate duplicate injections
        ## this will append 1 (then 2, then 3, etc.) to any duplicate component names
        ## we will also generate appropriate variable suffixes to use in plots, etc..
#        self.component_names = catch_duplicates(base_component_names)
        
        ## it's useful to have a version of this without the detector noise
        self.sgwb_component_names = [name for name in self.component_names if name!='noise']
        suffixes = gen_suffixes(self.component_names)
                        
        ## initialize components
        self.components = {}
        self.truevals = {}
        for component_name, suffix in zip(self.component_names,suffixes):
            cm = submodel(params,inj,component_name,fs,f0,tsegmid,injection=True,suffix=suffix)
            self.components[component_name] = cm
            self.truevals[component_name] = cm.truevals
            if cm.has_map:
                self.plot_skymaps(component_name)
        
        ## update colors as needed
        catch_color_duplicates(self)
    
    
    
    def compute_convolved_spectra(self,component_name,fs_new=None,channels='11',return_fs=False,imaginary=False):
        '''
        Wrapper to return the frozen injected detector-convolved GW spectra for the desired channels.
        
        Useful note - these frozen spectra are computed in diag_spectra(), as they are calculated and saved at the analysis frequencies.
        
        Also note that this is meant for plotting purposes only, and includes interpolation/absolute values that are not desirable in a data generation/analysis environment.
        
        Arguments
        -----------
        component_name (str) : the name (key) of the Injection component to use.
        fs_new (array) : If desired, frequencies at which to interpolate the convolved PSD
        channels (str) : Which channel cross/auto-correlation PSD to plot. Default is '11' auto-correlation, i.e. XX for XYZ, 11 for Michelson, AA for AET.
        return_fs (bool) : If True, also returns the frequencies at which the PSD has been evaluated. Default False.
        imaginary (bool) : If True, returns the magnitude of the imaginary component. Default False.
        
        Returns
        -----------
        PSD (array) : Power spectral density of the specified channels' auto/cross-correlation at the desired frequencies.
        fs (array, optional) : The PSD frequencies, if return_fs==True.
        
        '''
        
        cm = self.components[component_name]
        ## split the channel indicators
        c1_idx, c2_idx = int(channels[0]) - 1, int(channels[1]) - 1
        
        if not imaginary:
            PSD = np.abs(np.real(cm.frozen_convolved_spectra[c1_idx,c2_idx,:]))
        else:
            PSD = np.abs(np.imag(cm.frozen_convolved_spectra[c1_idx,c2_idx,:]))
        
        ## populations need some finessing due to frequency subtleties                
        if hasattr(cm,"ispop") and cm.ispop:
            fs = cm.population.frange_true
            if (fs_new is not None) and not np.array_equal(fs_new,cm.population.frange_true):
                with log_manager(logging.ERROR):
                    PSD_interp = interp1d(fs,PSD)
                    PSD = PSD_interp(fs_new)
                    fs = fs_new
        else:
            fs = self.frange
            if fs_new is not None:
                with log_manager(logging.ERROR):
                    PSD_interp = interp1d(fs,np.log10(np.maximum(PSD, np.finfo(float).tiny)))
                    PSD = 10**PSD_interp(fs_new)
                    fs = fs_new

        if imaginary:
            PSD = 1j * PSD

        if return_fs:
            return fs, PSD
        else:
            return PSD
        
    
    def plot_injected_spectra(self,component_name,fs_new=None,ax=None,convolved=False,legend=False,channels='11',return_PSD=False,scale='log',flim=None,ymins=None,**plt_kwargs):
        '''
        Wrapper to plot the injected spectrum component on the specified matplotlib axes (or current axes if unspecified).
        
        Arguments
        -----------
        component_name (str) : the name (key) of the Injection component to use.
        fs_new (array) : If desired, frequencies at which to interpolate the convolved PSD
        ax (matplotlib axes) : Axis on which to plot. Default None (will plot on current axes.)
        convolved (bool) : If True, convolve the injected spectra with the detector response. Default False.
        legend (bool) : If True, generate a legend entry. Default False.
        channels (str) : Which channel cross/auto-correlation PSD to plot. Default is '11' auto-correlation, i.e. XX for XYZ, 11 for Michelson, AA for AET.
        return_PSD (bool) : If True, also returns the plotted PSD. Default False.
        scale (str) : Matplotlib scale at which to plot ('log' or 'linear'). Default 'log'.
        flim (tuple) : (fmin,fmax) plot limits. Default None (will use fmin,fmax as specified in the params file.)
        ymins (list) : External list to which, if specified, will be added the lower ylim of the injected spectra.
        **plt_kwargs (kwargs) : matplotlib.pyplot keyword arguments
        
        Returns
        -----------
        PSD plot on specified axes.
        PSD (array, optional) : Power spectral density of the specified channels' auto/cross-correlation at the desired frequencies.

        '''
        ## grab component
        cm = self.components[component_name]
        
        ## set axes
        if ax is None:
            ax = plt.gca()
        
        ## set fmin/max to specified values, or default to the ones in params
        if flim is not None:
            fmin = flim[0]
            fmax = flim[1]
        else:
            fmin = self.params['fmin']
            fmax = self.params['fmax']
        
        ## special treatment of population frequencies
#        if hasattr(self.components[component_name],"ispop") and self.components[component_name].ispop:
#            fs_base = self.components[component_name].population.frange_true
#        else:
#        fs_base = self.frange
        
        ## get frozen injected spectra at original injection frequencies and convolve with detector response if desired
        if convolved:
            if component_name == 'noise':
                raise ValueError("Cannot convolve noise spectra with the detector GW response - this is not physical. (Set convolved=False in the function call!)")
            fs, PSD = self.compute_convolved_spectra(component_name,channels=channels,return_fs=True,fs_new=fs_new)
        else:
            ## special treatment for the population case
            if hasattr(cm,"ispop") and cm.ispop:
                PSD = cm.population.Sgw_true
                fs = cm.population.frange_true
                if fs_new is not None and not np.array_equal(fs_new,cm.population.frange_true):
                    ## the interpolator gets grumpy sometimes, but it's not an actual issue hence the logging wrapper
                    with log_manager(logging.ERROR):
                        PSD_interp = interp1d(fs,PSD)
                        PSD = PSD_interp(fs_new)
                        fs = fs_new
            else:
                PSD = cm.frozen_spectra
                ## noise will return the 3x3 covariance matrix, need to grab the desired channel cross-/auto-power
                ## generically capture anything that looks like a covariance matrix for future-proofing
                if (len(PSD.shape)==3) and (PSD.shape[0]==PSD.shape[1]==3):
                    I, J = int(channels[0]) - 1, int(channels[1]) - 1
                    PSD = PSD[I,J,:]

                ## downsample (or upsample, but why) if desired
                ## do the interpolation in log-space for better low-f fidelity
                if fs_new is not None:
                    with log_manager(logging.ERROR):
                        PSD_interp = interp1d(self.frange,np.log10(PSD))
                        PSD = 10**PSD_interp(fs_new)
                        fs = fs_new
                else:
                    fs = self.frange
        
        filt = (fs>fmin)*(fs<fmax)
        
        if legend:
            label = cm.fancyname
            if plt_kwargs is None:
                plt_kwargs = {}
                plt_kwargs['label'] = label
            else:
                if 'label' not in plt_kwargs.keys():
                    plt_kwargs['label'] = label
        
        if scale=='log':
            ax.loglog(fs[filt],PSD[filt],**plt_kwargs)
        elif scale=='linear':
            ax.plot(fs[filt],PSD[filt],**plt_kwargs)
        else:
            raise ValueError("We only support linear and log plots, there is no secret third option!")
        
        if ymins is not None:
            ymins.append(PSD.min())
        
        if return_PSD:
            return PSD
        else:
            return
        
    def plot_skymaps(self,component_name,**plt_kwargs):
        '''
        Function to plot the injected skymaps.
        
        NOTE - will need to be generalized when I add the astro injections
        '''
        cm = self.components[component_name]
        
        # deals with projection parameter 
        if self.params['projection'] is None:
            coord = 'E'
        elif self.params['projection']=='G' or self.params['projection']=='C':
            coord = ['E',self.params['projection']]
        elif self.params['projection']=='E':
            coord = self.params['projection']
        else:  
            raise TypeError('Invalid specification of projection, projection can be E, G, or C')
        
        ## dimensionless energy density at 1 mHz
        spec_args = [cm.truevals[parameter] for parameter in cm.spectral_parameters]
        Omega_1mHz = cm.omegaf(1e-3,*spec_args)
        Omegamap_inj = Omega_1mHz * cm.sph_skymap

        hp.mollview(Omegamap_inj, coord=coord, title='Injected angular distribution map $\Omega (f = 1 mHz)$', unit="$\\Omega(f= 1mHz)$")
        hp.graticule()
        
        plt.savefig(self.params['out_dir'] + '/inj_skymap'+component_name+'.png', dpi=150)
        print('saving injected skymap at ' +  self.params['out_dir'] + '/inj_skymap'+component_name+'.png')
        plt.close()
        
        return

    

def gen_blm_parameters(blmax):
    '''
    Function to make the blm parameter name strings for all blms of a given lmax, in the correct order.
    
    Arguments
    -----------
    blmax (int) : lmax for the blms
    
    Returns
    -----------
    blm_parameters (list of str) : Ordered list of blm parameter name strings
    
    '''
    
    blm_parameters = []
    for lval in range(1, blmax + 1):
        for mval in range(lval + 1):

            if mval == 0:
                blm_parameters.append(r'$b_{' + str(lval) + str(mval) + '}$' )
            else:
                blm_parameters.append(r'$|b_{' + str(lval) + str(mval) + '}|$' )
                blm_parameters.append(r'$\phi_{' + str(lval) + str(mval) + '}$' )
    
    return blm_parameters


def bespoke_inv(A):


    """

    compute inverse without division by det; ...xv3xc3 input, or array of matrices assumed

    Credit to Eelco Hoogendoorn at stackexchange for this piece of wizardy. This is > 3 times
    faster than numpy's det and inv methods used in a fully vectorized way as of numpy 1.19.1

    https://stackoverflow.com/questions/21828202/fast-inverse-and-transpose-matrix-in-python

    """


    AI = np.empty_like(A)

    for i in range(3):
        AI[...,i,:] = np.cross(A[...,i-2,:], A[...,i-1,:])

    det = np.einsum('...i,...i->...', AI, A).mean(axis=-1)

    inv_T =  AI / det[...,None,None]

    # inverse by swapping the inverse transpose
    return np.swapaxes(inv_T, -1,-2), det
