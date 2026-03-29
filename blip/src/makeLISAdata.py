import numpy as np
import os

class LISAdata():

    '''
    Class for lisa data. Includes methods for generation of gaussian instrumental noise, and generation
    of isotropic stochastic background. Any eventually signal models should be added as methods here. This
    has the Antennapatterns class as a super class.
    '''

    def __init__(self, params, inj):
        self.params = params
        self.inj = inj
        self.armlength = 2.5e9 ## armlength in meters

    def resolve_input_spectrum_path(self):

        '''
        Resolve the configured cached spectrum path.

        Bare filenames are stored inside params['out_dir'], preserving the
        historic BLIP behavior. Paths that already include a directory are
        treated as user-specified locations relative to the working directory
        unless they are already absolute.
        '''

        input_spectrum = self.params['input_spectrum']
        if os.path.isabs(input_spectrum):
            return input_spectrum
        if os.path.dirname(input_spectrum):
            return os.path.abspath(input_spectrum)
        return os.path.join(self.params['out_dir'], input_spectrum)

    def resolve_datafile_path(self):

        '''
        Resolve the configured external time-domain data file path.

        Data files are interpreted relative to the current working directory
        unless an absolute path is supplied.
        '''

        datafile = self.params['datafile']
        if datafile is None:
            return None
        if os.path.isabs(datafile):
            return datafile
        return os.path.abspath(datafile)

    def build_segment_times(self, nsegs):

        '''
        Reconstruct the segment start and midpoint arrays from the config.

        Cached frequency-domain spectra do not store the original time array,
        but BLIP only needs the per-segment timing for the detector response.
        '''

        tsegstart = self.params['tstart'] + self.params['seglen'] * np.arange(nsegs)
        tsegmid = tsegstart + 0.5 * self.params['seglen']
        return tsegstart, tsegmid

    def compute_covariance_square_root(self, covariance_stack):

        '''
        Return a matrix square root for a stack of Hermitian covariance matrices.

        The fast path uses a Cholesky decomposition. When numerical roundoff or an
        only-semi-definite anisotropic response prevents a strict Cholesky factor,
        fall back to an eigendecomposition and clip negative modes to zero.
        '''

        covariance_stack = 0.5 * (covariance_stack + np.conjugate(np.swapaxes(covariance_stack, -1, -2)))

        try:
            return np.linalg.cholesky(covariance_stack), None
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(covariance_stack)
            clipped_eigvals = np.clip(eigvals, 0.0, None)
            factor = eigvecs * np.sqrt(clipped_eigvals)[:, None, :]

            min_eigvals = eigvals[:, 0]
            largest_abs_eigvals = np.max(np.abs(eigvals), axis=1)
            worst_idx = int(np.argmin(min_eigvals))
            largest_abs_eig = float(largest_abs_eigvals[worst_idx])
            worst_min_eig = float(min_eigvals[worst_idx])
            relative_violation = 0.0
            if largest_abs_eig > 0.0:
                relative_violation = float(max(0.0, -worst_min_eig) / largest_abs_eig)

            stats = {
                'num_nonpositive': int(np.count_nonzero(min_eigvals <= 0.0)),
                'num_negative': int(np.count_nonzero(min_eigvals < 0.0)),
                'worst_min_eig': worst_min_eig,
                'largest_abs_eig': largest_abs_eig,
                'relative_violation': relative_violation,
            }

            return factor, stats


    ## Method for reading frequency domain spectral data if given in an npz file
    def read_spectrum(self):

        '''
        Read an input frequency domain data file. Returns the fourier transform of the data from the three channels and an array of reference frequencyes

        Returns
        ---------

        rA, rE, rT, fdata   :   float

        '''

        spectrum_path = self.resolve_input_spectrum_path()

        if os.path.isfile(spectrum_path) and not self.params['doPreProc']:
            print("loading freq domain data from input file {}".format(spectrum_path))

            data = np.load(spectrum_path)
            r1    = data['r1']
            r2    = data['r2']
            r3    = data['r3']
            fdata = data['fdata']

            return r1, r2, r3, fdata
        return None



    def add_sgwb_data(self, injmodel, tbreak = 0.0):
        
 
        N = self.Injection.Npersplice
        halfN = int(0.5*N)
        
        ## compute the astrophysical spectrum
        injmodel_args = [injmodel.truevals[parameter] for parameter in injmodel.spectral_parameters]
        
        Sgw = injmodel.compute_Sgw(self.Injection.frange,injmodel_args)
        
        injmodel.frozen_spectra = Sgw
        
        ## the spectrum of the frequecy domain gaussian for ifft
        norms = np.sqrt(self.params['fs']*Sgw*N)/2

        ## index array for one segment
        t_arr = np.arange(N)

        ## the window for splicing
        splice_win = np.sin(np.pi * t_arr/N)

        psd_projection_stats = []

        ## Loop over splice segments
        for ii in range(self.Injection.nsplice):
            ## move frequency to be the zeroth-axis, then cholesky decomp
            response_factor, projection_stats = self.compute_covariance_square_root(
                np.moveaxis(injmodel.inj_response_mat[:, :, :, ii], -1, 0)
            )
            if projection_stats is not None and projection_stats['num_nonpositive'] > 0:
                projection_stats['segment_index'] = ii
                psd_projection_stats.append(projection_stats)

            L_cholesky = norms[:, None, None] * response_factor
            
            ## generate standard normal complex data first
            z_norm = np.random.normal(size=(self.Injection.frange.size, 3)) + 1j * np.random.normal(size=(self.Injection.frange.size, 3))

            ## The data in z_norm is rescaled into z_scale using L_cholesky
            z_scale = np.einsum('ijk, ikl -> ijl', L_cholesky, z_norm[:, :, None])[:, :, 0]

            ## The three channels : concatenate with norm at f = 0 to be zero
            htilda1  = np.concatenate([ [0], z_scale[:, 0]])
            htilda2  = np.concatenate([ [0], z_scale[:, 1]])
            htilda3  = np.concatenate([ [0], z_scale[:, 2]])


            if ii == 0:
                # Take inverse fft to get time series data
                h1 = splice_win * np.fft.irfft(htilda1, N)
                h2 = splice_win * np.fft.irfft(htilda2, N)
                h3 = splice_win * np.fft.irfft(htilda3, N)

            else:

                ## First append half-splice worth of zeros
                h1 = np.append(h1, np.zeros(halfN))
                h2 = np.append(h2, np.zeros(halfN))
                h3 = np.append(h3, np.zeros(halfN))

                ## Then add the new splice segment
                h1[-N:] = h1[-N:] + splice_win * np.fft.irfft(htilda1, N)
                h2[-N:] = h2[-N:] + splice_win * np.fft.irfft(htilda2, N)
                h3[-N:] = h3[-N:] + splice_win * np.fft.irfft(htilda3, N)

        if len(psd_projection_stats) > 0:
            total_projected = int(np.sum([stats['num_nonpositive'] for stats in psd_projection_stats]))
            total_negative = int(np.sum([stats['num_negative'] for stats in psd_projection_stats]))
            worst_stats = min(psd_projection_stats, key=lambda stats: stats['worst_min_eig'])
            print(
                "[BLIP injection] projected {} non-positive response bins ({} strictly negative) onto the PSD cone for {}. "
                "Worst min eigenvalue = {:.3e} (relative {:.3e}) at splice segment {}/{}."
                .format(
                    total_projected,
                    total_negative,
                    injmodel.name,
                    worst_stats['worst_min_eig'],
                    worst_stats['relative_violation'],
                    worst_stats['segment_index'] + 1,
                    self.Injection.nsplice,
                )
            )

        ## remove the first half and the last half splice.
        h1, h2, h3 = h1[halfN:-halfN], h2[halfN:-halfN], h3[halfN:-halfN]

        tarr = self.params['tstart'] + tbreak +  np.arange(0, self.params['dur'], 1.0/self.params['fs'])

        return h1, h2, h3, tarr

    def read_data(self):

        '''
        Read mldc or other external domain data from an ascii txt file. Since this was used primarily for
        the MLDC, it assumes that the data is given in X,Y and Z channels.
        Returns
        ---------

        h1, h2, h3 : float
            Time series data for the three TDI channels


        '''

        datafile_path = self.resolve_datafile_path()
        if datafile_path is None:
            raise FileNotFoundError("No external datafile was configured.")
        if not os.path.isfile(datafile_path):
            raise FileNotFoundError(
                "{} not found. Set [params] datafile to an existing ASCII time-domain file, "
                "or set [run_params] input_spectrum to an existing cached .npz file with doPreProc=0."
                .format(datafile_path)
            )

        hoft = np.loadtxt(datafile_path)

        fs_default = 1.0/(hoft[1, 0] - hoft[0, 0])

        ## Read in the duration seconds of data + one segment of buffer
        end_idx = int((self.params['dur'] + self.params['seglen'])*fs_default)

        ## the mldc data is X,Y,Z tdi
        times, h1, h2, h3 = hoft[0:end_idx, 0], hoft[0:end_idx, 1], hoft[0:end_idx, 2], hoft[0:end_idx, 3]

        delt = times[1] - times[0]


        ## Check if the requested sampel rate is consistant
        if self.params['fs'] != 1.0/delt:
            self.params['fs'] = 1.0/delt

        return h1, h2, h3, times

    def process_external_data(self):
        '''
        Just a wrapper function to use the methods the LISAdata class to
        read data. Return frequency domain data. Since this was used
        primarily for the MLDC, this assumes that the data is doppler
        tracking and converts to strain data.
        '''

        cached_spectrum = self.read_spectrum()
        if cached_spectrum is not None:
            self.r1, self.r2, self.r3, self.fdata = cached_spectrum
            nsegs = self.r1.shape[1] if self.r1.ndim > 1 else 1
            self.tsegstart, self.tsegmid = self.build_segment_times(nsegs)

            # Characteristic frequency. Define f0.
            cspeed = 3e8
            fstar = cspeed/(2*np.pi*self.armlength)
            self.f0 = self.fdata/(2*fstar)

            # Cached spectra should already reflect the configured datatype/TDI
            # choice from the preprocessing run, so no additional conversion is
            # needed here.
            return

        h1, h2, h3, self.timearray = self.read_data()

        # Calculate other tdi combinations if necessary.
        if self.params['tdi_lev'] == 'aet':
            h1 = (1.0/3.0)*(2*h1 - h2 - h3)
            h2 = (1.0/np.sqrt(3.0))*(h3 - h2)
            h3 = (1.0/3.0)*(h1 + h2 + h3)

        # Generate lisa freq domain data from time domain data
        self.r1, self.r2, self.r3, self.fdata, self.tsegstart, self.tsegmid = self.tser2fser(h1, h2, h3, self.timearray)

        # Charactersitic frequency. Define f0
        cspeed = 3e8
        fstar = cspeed/(2*np.pi*self.armlength)
        self.f0 = self.fdata/(2*fstar)

        # Convert doppler data to strain if readfile datatype is doppler.
        if self.params['datatype'] == 'doppler':

            # This is needed to convert from doppler data to strain data.
            self.r1, self.r2, self.r3 = self.r1/(4*self.f0.reshape(self.f0.size, 1)), self.r2/(4*self.f0.reshape(self.f0.size, 1)), self.r3/(4*self.f0.reshape(self.f0.size, 1))

        elif self.params['datatype'] == 'strain':
            pass


    def tser2fser(self, h1, h2, h3, timearray):

        '''
        Convert time domain data to fourier domain and return ffts. The convention is that the
        the ffts are divided by the sampling frequency and corrected for windowing. A hann window
        is applied by default when moving to the fourier domain. The ffts are also normalized so that
        thier square gives the PSD.

        Parameters
        -----------
        h1, h2, h3 : float
            time series data for the three input channels

        timearray : float
            times corresponding to data in h1, h2, h3

        Returns
        ---------

        r1, r2, r3 : float
            frequency series data for the three input channels

        fdata : float
            Reference frequency series

        tsegstart : float
            Segmented time array giving segment start points

        tsegmid : float
            Segmented time array giving segment midpoints


        '''

        print ("Calculating fourier spectra... ")
        # Number of segmants
        nsegs = int(np.floor(self.params['dur']/self.params['seglen'])) -1

        Nperseg=int(self.params['fs']*self.params['seglen'])

        '''
        # Apply a cascading low pass filter
        b, a = sg.butter(2, 0.4*self.params['fs']/(self.params['fs']/2),\
                btype='lowpass', output='ba')
        #sos = sg.zpk2sos(zz, pp, kk)

        for ii in range(8):
            print('low pass filtering ...')
            h1 = sg.filtfilt(b, a, h1)
            h2 = sg.filtfilt(b, a, h2)
            h3 = sg.filtfilt(b, a, h3)
        '''

        fftfreqs = np.fft.rfftfreq(Nperseg, 1.0/self.params['fs'])


        # Map of spectrum
        r1 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r2 = np.zeros((fftfreqs.size, nsegs), dtype='complex')
        r3 = np.zeros((fftfreqs.size, nsegs), dtype='complex')


        # Hann Window
        hwin = np.hanning(Nperseg)
        win_fact = np.mean(hwin**2)


        zpad = np.zeros(Nperseg)

        ## Initiate time segment arrays
        tsegstart = np.zeros(nsegs)
        tsegmid = np.zeros(nsegs)

        # We will use 50% overlapping segments
        for ii in range(0, nsegs):

            idxmin = int(ii*Nperseg)
            idxmax = idxmin + Nperseg
            idxmid = idxmin + int(Nperseg/2)
            if hwin.size != h1[idxmin:idxmax].size:
                import pdb; pdb.set_trace()

            r1[:, ii] =   np.fft.rfft(hwin*h1[idxmin:idxmax], axis=0)
            r2[:, ii] =   np.fft.rfft(hwin*h2[idxmin:idxmax], axis=0)
            r3[:, ii] =   np.fft.rfft(hwin*h3[idxmin:idxmax], axis=0)


            ## There's probably a more pythonic way of doing this, but it'll work for now.
            tsegstart[ii] = timearray[idxmin]
            tsegmid[ii] = timearray[idxmid]

        # "Cut" to desired frequencies
        idx = np.logical_and(fftfreqs >=  self.params['fmin'] , fftfreqs <=  self.params['fmax'])

        # Output arrays
        fdata = fftfreqs[idx]


        # Get desired frequencies only
        # We want to normalize ffts so thier square give the psd
        # win_fact is to adjust for hann windowing, sqrt(2) for single sided
        r1 = np.sqrt(2/win_fact)*r1[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r2 = np.sqrt(2/win_fact)*r2[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))
        r3 = np.sqrt(2/win_fact)*r3[idx, :]/(self.params['fs']*np.sqrt(self.params['seglen']))


        spectrum_path = self.resolve_input_spectrum_path()
        spectrum_dir = os.path.dirname(spectrum_path)
        if spectrum_dir:
            os.makedirs(spectrum_dir, exist_ok=True)
        np.savez(spectrum_path, r1=r1, r2=r2, r3=r3, fdata=fdata)

        return r1, r2, r3, fdata, tsegstart, tsegmid

    
