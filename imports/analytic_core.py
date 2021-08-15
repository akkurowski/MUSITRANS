# -------------------------------------------------------------------------
# Music Pieces Melody Extractor Core Functions
# author: Adam Kurowski
# e-mail: akkurowski@gmail.com
# date:   23.07.2021
# -------------------------------------------------------------------------

import os
import copy as cp
import numpy as np
import numba as nb
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavio
from imports.tmp_files import *
from imports.util_communication import *
from scipy.signal.windows import kaiser, hamming, hann, blackman

# -------------------------------------------------------------------------
# helper procedures

# A procedure for automated reading of audio files even if they're
# encoded with FFMPEG-managable compressed audio formats (REQUIRES
# FFmpeg software installed and callable from the cmd/PowerShell,
# check if this is true BEFORE STARTING TO RELY on this feature).
def read_file(tmp_dir, fpath):

    # Initializing the TMP folder.
    tmp_files_mngr = TMPFilesManagerClass(tmp_dir)
    tmp_files_mngr.clean_tmp_folder()
    
    fname_base, fname_ext = os.path.splitext(os.path.basename(fpath))
    
    # No additional steps if WAVE files are processed.
    if fname_ext == '.wav':
        wav_read_fpath = fpath
    
    # If the file is not a WAVE file, then we need FFMPEG.
    else:
        infile_path  = fpath
        outfile_path = tmp_files_mngr.obtain_tmp_path('.wav')
        
        cmd = f'ffmpeg -i "{infile_path}" "{outfile_path}"'
        print(f'executing: {cmd}')
        
        os.system(cmd)
        
        if not os.path.isfile(outfile_path):
            print()
            print('---------------------------------------------------------------------------')
            print('ATENTION! FFmpeg-based audio file decoding process failed')
            print("If you don't have FFmpeg installed, download it from https://www.ffmpeg.org")
            print("Also, remember to add it to the Path environment variable")
            print("so it can be called from the console. Without FFmpeg you can")
            print("only use WAVE files.")
            print("The script will be now terminated.")
            print('---------------------------------------------------------------------------')
            print()
            exit()
            
        
        wav_read_fpath = outfile_path
    
    # After optional decoding (or without this step) - we have WAVE
    # file and we can tead it, optionally clean the TMP folder, and
    # return the result.
    fs,audio_data = wavio.read(wav_read_fpath)
    tmp_files_mngr.clean_current_tmp_files()
    
    return fs,audio_data

# Some processed files may be in a stereo format, this is a unified 
# procedure to convert them to mono.
def stereo2mono(audio_data):
    if len(audio_data.shape) == 2:
        audio_data = np.sum(audio_data, axis=1)
    return audio_data

# Fast, Numba-accelerated exponential averaging
@nb.jit
def exp_avg(input,alpha):
    output  = np.zeros_like(input)
    acc     = input[0]
    for i in range(0,len(input)-1):
        output[i] = acc
        acc = acc*(1-alpha) + alpha*input[i+1]
    output[-1] = acc
    return output


# Data have be to split into separate frames. Frames can overlap,
# thus we need a specific function to achieve such a goal.
def split2frames(audio_data, frame_length, noverlap=0):
    
    # Initialize the list for our result data.
    frames_of_data = []
    
    # Establish the initial indices values for starting and 
    # ending points of the frame.
    frame_start = 0
    frame_end   = frame_length
    
    # Until the algorithm hasn't reach the end of the file,
    # keep extracting frames by shifting the indices of starting
    # and ending point of current frame
    while True:
        frame = audio_data[frame_start:frame_end]
        frames_of_data.append(frame)
        frame_start += (frame_length - noverlap)
        frame_end   += (frame_length - noverlap)
        
        if frame_end >= len(audio_data):
            break
    
    # If the job is done, convert the data to a Numpy arrat,
    # and return it.
    return np.array(frames_of_data)

# A procedure of obtaining an oversampled FFT from a given audio signal.
def ov_spectrum(audio_frame, spectrum_ovsmpl = 32):
    
    # Oversampling is achieved by simply "adding zeros" to the input
    # to increase the length of the input vectors, which increases
    # the sampling density of the output:
    # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
    # https://stackoverflow.com/questions/23812541/is-my-understanding-of-fft-and-pitch-detection-correct-here
    
    # First - define the variable.
    audio_frame_ov = np.zeros(len(audio_frame)*spectrum_ovsmpl)
    
    # Next - fill the first values with the contents of an audio frame.
    audio_frame_ov[0:len(audio_frame)] = audio_frame
    
    # Rest of steps is standard - the output is an amplitude spectrum.
    frame_spectr  = np.fft.fft(audio_frame_ov)
    frame_spectr  = frame_spectr[0:len(frame_spectr)//2]
    frame_spectr  = np.abs(frame_spectr)
    
    return frame_spectr

# For detection of f0 a whole spectrogram is necessary, and the function below may be used to obtain it.
def oversampled_spectrogram(audio_signal, fs, frame_length, wnd_type='Kaiser', beta=14, ovlp_factor=0.9, spectrum_ovsmpl = 32):
    
    # Splitting the data into frames.
    audio_frames  = split2frames(audio_signal, frame_length, int(frame_length*ovlp_factor))
    
    # Definition of the output structure.
    output_data   = np.empty((frame_length*spectrum_ovsmpl//2, len(audio_frames)))
    
    print('computing oversampled spectrogram')
    
    # If a window function was specified - this is where it is calculated.
    if   wnd_type == 'rectangular':
        window_function = None
    elif wnd_type == 'Kaiser':
        window_function = kaiser(frame_length, beta=beta)
    elif wnd_type == 'Hamming':
        window_function = hamming(frame_length)
    elif wnd_type == 'Hann':
        window_function = hann(frame_length)
    elif wnd_type == 'Blackman':
        window_function = blackman(frame_length)
    else:
        raise RuntimeError('Bad window type was provided.')
    
    # For each frame multiple it by the window function (if necessary),
    # and calculate the oversampled FFT-based amplitude spectra as rows
    # of the output spectrogram.
    i = 0
    for audio_frame in tqdm(audio_frames):
        if window_function is not None:
            windowed_frame = audio_frame
        else:
            windowed_frame = audio_frame*window_function
        output_data[:,i] = ov_spectrum(windowed_frame, spectrum_ovsmpl)
        i += 1
    print()
    
    return output_data

# A Numba-enhanced function for calculating a membership function
# of e triangle-like shape (can be modified with the a parameter).
# It is used to implement triangular filters for the pre-chromogram
# calculation.
# More info:
# https://www.mathworks.com/help/fuzzy/trimf.html
# https://stackoverflow.com/questions/40197060/librosa-mel-filter-bank-decreasing-triangles
@nb.jit
def triangular_membership(f_lo, f_c, f_hi, f, a):
    # If the central frequency equalt to the checked frequency f, the output is 1
    if f == f_c:  return 1.
    
    # If f is outside the span of a triangle, the output sure is 0!
    if f  > f_hi: return 0.
    if f  < f_lo: return 0.
    
    # The "f is inside the triangular function" cases:
    if f > f_c:
        return np.power(1.-np.abs(f-f_c)/(f_hi-f_c),a)
    
    if f < f_c:
        return np.power(1.-np.abs(f-f_c)/(f_c-f_lo),a)


# An oversampled spectrogram is a bad choice for a stored version of pre-processed musical pieces, it would
# be to large and thus a lot of hard drive space would be wasted. Therefore - a version of spectrogram having
# only 88 freqiencies (associated with 88 piano key frequencies) is calculated. A triangular-like filter
# (with adjustable slope of "triangles" is used to calculate parameters associated with each key frequency.
# Pre-chromogram is an immediate result used in further steps of processing to calculate chromogram (chroma features).
def calculate_prechromogram(audio_signal, fs, frame_length, ovlp_factor=0.9, spectrum_ovsmpl = 32, filter_slope=3):
    
    # For this procedure, we do not calculate the whole spectrogram first, we do it on the fly, in
    # this way we can calculate the compact pre-processed pre-chromogram in a row-by-row manner and 
    # do not need a huge amounts of RAM to store the whole spectrogram in it.
    audio_frames  = split2frames(audio_signal, frame_length, int(frame_length*ovlp_factor))
    musical_freqs = get_musical_frequencies()
    output_data   = np.empty((len(musical_freqs), len(audio_frames)))
    
    # Calculation of a single pre-chromogram parameter - the loop has to be sped up with the
    # Numbas JIT compiler.
    @nb.jit
    def calculate_prechromogram_parameter(musical_freqs, superfft,f_lo, f_c, f_hi,fs,filter_slope):
        chromogram_param = 0
        for k, spec_pwr in enumerate(superfft):
            f = k/len(superfft)*(fs/2)
            chromogram_param += superfft[k]*triangular_membership(f_lo, f_c, f_hi, f, filter_slope)
        return chromogram_param
    
    # A function defined above is next used to calculate rows of a pre-chromogram (in a loop):
    i = 0
    print('computing pre-chromogram')
    for audio_frame in tqdm(audio_frames):
        
        # For each audio frame calculate the oversampled FFT-based amplitude spectrum:
        superfft          = ov_spectrum(audio_frame, spectrum_ovsmpl)
        
        # Define the output variable for a single row of a pre-chromogram.
        prechromogram_row = np.empty(len(musical_freqs))
        
        # Calculate the row employing the calculate_prechromogram_parameter function.
        for j,f_c in enumerate(musical_freqs):
            
            # Span of the triangular function starts a semitone below the central frequency...
            f_lo = f_c/np.power(2,1/12)
            
            # ... and end 1 semitone above it.
            f_hi = f_c*np.power(2,1/12)
            
            prechromogram_row[j] = calculate_prechromogram_parameter(musical_freqs, superfft,f_lo, f_c, f_hi,fs,filter_slope)
        
        # Update the structure keeping the whole pre-chromogram with a newly calculated 
        # pre-chromogram row.
        output_data[:,i] = prechromogram_row
        i += 1
        
    print()
    
    return output_data

# A procedure for obtaining frequencies of all keys from a piano keyboard.
def get_musical_frequencies(lowest_refnot_freq = 55, n_freqs=88):
    musical_notes_freqs = []
    for i in range(n_freqs):
        musical_notes_freqs.append(lowest_refnot_freq*np.power(2,i/12))
    return np.array(musical_notes_freqs)

# A procedure which converts the pre-chromogram into a chromogram (chroma-features).
# The chromogram is stored under the first 12 indices of the output, however values
# for greater key indices are not trimmed becaue of practical premises.
def compute_chromogram(prechromogram, musical_freqs):
    
    # Define the output array.
    output = np.zeros_like(prechromogram)
    
    # Calculation is performed by adding to the current value of 
    # pre-chromogram values of all octaves above it.
    for i_f in range(len(musical_freqs)):
        
        # Initialize the procedure by setting values for the first octave.
        j_f = i_f + 12
        output[i_f,:] = prechromogram[i_f,:]
        
        # Perform addition for higher octaves.
        while j_f < len(musical_freqs)-1:
            output[i_f,:] += prechromogram[j_f,:]
            j_f = j_f + 12
        
    return output

# To incentivize heuristics to focus more on the maximum value of the chromogram,
# a maximum value can be boosted by employing the procedure below.
def boost_maxval(array, boost_fctr):
    output = np.zeros_like(array)
    
    # For each row in an array find the maximum
    # and multiply it by boost_fctr.
    for i in range(array.shape[1]):
        output[:,i]    = array[:,i]
        amax           = np.argmax(output[:,i])
        output[amax,i]*=boost_fctr
    
    return output

# A function for extracting the maxima contour (thus - tracking the f0)
# from a chromogram.
def chroma2freqs(chromogram, musical_freqs):
    
    # Definition of the output variables.
    f    = np.empty(chromogram.shape[1])
    ampl = np.empty(chromogram.shape[1])
    
    # Estimation of frequencies and amplitudes of largest
    # maxima trajectory present in the input chromogram.
    for i in range(chromogram.shape[1]):
        idx     = np.argmax(chromogram[:,i])
        f[i]    = musical_freqs[idx]
        ampl[i] = chromogram[idx,i]
    
    return f, ampl

# A funciton used by the simulated annealing heuristic to find out how good is a given refined melody
# contour optimized by it.
@nb.jit
def calc_fitness(chromogram, melody_profile):
    # The heuristic used for fitness is as follows:
    # "The melody path should be as short as possible, but on the other hand,
    # it also should go through the points on the chromogram which have the greatest
    # amount of amplitude spectrum power".
    
    # We assess the length of a path by computing the mean distance between
    # consecutive frequencies (notes) on a melody path.
    melody_path_length = np.mean(np.abs(np.diff(melody_profile)))
    
    # We initialize a variable for storing the cumulative energy of all
    # points through which a melody path goes.
    cumulated_energy   = 0.
    
    # In the loop below, the fitness function based on the heuristic assumption
    # is being calculated:
    for i in range(chromogram.shape[1]):
        cumulated_energy += chromogram[melody_profile[i],i]
    
    # The fitness function is normalized by a path length, so the
    # function performs the same for both the long and short music pieces.
    melody_energy      = cumulated_energy/chromogram.shape[1]
    fitness            = melody_energy/(0.0001+2*melody_path_length)
    
    return fitness

# The heuristic algorithm for smoothing the melody path based on a heuristic provided in a calc_fitness function:
# https://en.wikipedia.org/wiki/Simulated_annealing
def melody_annealing(chromogram, num_iters   = 100, temperature = 0.0001, spread = None, k=1.2, epsilon=0.0000000001):
    
    # First, get all allowed frequencies.
    musical_freqs = get_musical_frequencies()
    
    # Initialize the frequency trajectory by tracking the largest maxima.
    melody_profile = np.argmax(chromogram,axis=0)
    
    # A temperature hyperparameter has to be at least equal to epsilon
    if temperature < epsilon: temperature = epsilon
    
    # Temperature profile for the annealing process (decreasing over time).
    temperature_v = np.power(np.linspace(1,0,num_iters),k)*(temperature-epsilon) + epsilon
    
    # Main heuristic algorithm loop
    print('Annealing the melodic line')
    pbar_temp = tqdm(temperature_v)
    
    # For each step of the algorithm (num_iters iterations will be carried out)...
    for i, inst_temp in enumerate(pbar_temp):
    
        # ...iterate over a single chromogram row.
        for n in range(chromogram.shape[1]):
            
            # Estimate a current value of fitness for current chromogram and melody profile. 
            current_fitness   = calc_fitness(chromogram, melody_profile)
            
            # Randomly choose a piano key index.
            random_idx = np.random.randint(0, len(musical_freqs)-1)
            
            # Replace the current (n-th) point on a melody trajectory
            # with a randomly chosen key index
            potential_mel_prof    = cp.copy(melody_profile)
            potential_mel_prof[n] = random_idx
            
            # Evaluate the fitness after the change made in a line above.
            potential_fitness = calc_fitness(chromogram, potential_mel_prof)
            
            # If this change improved the fitness - keep it!
            if potential_fitness > current_fitness:
                melody_profile = potential_mel_prof
            
            # If the change made things worse, do the following:
            else:
                # Calculate the change of fitness and calculate the annealing
                # statistic for the purpose of the test.
                delta       = current_fitness - potential_fitness
                anneal_stat = np.exp(-1*(delta/inst_temp))
                
                # Obtain a random number, if it is larger than an annealing`
                # statistic - keep the change, this may seem illogical, but 
                # keeping some "bad decisions" allows the algorithm to explore
                # solutions which can be reached only by keeping some "bad" step
                # steps first. This "keeping of bad states" is less and less probable
                # with the temperature of annealing constantly decresing, hence the
                # name of the method.
                rand_val = np.random.uniform()
                if anneal_stat>rand_val:
                    melody_profile = potential_mel_prof
        
        # Update information displayed on a progress bar.
        pbar_temp.set_description(f'temp.: {"%.10f"%inst_temp}, fit.: {"%.2f"%current_fitness}')
    
    return melody_profile


# Result of detection can be auralized, and this is a function which
# can be used in such a process. Sound can be synthesized from distinct
# harmonics, which are synthesized by a function below.
@nb.jit('float64[:](float64[:],float64[:], int64, float64)')
def synthesize_sine(freq_profile, ampl_profile, frame_length, fs):
    
    alpha = 0.99
    flt_inst_freq = freq_profile[0]
    flt_inst_ampl = ampl_profile[0]
    
    # We synthesize the signal in a step-by-step manner, and starting from
    # the phase signal. Initial phase is of course 0.
    phase_acc     = 0.
    
    # We also define the output variable and reserve the memory.
    output_signal = np.zeros(len(freq_profile)*frame_length, dtype=np.float64)
    
    # A variable for keeping the current index number.
    curr_idx = 0
    
    # Run the loop for consecutive values of instantaneous frequency, and amplitude.
    for inst_freq, inst_ampl in zip(freq_profile, ampl_profile):
        
        # Assume that frequency is last one if inst_freq is np.nan.
        if np.isnan(inst_freq):
            inst_freq = flt_inst_freq
        
        # For each single pair of frequency and amplitude values, a multitude of samples 
        # of the output has to be created, which is carried out in a loop below:
        for i in range(frame_length):
            
            # calculate the sine value based on current phase of a signal
            sine_sample = np.sin(phase_acc)
            
            # increment phase according to the current instantaneous frequency
            phase_increment = 2.*np.pi*flt_inst_freq/fs
            phase_acc += phase_increment
            
            flt_inst_freq = alpha*flt_inst_freq + (1-alpha)*inst_freq
            flt_inst_ampl = alpha*flt_inst_ampl + (1-alpha)*inst_ampl
            
            # The sample value under the index curr_idx is calculated!
            # We update the output array accordingly.
            output_signal[curr_idx] = sine_sample*flt_inst_ampl
            
            # Increment the index.
            curr_idx += 1
            
            # If the phase "overflows", it can be reduced to the range from 0 to pi.
            while phase_acc > 2.*np.pi:
                phase_acc -= 2.*np.pi
    
    return output_signal

# A function which performs musical notes tracking on the basis of the
# provided input chromogram.
def track_voice(chromogram, musical_freqs, annealing_iters=100, lowest_key=24, highest_key = 60, dmp_fctr  = 2):
    # As the detection can be carried out for a given set of musical notesm
    # some of them have to be discriminated and this is achieved by linear
    # weighting pattern, which is calculated below:
    freq_weighting                    = np.ones(88)
    freq_weighting[0:lowest_key]      = np.linspace(0.000001,1,lowest_key)
    freq_weighting[88-highest_key:88] = np.linspace(1,0.000001,highest_key)
    freq_weighting                    = np.power(freq_weighting,dmp_fctr)
    
    # Transcription is based upon a salience function derived from chroma features.
    # Example publication on this topic:
    # Justin Salamon, Emilia Gómez
    # A Chroma-based Salience Function for Melody and Bass Line Estimation From Music Audio Signals
    # https://doi.org/10.5281/zenodo.849573
    
    # Calculation of a salience function for a given chromogram:
    salience = np.empty_like(chromogram)
    for i in range(chromogram.shape[0]):
        salience[i,:] = chromogram[i,:]*chromogram[np.mod(i,12),:]*freq_weighting[i]
    salience = salience/np.max(chromogram)
    
    # Additional boost of maximum values of salience function:
    salience  = boost_maxval(salience, 2)
    
    # Application of simulated annealing heuristic to the melody obtained
    # from the salience tracking processing step:
    key_index = melody_annealing(salience, num_iters = annealing_iters)
    
    # Calculation of amplitude of the output notes:
    _, ampl   = chroma2freqs(chromogram, musical_freqs)
    ampl      = ampl/np.max(np.abs(ampl))
    
    return key_index, ampl

def synthesize_voice(vec_f0, ampl, frame_length, ovlp_factor, fs):
    # Synthesis of the output signal following the f0, which can be used
    # for auralization of the melody transcription result:
    time_increment  = int(frame_length*(1-ovlp_factor))
    out_signal  = synthesize_sine(vec_f0,   ampl,   time_increment, fs)
    out_signal += synthesize_sine(vec_f0*2, ampl/2, time_increment, fs)
    out_signal += synthesize_sine(vec_f0*4, ampl/3, time_increment, fs)
    out_signal += synthesize_sine(vec_f0*8, ampl/4, time_increment, fs)
    
    return out_signal

def draw_prechromogram(prechromogram, frame_length, ovlp_factor, spectrum_ovsmpl, fs):
    time_increment  = int(frame_length*(1-ovlp_factor))/fs
    
    t_start = 0
    t_stop  = prechromogram.shape[1]*time_increment
    
    plt.figure()
    plt.imshow(prechromogram, aspect='auto', origin='lower', extent=[t_start,t_stop,0,prechromogram.shape[0]], cmap='RdYlGn')
    cbar = plt.colorbar()
    cbar.set_label('prechromogram component power [-]')
    plt.xlabel('time [s]')
    plt.ylabel('piano key number [-]')
    plt.tight_layout()
    
# --------------------------------------------------------------------------- #
# Main menu commands definitions
# --------------------------------------------------------------------------- #

# A main menu procedure for customized or preset-driven exploratory analysis of
# oversampled input data spectra.
def cmd_show_oversampled_spectrograms(settings):
    input_data_dir      = settings['DIRECTORIES']['input_data_dir']
    tmp_data_dir        = settings['DIRECTORIES']['tmp_data_dir']
    spectr_floor        = settings['PROCESSING_PRESETS']['spectr_floor']
    
    if len(list(os.listdir(input_data_dir))) == 0:
        print('No files for processing were found.')
    
    choice_options  = [] 
    for fname in os.listdir(input_data_dir):
        fname_core,ext = os.path.splitext(fname)
        if ext in settings['PROCESSING_PRESETS']['allowed_input_ext']:
            choice_options.append(fname)
    
    fname           = ask_user_for_an_option_choice('Choose file for processing:', 'Item number: ',choice_options)
    print()
    t_start         = ask_user_for_a_float('Choose starting time of the visualized excerpt [s]: ')
    print()
    exc_duration    = ask_user_for_an_option_choice('Choose duration of the visualized excerpt [s]:', 'Excerpt duration: ',[15,30,45])
    print()
    scale_type      = ask_user_for_an_option_choice('Choose scale type:', 'Item number: ',['linear','logarithmic'])
    print()
    
    if ask_for_user_preference('Do you want to use default spectrogram calculation parameters?'):
        frame_length    = settings['PROCESSING_PRESETS']['frame_length']
        spectrum_ovsmpl = settings['PROCESSING_PRESETS']['spectrum_ovsmpl']
        ovlp_factor     = settings['PROCESSING_PRESETS']['ovlp_factor']
        window_type     = settings['PROCESSING_PRESETS']['window_type']
        kaiser_beta     = settings['PROCESSING_PRESETS']['kaiser_beta']
        print()
    else:
        frame_length    = ask_user_for_an_option_choice('Choose FFT frame length:', 'Item number: ',[256,512,1024,2048,4096,8192])
        print()
        spectrum_ovsmpl = ask_user_for_an_option_choice('Choose frequency oversampling factor:', 'Item number: ',[1,2,4,8,16,32])
        print()
        ovlp_factor     = ask_user_for_an_option_choice('Choose frames overlap factor:', 'Item number: ',[0.0,0.1,0.25,0.5, 0.75, 0.9])
        print()
        window_type     = ask_user_for_an_option_choice('Choose window type:', 'Item number: ',['rectangular','Kaiser','Hamming','Hann','Blackman'])
        print()
    
        kaiser_beta = None
        if window_type == 'Kaiser':
            print("Choose the beta parameter of the Kaiser window (14 is a good starting value)")
            kaiser_beta = ask_user_for_a_float('beta value [-]: ')
            print()
    
    t_stop       = t_start+exc_duration
    fname_core,ext = os.path.splitext(fname)
    
    print(f'reading file: {fname}')
    fs,orig_audio_data = read_file(tmp_data_dir, os.path.join(input_data_dir,fname))
    print()
    
    audio_duration = len(orig_audio_data)/fs
    if t_stop>audio_duration:
        t_start -= (t_stop-audio_duration)
        t_stop   = t_start+exc_duration
        print(f'The specified time range exceeds the audio signal duration, correcting the time range to {t_start} s - {t_stop} s.')
    
    n_start      = int(t_start*fs)
    n_stop       = int(t_stop*fs)
    
    audio_signal = stereo2mono(orig_audio_data).astype(np.float32)
    audio_signal = audio_signal[n_start:n_stop]
    
    print()
    sspect = oversampled_spectrogram(audio_signal, fs, frame_length,wnd_type=window_type, beta=kaiser_beta, ovlp_factor=ovlp_factor, spectrum_ovsmpl = spectrum_ovsmpl)
    
    sspect          = sspect+settings['PROCESSING_PRESETS']['spectr_floor']
    sspect          = 20*np.log10(sspect)
    
    plt.imshow(sspect, aspect='auto', origin='lower', extent=[t_start,t_stop,0,fs//2], cmap='RdYlGn')
    cbar = plt.colorbar()
    plt.title(f"{fname} ({t_start} s - {t_stop} s)")
    plt.xlabel('time [s]')
    plt.ylabel('frequency [Hz]')
    
    if   scale_type=='logarithmic':
        plt.gca().set_yscale('symlog')
        cbar.set_label('spectral component power [dB]')
    elif scale_type=='linear':
        cbar.set_label('spectral component power [-]')
    else:
        raise RuntimeError('Bad scale type was specified.')
    
    plt.tight_layout()
    plt.show()

# A pre-processing action which calculated pre-chromograms and stores them on the hard drive
# so they can be further processed by other main menu commands.
def cmd_generate_prechromograms(settings):
    DATA_DIR        = settings['DIRECTORIES']['input_data_dir']
    TMP_DIR         = settings['DIRECTORIES']['tmp_data_dir']
    FRAME_LENGTH    = settings['PROCESSING_PRESETS']['frame_length']
    ovlp_factor     = settings['PROCESSING_PRESETS']['ovlp_factor']
    spectrum_ovsmpl = settings['PROCESSING_PRESETS']['spectrum_ovsmpl']
    
    if len(list(os.listdir(DATA_DIR))) == 0:
        print('No files for processing were found.')
        
    for fname in os.listdir(DATA_DIR):
        _,ext = os.path.splitext(fname)
        if ext not in settings['PROCESSING_PRESETS']['allowed_input_ext']: continue
        print(f'reading file: {fname}')
        fs,orig_audio_data = read_file(TMP_DIR, os.path.join(DATA_DIR,fname))
        audio_signal       = stereo2mono(orig_audio_data).astype(np.float32)
        print()
        
        prechromogram = calculate_prechromogram(audio_signal, fs, FRAME_LENGTH, ovlp_factor, spectrum_ovsmpl)
        
        fname_core,_ = os.path.splitext(fname)
        
        saving_path = os.path.join(settings['DIRECTORIES']['prechromograms_data_dir'], fname_core)
        np.savez(saving_path,
            prechromogram=prechromogram, 
            frame_length=FRAME_LENGTH,
            ovlp_factor=ovlp_factor,
            spectrum_ovsmpl=spectrum_ovsmpl,
            fs=fs)

# A command for visualization of pre-chromograms obtained in a previous step.
def cmd_show_prechromogram(settings):
    prechromograms_data_dir = settings['DIRECTORIES']['prechromograms_data_dir']
    
    if len(list(os.listdir(prechromograms_data_dir))) == 0:
        print('No files for visualization were found.')
    
    choice_options  = [] 
    for fname in os.listdir(prechromograms_data_dir):
        fname_core,ext = os.path.splitext(fname)
        if ext in ['.npz']:
            choice_options.append(fname)
    
    fname = ask_user_for_an_option_choice('Choose file for visualization:', 'Item number: ',choice_options)
    print()
    
    fdata           = np.load(os.path.join(prechromograms_data_dir,fname))
    prechromogram   = fdata['prechromogram']
    frame_length    = fdata['frame_length']
    ovlp_factor     = fdata['ovlp_factor']
    spectrum_ovsmpl = fdata['spectrum_ovsmpl']
    fs              = fdata['fs']
    
    draw_prechromogram(prechromogram, frame_length, ovlp_factor, spectrum_ovsmpl, fs)
    plt.title(f"{fname}")
    plt.show()

# A command for visualization of pre-chromograms obtained in a previous step.
def cmd_show_melody_tracking_results(settings):
    melody_tracking_dir     = settings['DIRECTORIES']['melody_tracking_dir']
    prechromograms_data_dir = settings['DIRECTORIES']['prechromograms_data_dir']
    
    if len(list(os.listdir(melody_tracking_dir))) == 0:
        print('No files for visualization were found.')
    
    choice_options  = [] 
    for fname in os.listdir(melody_tracking_dir):
        fname_core,ext = os.path.splitext(fname)
        if ext in ['.npz']:
            choice_options.append(fname)
    
    fname = ask_user_for_an_option_choice('Choose file for visualization:', 'Item number: ',choice_options)
    print()
    
    fdata_tracking_results = np.load(os.path.join(melody_tracking_dir,fname), allow_pickle=True)
    fdata_prechromograms   = np.load(os.path.join(prechromograms_data_dir,str(fdata_tracking_results['source_fname'])))
    
    prechromogram   = fdata_prechromograms['prechromogram']
    tracked_voices  = fdata_tracking_results['tracked_voices']
    frame_length    = fdata_tracking_results['frame_length']
    ovlp_factor     = fdata_tracking_results['ovlp_factor']
    spectrum_ovsmpl = fdata_tracking_results['spectrum_ovsmpl']
    fs              = fdata_tracking_results['fs']
    
    draw_prechromogram(prechromogram, frame_length, ovlp_factor, spectrum_ovsmpl, fs)
    plt.title(f"{fname}")
    time_increment  = int(frame_length*(1-ovlp_factor))/fs
    
    for voice_dict in tracked_voices:
        freq_vec = voice_dict['key_index']
        t_vec    = np.linspace(0,prechromogram.shape[1]*time_increment,len(freq_vec))
        plt.plot(t_vec,freq_vec,label=voice_dict['name'],marker='o')
    
    plt.legend()
    plt.show()

# A procedure for detection of f0 and synthesis of the output preview (input signal with a musical
# transcription result synthesized and added to it).
def cmd_detect_f0(settings):
    prechromograms_data_dir = settings['DIRECTORIES']['prechromograms_data_dir']
    tmp_data_dir            = settings['DIRECTORIES']['tmp_data_dir']
    
    if len(list(os.listdir(prechromograms_data_dir))) == 0:
        print('No files for processing were found.')
    
    tracking_type   = ask_user_for_an_option_choice('Choose tracking type:', 'Item number: ',['mono','3-voiced'])
    print()
    
    annealing_iters = ask_user_for_an_option_choice('Choose number of annealing iterations:', 'Item number: ',[1,10,50,100,200])
    print()
    
    for fname in os.listdir(prechromograms_data_dir):
        INFILE = os.path.join(prechromograms_data_dir,fname)
        _, ext = os.path.splitext(fname)
        
        if ext not in ['.npz']:
            continue
        
        print(f'processing path: {fname}')
        print()
        
        fdata           = np.load(os.path.join(prechromograms_data_dir,fname))
        prechromogram   = fdata['prechromogram']
        frame_length    = int(fdata['frame_length'])
        ovlp_factor     = fdata['ovlp_factor']
        spectrum_ovsmpl = fdata['spectrum_ovsmpl']
        fs              = int(fdata['fs'])
        musical_freqs   = get_musical_frequencies()
        
        chromogram = compute_chromogram(prechromogram, musical_freqs)
        
        
        tracked_voices = []
        if tracking_type == 'mono':
            tracked_voices.append({'name':'mono','lowest_key':12,'highest_key':60})
        elif tracking_type == '3-voiced':
            tracked_voices.append({'name':'high','lowest_key':36,'highest_key':60})
            tracked_voices.append({'name':'mid', 'lowest_key':24,'highest_key':48})
            tracked_voices.append({'name':'low', 'lowest_key':12,'highest_key':36})
        
        for i, voice_dict in enumerate(tracked_voices):
            print(f'Tracking voice: {voice_dict["name"]}')
            key_index, amplitude = track_voice(chromogram, musical_freqs, annealing_iters, lowest_key=voice_dict["lowest_key"], highest_key = voice_dict["highest_key"])
            tracked_voices[i].update({'key_index':key_index,'amplitude':amplitude})
            print()
        
        output_structure = {}
        output_structure.update({'source_fname':fname})
        output_structure.update({'tracking_type':tracking_type})
        output_structure.update({'tracked_voices':tracked_voices})
        output_structure.update({'annealing_iters':annealing_iters})
        output_structure.update({'frame_length':frame_length})
        output_structure.update({'ovlp_factor':ovlp_factor})
        output_structure.update({'spectrum_ovsmpl':spectrum_ovsmpl})
        output_structure.update({'fs':fs})
        
        fname_core,_ = os.path.splitext(fname)
        saving_dir   = os.path.join(settings['DIRECTORIES']['melody_tracking_dir'], f"tracked_{fname_core}_{tracking_type}_{annealing_iters}_iters")
        
        np.savez(saving_dir, **output_structure)

def synthesize_tracking_results(settings):
    tmp_data_dir        = settings['DIRECTORIES']['tmp_data_dir']
    
    mix_with_originals = False
    amount_of_original = 0
    if ask_for_user_preference('Should the script mix the synthesized output with an original audio?'):
        print()
        mix_with_originals = True
        amount_of_original = ask_user_for_a_float('Desired ratio of the original signal to the synthesized one: ')
    print()
    
    musical_freqs = get_musical_frequencies()
    
    for fname in os.listdir(settings['DIRECTORIES']['melody_tracking_dir']):
        _, ext = os.path.splitext(fname)
        
        if ext not in ['.npz']:
            continue
        
        melody_tracking_dir    = os.path.join(settings['DIRECTORIES']['melody_tracking_dir'], fname)
        tracking_data          = np.load(melody_tracking_dir, allow_pickle=True)
        
        source_fname    = str(tracking_data['source_fname'])
        tracked_voices  = tracking_data['tracked_voices']
        frame_length    = int(tracking_data['frame_length'])
        ovlp_factor     = tracking_data['ovlp_factor']
        fs              = int(tracking_data['fs'])
        
        print(f"processing file: {fname}")
        if len(tracked_voices)==0:
            print("\tno voice tracking data found, aborting...")
            continue
        
        synthesized_signals = []
        for voice_dict in tracked_voices:    
            # Translation of key indices to frequencies:
            vec_f0             = musical_freqs[voice_dict['key_index']]
            synthesized_signal = synthesize_voice(vec_f0, voice_dict['amplitude'], frame_length, ovlp_factor, fs)
            synthesized_signals.append(synthesized_signal)
        
        out_signal  = np.mean(synthesized_signals, axis=0)
        out_ncore,_ = os.path.splitext(fname)
        
        if mix_with_originals:
            original_input_fpath = None
            fname_core,_         = os.path.splitext(source_fname)
            
            for potential_original_fname in os.listdir(settings['DIRECTORIES']['input_data_dir']):
                input_fncore, input_ext = os.path.splitext(potential_original_fname)
                if fname_core == input_fncore and input_ext in settings['PROCESSING_PRESETS']['allowed_input_ext']:
                    original_input_fpath = os.path.join(settings['DIRECTORIES']['input_data_dir'],potential_original_fname)
                    fs, orig_signal      = read_file(tmp_data_dir, original_input_fpath)
                    orig_signal          = stereo2mono(orig_signal).astype(np.float32)
                    orig_signal          = orig_signal/np.max(np.abs(orig_signal))
            
            common_length = np.min([len(out_signal), len(orig_signal)])
            out_signal    = out_signal[0:common_length]
            orig_signal   = orig_signal[0:common_length]
            
            if original_input_fpath is not None:
                out_signal = (out_signal/np.mean(np.power(out_signal,2)) + amount_of_original*orig_signal/np.mean(np.power(orig_signal,2)))/(1+amount_of_original)
        
        out_signal  = out_signal/np.max(out_signal)*0.4
        out_signal *= np.power(2,15)
        out_signal  = out_signal.astype(np.int16)
        
        fname_core,_ = os.path.splitext(fname)
        wavio.write(settings['DIRECTORIES']['music_output']+f'/synth_{fname_core}_{amount_of_original}_of_original.wav', fs, out_signal)
    print()

