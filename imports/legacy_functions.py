# -------------------------------------------------------------------------
# Legacy Procedures from Melody Extractor Core Functions
# author: Adam Kurowski
# e-mail: akkurowski@gmail.com
# date:   23.07.2021
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# imports & presets

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavio
from imports.tmp_files import *
import numba as nb
from scipy.signal import butter, lfilter, spectrogram
from tqdm import tqdm
from collections import Counter
import librosa
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import seaborn as sns
import copy as cp
from imports.util_communication import *
from scipy.signal.windows import kaiser, hamming, hann, blackman

# -------------------------------------------------------------------------

def musical_quantize(historical_f0, musical_freqs):
    output = []
    
    for f in historical_f0:
        differences = np.abs(musical_freqs-f)
        best_idx = np.argmin(differences)
        output.append(musical_freqs[best_idx])
    
    output=np.array(output)
    return output

@nb.jit('float64[:,:](float64[:],float64[:],float64[:],int64,int64)')
def plot_scatter_heatmap_corefunc(vec_x, vec_y, vec_z, width, height):
    canvas    = np.zeros((width,height), dtype=np.float64)
    num_items = np.zeros((width,height), dtype=np.float64)

    min_x = np.min(vec_x)
    max_x = np.max(vec_x)
    min_y = np.min(vec_y)
    max_y = np.max(vec_y)
    
    n_samples = len(vec_x)
    
    for i in range(n_samples):
        increment_x_idx = int((vec_x[i] - min_x)/(max_x-min_x)*(width-1))
        increment_y_idx = int((vec_y[i] - min_y)/(max_y-min_y)*(height-1))
        
        canvas[increment_x_idx,increment_y_idx]    += vec_z[i]
        num_items[increment_x_idx,increment_y_idx] += 1
    canvas = canvas/num_items
    
    return canvas
def plot_scatter_heatmap(df, col_x, col_y, col_z, width, height):
    canvas = plot_scatter_heatmap_corefunc(df[col_x].to_numpy(), df[col_y].to_numpy(), df[col_z].to_numpy(), width, height)
    plt.imshow(canvas.T, origin='lower', aspect='auto', extent=[np.min(df[col_x]),np.max(df[col_x]),np.min(df[col_y]),np.max(df[col_y])])
    plt.colorbar()
    plt.xlabel(col_x)
    plt.ylabel(col_y)


def calculate_freq_difs(harmonic_freqs):
    diff_freqs = []
    num_freqs  = len(harmonic_freqs)
    for i in range(num_freqs):
        for j in range(num_freqs):
            if i<=j: continue
            f1 = harmonic_freqs[i]
            f2 = harmonic_freqs[j]
            f_diff = np.abs(f2-f1)
            diff_freqs.append(f_diff)
    return diff_freqs


def calculate_freq_dif_wght_mean(harmonic_freqs, maxima_powers):
    diff_freqs = []
    diff_ampls = []
    num_freqs  = len(harmonic_freqs)
    for i in range(num_freqs):
        for j in range(num_freqs):
            if i<=j: continue
            f1 = harmonic_freqs[i]
            a1 = maxima_powers[i]
            f2 = harmonic_freqs[j]
            a2 = maxima_powers[j]
            f_diff = np.abs(f2-f1)
            a_diff = np.abs(a2-a1)
            diff_freqs.append(f_diff)
            diff_ampls.append(a_diff)
    diff_ampls = np.array(diff_ampls)
    diff_ampls = np.power(diff_ampls,4)
    diff_ampls = diff_ampls/np.sum(diff_ampls)
    return np.sum(diff_freqs*diff_ampls)

def extract_peaks_in_frame(audio_frame):
    frame_spectr  = np.fft.fft(audio_frame)
    frame_spectr  = frame_spectr[0:FRAME_LENGTH//2]
    frame_spectr  = np.abs(frame_spectr)

    b_den, a_den = butter(8, 0.45, 'low')
    b_trn, a_trn = butter(8, 0.05, 'low')

    denoised_frame = lfilter(b_den,a_den,frame_spectr)
    trend_frame    = lfilter(b_trn,a_trn,frame_spectr)
    detrend_frame  = denoised_frame - trend_frame
    
    max_val         = np.max(detrend_frame)
    threshold       = 0.35*max_val
    detection_frame = detrend_frame>threshold
    detection_frame = np.diff(detection_frame, prepend=0)
    return detection_frame

def detect_f0(audio_frame):
    frame_peaks     = extract_peaks_in_frame(audio_frame)
    harmonic_freqs  = np.array(np.where(frame_peaks==1))[0]/FRAME_LENGTH/2*fs
    diff_freqs     = calculate_freq_difs(harmonic_freqs)
    diff_freqs     = np.array(diff_freqs)
    
    diff_freqs = diff_freqs[diff_freqs>80]
    diff_freqs = diff_freqs[diff_freqs<8000]
    
    if len(diff_freqs) > 0:
        detected_f0 = np.mean(diff_freqs)
    else:
        detected_f0 = 0
    
    return frame_peaks, detected_f0


def extract_freq_peaks(audio_data):
    audio_data      = split2frames(audio_data, FRAME_LENGTH, int(FRAME_LENGTH//2))
    
    f0_history = []
    for i in tqdm(range(audio_data.shape[1])):
        audio_frame   = audio_data[i,:]
        f0_freq = extract_peaks_in_frame(audio_frame)
        f0_freq[f0_freq==-1] = 0
        f0_history.append(f0_freq)
    
    plt.figure(sigsize=(16,9))
    f0_history = np.array(f0_history)
    plt.figure()
    plt.imshow(f0_history, origin='lower', cmap='Blues', aspect='auto')
    plt.colorbar()
    plt.savefig('contuours.png')
    
    plt.show()

def extract_f0_profile(audio_data):

    b,a = butter(5, [200/fs*2, 8000/fs*2], 'band')
    # audio_data = lfilter(b,a,audio_data)

    ovlp_length = int(FRAME_LENGTH/(4/3))
    # ovlp_length = 0
    audio_data  = split2frames(audio_data, FRAME_LENGTH, ovlp_length)
    
    print(audio_data.shape)
    
    print()
    print('tracking the f0')
    f0_history  = []
    frame_peaks_arr = []
    for i in tqdm(range(audio_data.shape[0])):
        audio_frame   = audio_data[i,:]
        frame_peaks, f0_freq = detect_f0(audio_frame)
        frame_peaks_arr.append(frame_peaks)
        f0_history.append(f0_freq)
    f0_history = np.array(f0_history)
    f0_history = exp_avg(f0_history,0.2)
    print()
    
    frame_peaks_arr = np.array(frame_peaks_arr).T
    frame_peaks_arr[frame_peaks_arr==-1]=0
    plt.figure()
    plt.imshow(frame_peaks_arr, origin='lower', cmap='Blues', aspect='auto')
    plt.colorbar()

    print()
    print('synthesizing output')
    output_signal = np.sign(synthesize_sine(f0_history, FRAME_LENGTH-ovlp_length, fs))

    output_signal = (output_signal + np.mean(orig_audio_data, axis=1)[:len(output_signal)]/5000)/100

    print()
    print('writing output file')
    wavio.write('outfile.wav', fs, output_signal.astype(np.float32))

    plt.figure()
    plt.plot(f0_history)
    plt.savefig('f0_plot.png')
    plt.show()

def find_peaks(insignal):
    spectr_1diff  = np.diff(insignal, prepend=insignal[0])
    spectr_2diff  = np.diff(spectr_1diff, prepend=spectr_1diff[0])
    
    sign_fnd_dif  = np.sign(spectr_1diff[0:-1])+np.sign(spectr_1diff[1:len(spectr_1diff)])
    
    max_idx_ind   = (sign_fnd_dif==0).astype(int)
    extrem_ind    = max_idx_ind*spectr_2diff[0:len(spectr_2diff)-1]
    
    if np.max(extrem_ind) != 0:
        extrem_ind   /=np.max(extrem_ind)
    
    maxpeak_ind = np.zeros_like(extrem_ind)
    maxpeak_ind[np.where(extrem_ind<0)] = insignal[np.where(extrem_ind<0)]
    
    maxima_indices = np.array(np.where(maxpeak_ind!=0))[0]
    maxima_powers = maxpeak_ind[maxima_indices]
    
    sort_order = np.argsort(maxima_powers)[::-1]
    maxima_indices = maxima_indices[sort_order]
    maxima_powers  = maxima_powers[sort_order]
    
    # num_harmonics = num_harmonics
    # if len(maxima_indices)>num_harmonics:
        # maxima_indices = maxima_indices[0:num_harmonics]
        # maxima_powers  = maxima_powers[0:num_harmonics]
    
    return maxima_indices, maxima_powers


def find_peak_freqs(audio_frame, spectrum_ovsmpl = 32):
    audio_frame_ov = np.zeros(len(audio_frame)*spectrum_ovsmpl)
    audio_frame_ov[0:len(audio_frame)] = audio_frame
    audio_frame_ov = np.kaiser(len(audio_frame_ov),0.1)*audio_frame_ov
    
    frame_spectr  = np.fft.fft(audio_frame_ov)
    frame_spectr  = frame_spectr[0:len(frame_spectr)//2]
    frame_spectr  = np.abs(frame_spectr)
    
    maxima_indices, maxima_powers = find_peaks(frame_spectr, 40)
    
    maxima_indices = maxima_indices/spectrum_ovsmpl
    
    return maxima_indices, maxima_powers


def median_filter(insignal, n):
    output = []
    for i in range(len(insignal)-n):
        output.append(np.median(insignal[i:i+n]))
    for i in range(n):
        output.append(insignal[len(insignal)-n+i])
    return np.array(output)

@nb.jit
def harmonics_to_musical_freqs(maxima_freqs, maxima_powers, musical_freqs):
    output = np.zeros(len(musical_freqs))
    
    max_idx = len(musical_freqs)-1
    for i, f_c in enumerate(musical_freqs):
        if i == 0:
            f_lo = musical_freqs[0]
            f_c  = musical_freqs[0]
            f_hi = musical_freqs[1]
        elif i == max_idx:
            f_lo = musical_freqs[max_idx-1]
            f_c  = musical_freqs[max_idx]
            f_hi = musical_freqs[max_idx]
        else:
            f_lo = musical_freqs[i-1]
            f_c  = musical_freqs[i]
            f_hi = musical_freqs[i+1]
        
        for max_freq, max_pwr in zip(maxima_freqs, maxima_powers):
            output[i] += triangular_membership(f_lo, f_c, f_hi, max_freq, 2)*max_pwr
    
    return output


def DBSCAN_proc():

    fundamental_period = []
    fundamental_power = []
    harmdata = []

    frame_number = 0
    for afrm in tqdm(audio_frames):
        maxima_indices, maxima_powers = find_peak_freqs(afrm)
        maxima_freqs = np.array(np.where(maxima_indices))[0]/len(maxima_indices)*fs/2
        
        peak_vec = np.zeros(16*FRAME_LENGTH)
        peak_vec[maxima_indices] = maxima_powers
        peak_vec = np.fft.fft(peak_vec)
        peak_vec = np.abs(peak_vec)
        peak_vec = peak_vec[0:len(peak_vec)//2]
        cepstral_peaks, ceps_pk_pwr = find_peaks(peak_vec, 5)
        
        if len(cepstral_peaks) > 0:
            idx = np.argmax(ceps_pk_pwr)
            harmdata.append({'x':frame_number,'y':cepstral_peaks[idx],'power':ceps_pk_pwr[idx]})
            fundamental_period.append(cepstral_peaks[idx]/fs)
            fundamental_power.append( ceps_pk_pwr[idx])
        
        frame_number += 1

    harmdata_df = pd.DataFrame(harmdata)
    norm_harmdata_df = cp.copy(harmdata_df)

    for colname in norm_harmdata_df.columns:
        vals = norm_harmdata_df.loc[:,colname]
        # norm_harmdata_df.loc[:,colname] = vals/np.max(np.abs(vals))
        norm_harmdata_df.loc[:,colname] = (vals-np.mean(vals))/np.std(vals)

    print(norm_harmdata_df)

    alg = DBSCAN(eps=0.4, min_samples=16, metric = 'euclidean',algorithm ='auto')
    # alg = AgglomerativeClustering(n_clusters=2)

    alg.fit(norm_harmdata_df)
    labels = alg.labels_
    harmdata_df = harmdata_df.assign(cluster_id = labels, linkage='complete')
    fundamental_period = np.array(fundamental_period)
    labels = np.array(labels)

    if False:
        for label in set(labels):
            if label == -1: continue
            x_vec = np.arange(len(fundamental_period))[labels==label]
            t_vec = x_vec/fs*(FRAME_LENGTH//2)
            plt.scatter(t_vec, fundamental_period[labels==label], label=label)
        plt.legend()
        plt.show()

    fundamental_period[labels==-1] = np.nan
    fundamental_period[fundamental_period>3*np.std(fundamental_period)] = np.nan

    fundamental_frequency = 1/fundamental_period

    print()
    print('synthesizing output')
    output_signal = np.sign(synthesize_sine(fundamental_frequency, FRAME_LENGTH-int(FRAME_LENGTH//2), fs))

    output_signal = (output_signal + np.mean(orig_audio_data, axis=1)[:len(output_signal)]/5000)/100

    print()
    print('writing output file')
    wavio.write('outfile.wav', fs, output_signal.astype(np.float32))


    plt.figure()
    plt.plot(fundamental_frequency)
    plt.show()

def calculate_full_presences(maxima_freqs, maxima_powers, musical_notes_freqs):
    output_vec = []
    for freq in musical_notes_freqs:
        output_vec.append(calculate_pitch_presence(maxima_freqs, maxima_powers, freq))
    return output_vec

@nb.jit
def calculate_pitch_presence(maxima_freqs, maxima_powers, pitch_freq):
    if len(maxima_freqs) == 0:
        return np.nan
    
    max_freq = 22000
    n_harmos = int(max_freq/pitch_freq)
    pitch_harmonics = np.zeros(n_harmos)
    for i in range(n_harmos):
        pitch_harmonics[i] = pitch_freq*(i+1)
    
    pharm_distances = np.zeros(len(pitch_harmonics))
    for j, pharm in enumerate(pitch_harmonics):
        
        harm_distances = np.zeros(len(maxima_freqs))
        i = 0
        for mfreq, mpower in zip(maxima_freqs, maxima_powers):
            harm_distances[i] = np.abs(pharm-mfreq)
            i += 1

        closest_idx       = np.argmin(harm_distances)
        closest_harmonics = maxima_freqs[closest_idx]
        closest_harm_pwr  = maxima_powers[closest_idx]
        
        pharm_distances[j] = closest_harm_pwr/np.abs(pharm-closest_harmonics)
    
    return np.sum(pharm_distances)


def melody_ga_fitting(h_chromogram, num_iters   = 50, pop_size=30, n_best=5, prob_mut_specimen=1, prob_mut_element=0.1, mut_distance=6):
    # initialization
    melody_profiles = []
    for i in range(pop_size):
        start_specimen       = np.argmax(h_chromogram,axis=0)
        
        if i > -1:
            modification_pattern = np.random.uniform(0,1,len(start_specimen))
            modification_pattern = modification_pattern<0.2
            
            mod_vals = np.random.randint(-2,2,h_chromogram.shape[1])
            
            start_specimen[modification_pattern] += mod_vals[modification_pattern]
            
            start_specimen[start_specimen<0] = 0
            start_specimen[start_specimen>h_chromogram.shape[0]-1] = h_chromogram.shape[0]-1
        
        melody_profiles.append(start_specimen)
    
    for i_iter in range(num_iters):
        # selection of best fitted
        best_profiles = []
        fitness_functions = np.empty(len(melody_profiles))
        
        for i, prof in enumerate(melody_profiles):
            fitness_functions[i] = calc_fitness(h_chromogram, prof)
        
        best_sort_order = fitness_functions.argsort()[::-1]
        
        print(i_iter,fitness_functions[0:n_best])
        
        for i in range(n_best):
            best_profiles.append(melody_profiles[best_sort_order[i]])
        
        # crossing-over
        cx_iter_num = 0
        child_profiles = []
        for cx_i in range(len(best_profiles)):
            for cx_j in range(len(best_profiles)):
                if cx_iter_num >= pop_size: break
                parent_A = best_profiles[cx_i]
                parent_B = best_profiles[cx_j]
                
                split_idx    = int(np.random.uniform()*(len(parent_A)-1))
                recep_parent = np.random.uniform()
                
                if recep_parent<0.5:
                    cx_new_profile = cp.copy(parent_A)
                    cx_new_profile[0:split_idx] = parent_B[0:split_idx]
                else:
                    cx_new_profile = cp.copy(parent_B)
                    cx_new_profile[0:split_idx] = parent_A[0:split_idx]
                
                child_profiles.append(cx_new_profile)
                cx_iter_num += 1
        
        # mutation of new specimens
        for new_spec in child_profiles:
            if np.random.uniform()<prob_mut_specimen:
                for mut_i in range(len(new_spec)):
                    if np.random.uniform()<prob_mut_element:
                        distance = np.random.randint(1,mut_distance)*(np.random.randint(0,1)*2-1)
                        new_spec[mut_i] += distance
                        
                        if new_spec[mut_i] < 0:
                            new_spec[mut_i] = 0
                        
                        if new_spec[mut_i] > h_chromogram.shape[0]-1:
                            new_spec[mut_i] = h_chromogram.shape[0]-1
        
        # creation of a new generation
        melody_profiles = best_profiles+child_profiles
    
    return melody_profiles[0]
    