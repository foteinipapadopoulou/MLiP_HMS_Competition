import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

from statsmodels.tsa.stattools import acf, ccf, acovf

def eeg_bandpass(eeg, min_freq, max_freq):
    """
    Applies a filter to the eeg signal to keep the (min_freq,max_freq) range of the signal
    
    Returns the filtered signal
    """
    #this function get signal and range of frequencies we interested and filter out every other frequency components
#     from signal
    
    # 10 is the order of the filter, 'lp'/'hp' stands for low-pass/high-pass filter that we are interested in 
#     fs is the frequency of the sample
    sos_lp = butter(10, max_freq, 'lp', fs=200, output='sos') #deletes the signals above max_freq
    sos_hp = butter(10, min_freq, 'hp', fs=200, output='sos') #deletes the signals below min_freq
    
    eeg_low = sosfilt(sos_lp, eeg) #low-pass filtering
    eeg_high = sosfilt(sos_hp, eeg_low) #high-pass filtering
    
    return eeg_high

def eeg2band(eeg, band='alpha'):
    """
    Passes the min and the max frequencies for a band to be used for filtering
    
    Returns the filtered eeg on the specific band
    """
    bands = {'alpha':(8, 12), 'beta':(12,30), 'gamma':(35,99), 'delta':(0.5,4), 'theta':(4,8)}
    band_range = bands[band]
    min_freq, max_freq = band_range[0], band_range[1]
    
    return eeg_bandpass(eeg, min_freq, max_freq)    

def calculate_channel_energies(eeg_channel):
    """
    Calculates the percentage of energy for the different bands of an eeg channel and adds the total energy of the channel in the end
    
    Returns an array with the channel's energy
    """
    bands = ['delta','theta' ,'alpha', 'beta', 'gamma']
    #Calculates the total energy of the channel
    squared = np.abs(eeg_channel)**2
    energy = np.sum(squared)
    
    energies = []
    for band in bands:
        #Filters the signal to a specific band
        eeg_band = eeg2band(eeg_channel,band)
        
        #Calculates the percentage of a band's energy
        percentage = np.sum(eeg_band)/energy
        energies.append(percentage)
        
    energies.append(energy)
    channel_energies = np.array([energies])

    return channel_energies

def calculate_eeg_energies(data_array):
    """
    Calculates the percentage of energies for all channels of an eeg
    
    Returns an array of the percentages of energies
    """
    eeg_energy = []
    
    #Iterate through every column
    for i in range(data_array.shape[1]):
        eeg_channel = data_array[:,i]
        channel_energies = calculate_channel_energies(eeg_channel)
        eeg_energy.append(channel_energies)
    
    eeg_energy_array = np.array([np.concatenate(eeg_energy).flat])

    return eeg_energy_array

def max_cross_corr(data_array):
    """ 
    Calculates the maximum cross-correlation coefficient (c_max) for every pair of channels of an eeg array
    
    Returns an array with the c_max for each pair
    """
    
    num_channels = data_array.shape[1]
    max_cross_values = []
    
    # Calculate autocorrelation for each channel
    autocorrs = [acf(data_array[:, i], adjusted=True)[0] for i in range(num_channels)]
    
    # Iterate over all combinations of two channels
    for i in range(num_channels):
        for j in range(i + 1, num_channels):  # Avoid repeating pairs
            autocorr1 = autocorrs[i]
            autocorr2 = autocorrs[j]
            cross_corr = ccf(data_array[:, i], data_array[:, j]) / (np.sqrt(autocorr1 * autocorr2))
            abs_cross_corr = np.abs(cross_corr)
            c_max = np.max(abs_cross_corr)
            max_cross_values.append(c_max)
    
    return np.array([max_cross_values])

def decorrelation_time(data_array):
    """ 
    Calculates the decorrelation time (ie first zero-crossing of autocorrelation function) in the columns of an array
    
    Returns an array with the decorrelation time of each column
    """
     
    # Initialize dictionary to store 
    decorrelation_times = {}

    # Iterate through the columns of the array
    for i in range(data_array.shape[1]):
        # Use acf to calculate the autocorrelation function
        autocorr = acf(data_array[:,i])
        
        #Find the indices that autocorrelation function changes signs
        zero_cross = np.where(np.diff(np.sign(autocorr)))[0]
        
        if len(zero_cross)!= 0:
            decorrelation_times[i] = np.min(zero_cross)
        else:
            decorrelation_times[i] = -1

    decorr = np.array([list(decorrelation_times.values())])
    return decorr

def count_sign_changes(data_array):
    """ 
    Calculates the number of zero-crossings in the columns of an array
    
    Returns an array with the amount of sign changes in each column
    """
    # Initialize dictionary to store sign change counts
    sign_changes_counts = {}
    
    # Iterate through the columns of the array
    for i in range(data_array.shape[1]):
        #Find the positions of any sign change and find the counts of them
        sign_changes =  len(np.where(np.diff(np.sign(data_array[:, i])))[0])
        sign_changes_counts[i] = sign_changes
    
    counts = np.array([list(sign_changes_counts.values())])
    return counts

def calc_eeg_statistics(data_array):
    """
    Calculates statistics for a sequence of eeg readings
    The statistics it calculates are the Mean, the Median, Variance, Standard Deviation, Skewness,
    Kurtosis and the Peak-to-Peak values
    
    Returns an array with the statistics calculated 
    """

    #Calculate the features for each column of the EEG

    mean = np.nanmean(data_array, axis=0)
    median = np.nanmedian(data_array, axis=0)
    var = np.nanvar(data_array, axis=0)
    std = np.nanstd(data_array, axis=0)
    skew = stats.skew(data_array, axis=0, nan_policy='omit')
    kurt = stats.kurtosis(data_array, axis=0, nan_policy='omit')
    pkpk = np.nanmax(data_array, axis=0) - np.nanmin(data_array, axis=0)
    area = np.abs(data_array).sum(axis = 0)
    
    #Stack the features of the EEG in new columns and name them accordingly
    features = np.column_stack((mean, median, var, std,skew, kurt,pkpk,area)).reshape(1,-1)

    return features

def preprocess_eeg(eeg):
    """
    Preprocess the eeg data and create new features
    First we calculate the differences of the eeg readings and then calculate some features for these differences.
    
    Returns an array with the new values we calculated
    """
    eeg = eeg.dropna()
    # Calculate the new features that will be the differences of the actual electrode readings
    eeg['Fp1 - F7'] = eeg['Fp1'] - eeg['F7']
    eeg['F7 - T3'] = eeg['F7'] - eeg['T3']
    eeg['T3 - T5'] = eeg['T3'] - eeg['T5']
    eeg['T5 - O1'] = eeg['T5'] - eeg['O1']

    eeg['Fp2 - F8'] = eeg['Fp2'] - eeg['F8']
    eeg['F8 - T4'] = eeg['F8'] - eeg['T4']
    eeg['T4 - T6'] = eeg['T4'] - eeg['T6']
    eeg['T6 - O2'] = eeg['T6'] - eeg['O2']

    eeg['Fp1 - F3'] = eeg['Fp1'] - eeg['F3']
    eeg['F3 - C3'] = eeg['F3'] - eeg['C3']
    eeg['C3 - P3'] = eeg['C3'] - eeg['P3']
    eeg['P3 - O1'] = eeg['P3'] - eeg['O1']

    eeg['Fp2 - F4'] = eeg['Fp2'] - eeg['F4']
    eeg['F4 - C4'] = eeg['F4'] - eeg['C4']
    eeg['C4 - P4'] = eeg['C4'] - eeg['P4']
    eeg['P4 - O2'] = eeg['P4'] - eeg['O2']
    
    eeg['Fz - Cz'] = eeg['Fz'] - eeg['Cz']
    eeg['Cz - Pz'] = eeg['Cz'] - eeg['Pz']

    # drop the single electrode and heart beat raedings because we will not use them
    eeg = eeg.drop(['Fp1','F3','F7','T3','T5','P3','C3','O1','Fp2','F4','F8','T4','P4','T6','C4','O2','Fz','Cz','Pz','EKG'], axis = 1)

    # Convert DataFrame to NumPy array
    data_array = eeg.to_numpy()
    
    # Calculate the statistics of the EEG
    statistics = calc_eeg_statistics(data_array)
    counts = count_sign_changes(data_array)
    decor = decorrelation_time(data_array)
    c_max = max_cross_corr(data_array)
    preprocess = np.column_stack((statistics, counts, decor,c_max)).reshape(1,-1)

    return preprocess