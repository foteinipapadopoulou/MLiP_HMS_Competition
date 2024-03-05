import time
import pandas as pd
import numpy as np
import scipy.stats as stats
import sys

from statsmodels.tsa.stattools import acf, ccf, acovf

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
    Preprocess the eeg data for the time frame that has been labeled and create new features
    First we calculate the differences of the eeg readings and then calculate some statistics
    for these differences. The statistics we calculated will be the features we will use
    
    Returns a data frame with the eeg_id, the offset, the class and the new values we calculated
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
    preprocess = np.column_stack((statistics, counts, decor)).reshape(1,-1)

    return preprocess