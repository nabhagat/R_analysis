BMI_MAHI_MATLAB_files
=====================
Last update to README.md: 12-29-2014

This folder contains Matlab and R program files for analyzing data for TNSRE paper

Details of files included here are as follows: 

1. closedloopdata_analysis.R
	This file reads in the kinematic.txt and closedloop_results.mat files for preparing an array which includes information about each closedloop session and all its blocks. This file also needs information about classifier, which it gets from output created by conver_classifier_info_from_matlab_to_R.m file. 
	This file outputs the _cloop_statistics.csv and _cloop_eeg_epochs.mat files which are used for analyzing results. 

2. acc_err_regression.R - Calculates the slope of regression line for block TPR and FPR/min and checks for significance.

3. Analyzing Catch Trials_Nikunj.xlsx
	This MS-excel file includes details for each subjects such as how many block were tested, which blocks are valid and used for analyzing the results, etc.

4. acc_err_plots.m - creates figures for plotting BMI-performance and Behavioral-performance

5. convert_classifier_info_from_matlab_to_R.m - converts the classifier cell variables into arrays for so it can be read by R programs. 

6. import_markers_for_source_localization.m
	This file processes event markers for each closed-loop block and save it as block_X_event_markers.txt, which are later imported into EEGLAB dataset. The output EEGLAB dataset which combines all closed-loop blocks under each session will be later used for Source Localization analysis. 
	Additionally this file also calculates the latency between EEG decision and EMG decision and save it in _closeloop_eeg_emg_latencies.mat file. This data will be uused later on to plot histograms of EEG-EMG latencies.

7. Move_attempts_plot.m - Plot the number of failed movement attempts per block. Use of this file will be discontinued later on. 

8. Robotic_assessment_Ted.m - Plot the robotic assessment metrics. File provided by Ted. Will either be updated or discontinued later on. 
	


