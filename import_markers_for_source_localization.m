% Program to import time stamps (triggers) into raw EEG data and conver to EEGLAB dataset
% Author: Nikunj A. Bhagat, Graduate Student, University of Houston, 
% Contact: nbhagat08[at]gmail.com
% Date: December 16, 2014
%------------------------------------------------------------------------------------------------------------------------
% Revisions
% 02/20/15 - Resolved bugs in writing catch trials to event markers file
%                    - For LSGR_ses4, block 1 is skipped because missing
%                    Start_of_Experiment (S10) marker
%------------------------------------------------------------------------------------------------------------------------
clear;
eeglab; % Start EEGLAB
% Subject Details
Subject_name = 'LSGR';      %change1
closeloop_Sess_num = 5;     %change2
raster_plot_block_num = 5;  % only required if plotting raster plot
Cond_num = 3;
folder_path = ['F:\Nikunj_Data\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; 

%ERWS_ses4
%biceps_threshold = 50*ones(1,8);                  
%triceps_threshold = [12.5 12.5 12.5 12.5 500 12.5 12.5 12.5];

%ERWS_ses5
%biceps_threshold = 35*ones(1,5);                  
%triceps_threshold = 10*ones(1,5);

% %BNBO_ses4
%biceps_threshold = 25*ones(1,8);                  
%triceps_threshold = 25*ones(1,8);

% %BNBO_ses5
% biceps_threshold = 31*ones(1,8);                  
% triceps_threshold = 21*ones(1,8);


%PLSH_ses4
%biceps_threshold = [9 9 13 13 13 13 13 13];                  
%triceps_threshold = [9 9 9 9 9 9 9 9];

%PLSH_ses5
%biceps_threshold = [9 10 10 7 6 6 6 6];                  
%triceps_threshold = [5.5 7.5 7 7 6 6 6 6];

% %LSGR_ses4                                                    %change4
%biceps_threshold = [11 11 11 7 8 8 8 8];                  
%triceps_threshold = [6 6 8 8 6.5 6.5 6.5 6.5];

%LSGR_ses5
biceps_threshold = [8 8 8 10 8 8 8 8 8];                  
triceps_threshold = [6.5 6.5 6.5 8 7.2 7.2 7.2 7.2 7.2];

% Flags for selecting parts of code
import_event_markers_into_eeglab = 0; % must select estimate_eeg_emg_delays; both are dependent
estimate_eeg_emg_delays = 1;
create_raster_plot = 0;

% Fixed variables
Fs_eeg = 500; 
resamp_Fs = 500;            
EEG_channels_to_import = 1:64;
EMG_channels = [17 22 41 46];
if create_raster_plot == 0
    EEG_channels_to_import(EMG_channels) = [];   % Dont remove EMG channels for raster plot
end
Total_EEG_EMG_latency = [];
EEG_EMG_latency_calculated = [];
%EEG_EMG_latency_observed = [];     - Not accurate
EEG_detected_times = [];
EMG_detected_times = [];
%EMG_onset_times = [];
EEG_kinematic_latency = [];
Accurate_ind_EEG_Go = [];

% Load cloop_statistics.csv file 
if create_raster_plot == 1
    cl_ses_data = dlmread([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_block' num2str(raster_plot_block_num)...
        '_cloop_statistics.csv'],',',7,1); 
    %load([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_block' num2str(raster_plot_block_num)...
    %    '_cloop_kinematic_params.mat']);
    load(['F:\Nikunj_Data\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session2'  '\' ...
        Subject_name '_ses2_cond' num2str(Cond_num) '_block160_classifier_parameters.mat']);
    classifier_channels = channels;
    
    % Processing kinematic data
    kinematics_raw = dlmread([folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
                    '_block' num2str(raster_plot_block_num) '_closeloop_kinematics.txt'],'\t',15,1); % Raw data sampled at 200 Hz
    Fs_kin_raw = 1000;  % For MAHI, sampling at 1kHz

    % Correct data before initialization triggers. i.e. data segment before
    % simultaneous stimulus and response triggers are received.
    if find(kinematics_raw(:,17)==0,1,'first') == find(kinematics_raw(:,18)==0,1,'first')
       kinematics_raw(1:find(kinematics_raw(:,17)==0,1,'first'),:) = [];
       trig_correction = find(kinematics_raw(:,17)==5,1,'first');
       kinematics_raw(1:trig_correction,17:18) = 5;
    else
        error('No Initialization Triggers found. Check kinematics data');
    end

    % Set zero time
    t0_kin = kinematics_raw(1,1);
    kinematics_raw(:,1) = kinematics_raw(:,1)-t0_kin;

    % Downsample to 500 Hz; better than resample() 
    kinematics = downsample(kinematics_raw,2); % Decimation by 1/5
    %kinematics = [kinematics; ones(1000,size(kinematics,2))]; % Padding with ones to compensate for missing data
    Fs_kin = 500;      % Frequency

    % Low Pass Filter Position and Velocity
    position_f = []; velocity_f = [];
    [fnum,fden] = butter(4,4/(Fs_kin/2),'low');
    %freqz(fnum,fden,128,Fs_kin);
    position_f(:,1) = (180/pi).*filtfilt(fnum,fden,kinematics(:,2));      % Elbow position (deg)
    %position_f(:,2) = filtfilt(fnum,fden,kinematics(:,3));
    velocity_f(:,1) = (180/pi).*filtfilt(fnum,fden,kinematics(:,7));     % Elbow velocity (deg/s)
    %velocity_f(:,1) = (180/pi).*sgolayfilt(kinematics(:,7),3,51);
    %velocity_f(:,2) = filtfilt(fnum,fden,kinematics(:,5));
    %tan_velocity = sqrt(velocity_f(:,1).^2 + velocity_f(:,2).^2);  %InMotion
    %tan_velocity = velocity_f(:,1);
    %torque_f =  kinematics(:,12);
    
else
    cl_ses_data = dlmread([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_cloop_statistics.csv'],',',7,1); 
end
unique_blocks = unique(cl_ses_data(:,1));
EEGLAB_dataset_to_merge = [];


for m = 1:length(unique_blocks)
    if (strcmp(Subject_name,'LSGR') && (closeloop_Sess_num == 4)) 
        if (m == 1) || (m==3) || (m==4)           % LSGR_ses4
            continue
        end
    elseif (strcmp(Subject_name,'ERWS') && (closeloop_Sess_num == 5)) 
        if (m == 1) 
            continue
    end
    end
    closeloop_Block_num = unique_blocks(m);
    block_performance = cl_ses_data(cl_ses_data(:,1) == closeloop_Block_num,:);

    % Load closeloop_results.mat file
    load([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_block' num2str(closeloop_Block_num) '_closeloop_results.mat']); 
    marker_block = double(marker_block);
    % Load  Brain Vision (.eeg, .vhdr, .vmrk) files
    if closeloop_Block_num > 9
        EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop00' num2str(closeloop_Block_num) '.vhdr'], [], EEG_channels_to_import);
   else
        EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop000' num2str(closeloop_Block_num) '.vhdr'], [], EEG_channels_to_import);
   end
    
    EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(closeloop_Block_num) '_eeg_raw'];
    EEG = eeg_checkset( EEG );
    EEG=pop_chanedit(EEG, 'lookup','F:\Nikunj_Data\MATLAB_libraries\EEGLAB_13_1_1b\eeglab13_1_1b\plugins\dipfit2.2\standard_BESA\\standard-10-5-cap385.elp');
    EEG = eeg_checkset( EEG );
   
    %EEG = pop_saveset( EEG, 'filename',['BNBO_ses2_cond1_block' num2str(block_num) '_eeg_raw.set'],'filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
    %EEG = eeg_checkset( EEG );

    % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
         EEG_dataset_no = CURRENTSET; % Used later for retriving dataset in EEGLAB
    %    eeglab redraw;

    % Load event (trigger) times from EEGLAB
    eeg_response_trig = [];
    eeg_stimulus_trig = [];
    eeg_move_trig = [];
    eeg_start_stop_samples = [];
    move_trig_label = 'S 16';  % 'S 32'; %'S  8'; %'100';
    rest_trig_label = 'S  2';  % 'S  2'; %'200';
    target_reached_trig_label = 'S  8';
    block_start_recvd = false;
    block_stop_recvd = false;

    for j=1:length(EEG.event)
                        if (strcmp(EEG.event(j).type,target_reached_trig_label) || strcmp(EEG.event(j).type,'S 12'))
                            eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
                        elseif (strcmp(EEG.event(j).type,move_trig_label))
                            eeg_move_trig = [eeg_move_trig; EEG.event(j).latency];
                        elseif (strcmp(EEG.event(j).type,rest_trig_label))
                            eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
                        elseif (strcmp(EEG.event(j).type,'S 10')) 
                            eeg_start_stop_samples = [eeg_start_stop_samples; EEG.event(j).latency];
                            block_start_recvd = true;
                        elseif (strcmp(EEG.event(j).type,'S 26'))
                            eeg_start_stop_samples = [eeg_start_stop_samples; EEG.event(j).latency];
                            block_stop_recvd = true;
                        else
                            
                            % Do nothing
    %                         if initial_time_flag == 0
    %                             initial_time_flag = 1;
    %                             stimulus_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
    %                             response_time_correction = EEG.event(j).latency/Fs_eeg;  % Save value for later
    %                         else
    %                             eeg_response_trig = [eeg_response_trig; EEG.event(j).latency];
    %                             eeg_stimulus_trig = [eeg_stimulus_trig; EEG.event(j).latency];
    %                        end
                        end
    end
    
    % Appy correction if either block start or stop triggers are missing
    if ((block_start_recvd == false) && (block_stop_recvd == true))
        eeg_start_stop_samples %= [(eeg_start_stop_samples - diff(matlab_data_start_stop_samples))...
                                                      %      eeg_start_stop_samples];
        continue
    elseif ((block_start_recvd == true) && (block_stop_recvd == false))
        eeg_start_stop_samples %= [eeg_start_stop_samples ...
                                                       %   (eeg_start_stop_samples + diff(matlab_data_start_stop_samples))];
        continue
    end                                                                
        
    % Determine time difference between BrainVision and MATLAB data capture
    matlab_data_start_stop_samples = marker_block(marker_block(:,2) == 50,1);
    eeg_data_start_stop_samples = matlab_data_start_stop_samples - eeg_start_stop_samples;
    eeg_time_correction = min(eeg_data_start_stop_samples);     %sec, BNBO_ses4_block6 = 2.78, BNBO_ses5_block5 = 1.38
    marker_block(:,1) = marker_block(:,1) - eeg_time_correction;

    % Find time stamps of intent_detected
    ind100 = find(marker_block(:,2) == 100);    % Robot moves during valid and catch trials
    Intents_detected = find(block_performance(:,5) == 1);
    Intents_labels = ones(length(Intents_detected),1);
    catch_intent =  find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
    if ~isempty(catch_intent)
        for u = 1:length(catch_intent)
           Intents_labels(Intents_detected == catch_intent(u)) = 2;
        end
    end

    ind_move_onset = ind100(Intents_labels == 1);   
    ind_move_onset_catch = ind100(Intents_labels == 2);

    ind_EEG_EMG_Go = ind_move_onset - 1;                                          % EEG+EMG detected
    ind_EEG_Go = ind_move_onset - 2;                                                    % EEG detected 

    ind_EEG_EMG_Go_catch = ind_move_onset_catch - 1;                 % EEG+EMG detected
    ind_EEG_Go_catch = ind_move_onset_catch - 2;                            % EEG detected 

    target_reached = eeg_response_trig(Intents_labels == 1);
    target_reached_catch = eeg_response_trig(Intents_labels == 2);
    
    target_shown = eeg_stimulus_trig((block_performance(:,4) == 1));        % Must subtract eeg_start_stop_samples - but not over here since merging with EEGLAB
    target_shown_catch = eeg_stimulus_trig((block_performance(:,4) == 2));  % Must subtract eeg_start_stop_samples
    target_location = block_performance(:,18);  % DO NOT subtract eeg_start_stop_samples
      
    %EEG_EMG_latency_observed = [EEG_EMG_latency_observed (marker_block(ind_EEG_EMG_Go,1)' - marker_block(ind_EEG_Go,1)')/Fs_eeg];
    %EEG_EMG_latency_observed = [EEG_EMG_latency_observed (marker_block(ind100-1,1)' - marker_block(ind100-2,1)')/Fs_eeg];
    
    if create_raster_plot == 1
        raw_eeg = EEG.data;
        raw_eeg_Fs = EEG.srate; 
        
        % spectral and spatial filtering of eeg
        hpfc = 0.1; lpfc = 1;
        [num_hpf,den_hpf] = butter(4,(hpfc/(raw_eeg_Fs/2)),'high');
        [num_lpf,den_lpf] = butter(4,(lpfc/(raw_eeg_Fs/2)));            % IIR Filter
        HPFred_eeg = filter(num_hpf,den_hpf,double(raw_eeg),[],2);             % filtering with phase delay 
        SPFred_eeg = HPFred_eeg;
        SPFred_eeg = spatial_filter(SPFred_eeg,'LLAP',[]);
        LPFred_eeg = filter(num_lpf,den_lpf,SPFred_eeg,[],2);           % filtering with phase delay
        EEG_rastor = LPFred_eeg(classifier_channels,eeg_start_stop_samples(1):eeg_start_stop_samples(2));
    end
    
    
       % Use main file for extracting EMG channels
       % [EEG, com] = pop_loadbv(path, hdrfile, srange, chans);
       if closeloop_Block_num > 9
            EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop00' num2str(closeloop_Block_num) '.vhdr'],...
                [], EMG_channels);
       else
            EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop000' num2str(closeloop_Block_num) '.vhdr'],...
                [], EMG_channels);
       end
        EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block' num2str(closeloop_Block_num) '_emg_raw'];
        EEG = eeg_checkset( EEG );
        %EEG = pop_saveset( EEG, 'filename',['BNBO_ses2_cond1_block' num2str(block_num) '_emg_raw.set'],'filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
        %EEG = eeg_checkset( EEG );
        % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        EMG_dataset_no = CURRENTSET;
        
        % Process EMG data
        emg_Fs = EEG.srate;
        raw_emg = EEG.data;
        raw_emg_t = 0:1/emg_Fs: (length(raw_emg) - 1)/emg_Fs;
        BPFred_emg = [];
        Diff_emg = [];
        EMG_rms = [];
        
        %1. 4th order Butterworth Band-pass Filter (30Hz  - 200Hz)
        emgbpfc = [30 200];     % Cutoff frequency = 15 - 48 Hz
        [num_bpf,den_bpf] = butter(4,[emgbpfc(1)/(emg_Fs/2) emgbpfc(2)/(emg_Fs/2)]);
        for i = 1:size(raw_emg,1)
            BPFred_emg(i,:) = filter(num_bpf,den_bpf,double(raw_emg(i,:)));
        end
    
        Diff_emg(1,:) = BPFred_emg(1,:) - BPFred_emg(3,:);
        Diff_emg(2,:) = BPFred_emg(2,:) - BPFred_emg(4,:);
        EMG_rms(1,:) = sqrt(smooth(Diff_emg(1,:).^2,150));
        EMG_rms(2,:) = sqrt(smooth(Diff_emg(2,:).^2,150));
        
        EMG_rms_resampled = EMG_rms(:,eeg_start_stop_samples(1):eeg_start_stop_samples(2)); %resample(EMG_rms',resamp_Fs,emg_Fs)';       % redundant bcoz resamp_Fs = emg_Fs
        emg_time_vector = 0:1/resamp_Fs:(length(EMG_rms_resampled)-1)/resamp_Fs;
        emg_time_vector = round(emg_time_vector.*500)/500;
        move_counts_time_vector = 0:1/20:(length(move_counts)-1)/20;     
        
%         if length(EMG_rms_resampled) ~= length(move_counts)
%             EMG_rms_corrected = interp1(emg_time_vector, EMG_rms_resampled', move_counts_time_vector,'pchip')';
%         else
%             EMG_rms_corrected = EMG_rms_resampled;
%         end

        if length(EMG_rms_resampled) ~= length(move_counts)
            %EMG_rms_corrected = interp1(emg_time_vector, EMG_rms_resampled', move_counts_time_vector,'pchip')';
            move_counts_upsampled = floor(interp1(move_counts_time_vector,move_counts,emg_time_vector,'pchip'));
            move_counts_upsampled(move_counts_upsampled <= 0) = 0;
            cnts_threshold_upsampled = interp1(move_counts_time_vector,all_cloop_cnts_threshold,emg_time_vector,'pchip');
            EMG_rms_corrected = EMG_rms_resampled;
        else
            EMG_rms_corrected = EMG_rms_resampled;
        end

        %cnts_threshold_upsampled(cnts_threshold_upsampled < 2) = 2;
        above_threshold_move_counts = ((cnts_threshold_upsampled - move_counts_upsampled) <= 0);
        %above_threshold_move_counts_index = find(above_threshold_move_counts);
        indexA = find(above_threshold_move_counts == 1);            % length(indexA) = length(indexB)
        indexB = find(marker_block(:,2) == 300);                                 % 
        
        if length(indexA) < length(indexB)
            indexB = indexB(1:length(indexA));
        elseif length(indexA) > length(indexB)
            indexA = indexA(1:length(indexB));
        end
            
        indexA(indexB > max(ind100)) = [];
        indexB(indexB > max(ind100)) = [];                                            % Remove 300 markers beyond 100 markers
       
         first_sample = zeros(length(indexA),1);
         eeg_emg_latency = zeros(length(indexA),1);
        for j = 1:length(indexA)
            sample_num = indexA(j);
            biceps_sample_num = find(EMG_rms_corrected(1,sample_num:end) >= biceps_threshold(m),1);
            triceps_sample_num = find(EMG_rms_corrected(2,sample_num:end) >= triceps_threshold(m),1);
            
            if isempty(biceps_sample_num) && isempty(triceps_sample_num)
                first_sample(j) = 7501; % Timeout = 15 sec
            elseif isempty(triceps_sample_num)
                % if b is empty, return a
                first_sample(j) = biceps_sample_num;
            elseif isempty(biceps_sample_num)
                % if a is empty, return b
                first_sample(j) = triceps_sample_num;
            else
                % if a and b are non-empty, return the min(a,b)
                first_sample(j) = min(biceps_sample_num,triceps_sample_num);
            end
            eeg_emg_latency(j) =  (first_sample(j) - 1)/resamp_Fs;
        end
        
        detected_move_counts_indexA = indexA(eeg_emg_latency <= 1); 
        % detected_move_counts_index = block_performance(Intents_detected,14);
        
        detected_move_counts_indexB = indexB(eeg_emg_latency <= 1);
        original_latencies = eeg_emg_latency(eeg_emg_latency <= 1);
        
        % group together  events
        combined_latencies = zeros(length(ind100),1);
        accurate_ind_EEG_Go = zeros(length(ind100),1);
        eeg_detected_times = zeros(length(ind100),1);
        
        for h = 1:length(ind100)
            if h == 1
                ind400 = find(marker_block(1:ind100(h),2) == 400);
                if length(ind400) > 1
                    neighbor_events = find((detected_move_counts_indexB >= ind400(end-1)) & (detected_move_counts_indexB < ind100(h)));
                else              
                    neighbor_events = find(detected_move_counts_indexB < ind100(h));                    
                end
            else
                ind400 = find(marker_block(ind100(h-1):ind100(h),2) == 400);
                if length(ind400) > 1
                    neighbor_events = find((detected_move_counts_indexB >= (ind100(h-1) + ind400(end-1))) & (detected_move_counts_indexB < ind100(h)));
                else              
                    neighbor_events = find((detected_move_counts_indexB > ind100(h-1)) & (detected_move_counts_indexB < ind100(h)));
                end
            end
            if ~isempty(neighbor_events)
                event_ind = neighbor_events((marker_block(ind100(h)-1,1)' - marker_block(detected_move_counts_indexB(neighbor_events),1)') <= 2*resamp_Fs);
                if isempty(event_ind)
                    event_ind = neighbor_events(end);
                end
               combined_latencies(h) = original_latencies(event_ind(1));       % Use first occuring event instead of average
               accurate_ind_EEG_Go(h) =  detected_move_counts_indexB(event_ind(1));
               eeg_detected_times(h) = detected_move_counts_indexA(event_ind(1));
            else
                accurate_ind_EEG_Go(h) = ind100(h)-2;          % Just to have some value instead of NaN---check?
                combined_latencies(h) = NaN;
                eeg_detected_times(h) = NaN;
            end
            %disp(neighbor_events')
        end
        
        detected_above_threshold_move_counts = zeros(1,length(above_threshold_move_counts));
        finite_eeg_detected_times = eeg_detected_times(~isnan(eeg_detected_times));
       detected_above_threshold_move_counts(finite_eeg_detected_times) = 1;
        
        % Use kinematic_onset time - 5/7/15
        finite_Intent_detections = Intents_detected(~isnan(eeg_detected_times));
        Kinematic_onset_times = round(block_performance(finite_Intent_detections,19)./2);
        detected_kinematic_move = zeros(1,length(above_threshold_move_counts));
        detected_kinematic_move(Kinematic_onset_times) = 1;

        target_shown_from_kin = round(block_performance(:,2)/2);
        Target_trace = zeros(size(emg_time_vector));
        for loc = 1:length(target_shown_from_kin)-1
           Target_trace(target_shown_from_kin(loc):target_shown_from_kin(loc+1)) = block_performance(loc,18); 
        end
        Target_trace(target_shown_from_kin(end):end) = block_performance(end,18); 
        
        % Code by Andrew to detect EMG onset
% %         Fs = resamp_Fs;
% %         firstTrial = 1;
% %         nTrials = length(finite_eeg_detected_times);
% %         onsetTimes = zeros(nTrials,1);
                
        trialFig = figure('Position',[100 1300 1000 500]);
        hold on;
        plot(emg_time_vector,EMG_rms_corrected','LineWidth',0.5);
        %plot(emg_time_vector,100.*above_threshold_move_counts,'k','LineWidth',0.5);
        plot(emg_time_vector,100.*detected_above_threshold_move_counts,'k','LineWidth',0.5);
        %plot(emg_time_vector,50.*Target_trace,'Color','m','LineWidth',0.5);
        plot(emg_time_vector,100.*detected_kinematic_move,'m','LineWidth',0.5);
        
%% 
% %         for trial = firstTrial:nTrials
% % %             oneForceTrial = forceTrials{trial}(:,3);
% % %             oneForceTrial = lowPassFilter(oneForceTrial,8,50/(Fs/2));
% % %             oneForceTrial = -oneForceTrial;
% %             oneEMGTrial = EMG_rms_corrected(:,finite_eeg_detected_times(trial)- (3*resamp_Fs):finite_eeg_detected_times(trial)+(3*resamp_Fs)); % +/- 3 sec. from intent detection 
% %             %t = 0:1/Fs:length(oneEMGTrial)/Fs - 1/Fs;
% %             t = emg_time_vector(finite_eeg_detected_times(trial)- (3*resamp_Fs):finite_eeg_detected_times(trial)+(3*resamp_Fs));
% %             baselineEMG = mean(oneEMGTrial(:,1:200)')';
% %             
% %             %plot(t,oneEMGTrial);
% %             %hold on;
% %             emg_thresholds = 0.1.*(max(oneEMGTrial')');
% %             %onsetIDX = find(oneForceTrial>baselineForce+OnsetOffsetThreshold,1,'first');
% %             bicep_onset = find(oneEMGTrial(1,:) > baselineEMG(1) + emg_thresholds(1),1,'first');
% %             if isempty(bicep_onset)
% %                 bicep_onset = 1;
% %             end
% %             tricep_onset = find(oneEMGTrial(2,:) > baselineEMG(2) + emg_thresholds(2),1,'first');
% %             if isempty(tricep_onset)
% %                 tricep_onset = 1;
% %             end
% %             
% %             % Using with biceps or triceps for determining movement onset
% % %             onsetIDX = min(bicep_onset,tricep_onset);
% % %             if onsetIDX == bicep_onset
% % %                 onsetDot = plot(t(onsetIDX),oneEMGTrial(1,onsetIDX),'r.','MarkerSize',20);
% % %             elseif onsetIDX == tricep_onset
% % %                 onsetDot = plot(t(onsetIDX),oneEMGTrial(2,onsetIDX),'r.','MarkerSize',20);
% % %             end
% % 
% %             % Using target information to select biceps or triceps muscles for determining movement onset
% %             if Target_trace(finite_eeg_detected_times(trial)) == 3 % Moving Up - Elbow flexion
% %                 onsetIDX = bicep_onset;
% %                 plot(t,oneEMGTrial(1,:),'b','LineWidth',2);
% %                 onsetDot = plot(t(onsetIDX),oneEMGTrial(1,onsetIDX),'r.','MarkerSize',20);
% %             elseif Target_trace(finite_eeg_detected_times(trial)) == 1 % Moving Down - Elbow extension
% %                 onsetIDX = tricep_onset;
% %                 plot(t,oneEMGTrial(2,:),'g','LineWidth',2);
% %                 onsetDot = plot(t(onsetIDX),oneEMGTrial(2,onsetIDX),'r.','MarkerSize',20);
% %             else
% %                disp('Error: Target infomation not available');
% %                onsetIDX = min(bicep_onset,tricep_onset);
% %             end
% %             onsetTimes(trial) = t(onsetIDX);
% %             axis([t(1)-2 t(end)+2 0 (max(max(oneEMGTrial'))+50)]);
% %             title(['Block ' num2str(closeloop_Block_num) ', Trial ' int2str(trial) ' of ' num2str(nTrials) '. Is the onset ok?']);    
% %             someResponse = input('Is the onset ok? \n', 's');
% % 
% %             if strcmp(someResponse,'q');
% %                 disp(['You were on trial' int2str(trial)]);
% %                 break;
% %             end
% % 
% %             while strcmp(someResponse,'n')
% %                 disp(['select data range on the graph and press any key!']);
% %                 %title(['select data on the graph for the onset']);                              
% %                 data_range = ginput(2);
% %                 if data_range(1,1) > data_range(2,1)
% %                     disp('first time point should occur earlier than the second!!');
% %                     continue;
% %                 else
% %                     oneEMGTrial = EMG_rms_corrected(:,find(emg_time_vector == round(data_range(1,1).*500)/500,1,'first'):...
% %                         find(emg_time_vector == round(data_range(2,1).*500)/500,1,'first'));
% %                     t = emg_time_vector(find(emg_time_vector == round(data_range(1,1).*500)/500,1,'first'):...
% %                         find(emg_time_vector == round(data_range(2,1).*500)/500,1,'first'));
% %                     baselineEMG = mean(oneEMGTrial(:,1:200)')';
% %                     emg_thresholds = 0.1.*(max(oneEMGTrial')');
% %                     bicep_onset = find(oneEMGTrial(1,:) > baselineEMG(1) + emg_thresholds(1),1,'first');
% %                     tricep_onset = find(oneEMGTrial(2,:) > baselineEMG(2) + emg_thresholds(2),1,'first');
% %                     
% %                     % Using with biceps or triceps for determining movement onset
% % %                     onsetIDX = min(bicep_onset,tricep_onset);
% % %                     if onsetIDX == bicep_onset
% % %                         onsetDot = plot(t(onsetIDX),oneEMGTrial(1,onsetIDX),'g.','MarkerSize',20);
% % %                     elseif onsetIDX == tricep_onset
% % %                         onsetDot = plot(t(onsetIDX),oneEMGTrial(2,onsetIDX),'g.','MarkerSize',20);
% % %                     end
% % 
% %                     % Using target information to select biceps or triceps muscles for determining movement onset
% %                     if Target_trace(finite_eeg_detected_times(trial)) == 3 % Moving Up - Elbow flexion
% %                         onsetIDX = bicep_onset;
% %                         plot(t,oneEMGTrial(1,:),'Color',[0.5 0.5 0.5],'LineWidth',2);
% %                         onsetDot = plot(t(onsetIDX),oneEMGTrial(1,onsetIDX),'g.','MarkerSize',20);
% %                     elseif Target_trace(finite_eeg_detected_times(trial)) == 1 % Moving Down - Elbow extension
% %                         onsetIDX = tricep_onset;
% %                         plot(t,oneEMGTrial(2,:),'Color',[0.5 0.5 0.5],'LineWidth',2);
% %                         onsetDot = plot(t(onsetIDX),oneEMGTrial(2,onsetIDX),'g.','MarkerSize',20);
% %                     else
% %                        disp('Error: Target infomation not available');
% %                        onsetIDX = min(bicep_onset,tricep_onset);
% %                     end
% %             
% %                     onsetTimes(trial) = t(onsetIDX);
% %                     title(['Block ' num2str(closeloop_Block_num) ', Trial ' int2str(trial) ' of ' num2str(nTrials) '. Is the onset ok?']);    
% %                     someResponse = input('Is the onset ok? \n', 's');
% % 
% %                     if strcmp(someResponse,'q');
% %                         disp(['You were on trial' int2str(trial)]);
% %                         break;
% %                     end
% %                 end
% %             end           
% %         end
% %         title(['Block ' num2str(closeloop_Block_num) ' done! yeeee!']);            
% %         EMG_onset_times = [EMG_onset_times; onsetTimes];

%%               
        EEG_kinematic_latency = [EEG_kinematic_latency; (emg_time_vector(finite_eeg_detected_times) - emg_time_vector(Kinematic_onset_times))'];
        
        %figure; hold on;
        %plot(emg_time_vector,EMG_rms_corrected');
        %plot(emg_time_vector,100.*above_threshold_move_counts,'k','LineWidth',1);
        %plot(emg_time_vector,100.*detected_above_threshold_move_counts,'or','LineWidth',2);
        %line([emg_time_vector(1) emg_time_vector(end)],[biceps_threshold(m) biceps_threshold(m)],'Color','b',...
        %    'LineWidth',1.5','LineStyle','--');
        %line([emg_time_vector(1) emg_time_vector(end)],[triceps_threshold(m) triceps_threshold(m)],'Color','g',...
        %    'LineWidth',1.5','LineStyle','--');     
        
        eeg_detected_times = eeg_detected_times + eeg_start_stop_samples(1);
        emg_detected_times = eeg_detected_times + (combined_latencies.*resamp_Fs);
            
        EEG_detected_times = [EEG_detected_times; eeg_detected_times];
        EMG_detected_times = [EMG_detected_times; emg_detected_times];
        EEG_EMG_latency_calculated = [EEG_EMG_latency_calculated; combined_latencies];
        Accurate_ind_EEG_Go = [Accurate_ind_EEG_Go; accurate_ind_EEG_Go];
        Total_EEG_EMG_latency = [Total_EEG_EMG_latency; eeg_emg_latency];
        
       
        
%         eeg_detected_times = detected_move_counts_index(Intents_labels == 1)./resamp_Fs + eeg_start_stop_samples(1)/Fs_eeg;
%         emg_detected_times = eeg_detected_times + EEG_EMG_latency_calculated(Intents_labels == 1);
%         
%         catch_eeg_detected_times = detected_move_counts_index(Intents_labels == 2)./resamp_Fs + eeg_start_stop_samples(1)/Fs_eeg;
%         catch_emg_detected_times = catch_eeg_detected_times + EEG_EMG_latency_calculated(Intents_labels == 2);
          %  [[ind_EEG_Go; ind_EEG_Go_catch] accurate_ind_EEG_Go ind100]
          %  combined_latencies'
    
    
     if import_event_markers_into_eeglab == 1
        % markers = ([[marker_block(ind_move_onset,2)                     marker_block(ind_move_onset,1)];
        %                       [101*ones(length(ind_move_onset_catch),1) marker_block(ind_move_onset_catch,1)];  
        %                       [marker_block(ind_EEG_Go,2) marker_block(ind_EEG_Go,1)];
        %                       [301*ones(length(ind_EEG_Go_catch),1) marker_block(ind_EEG_Go_catch,1)];  
        %                       [marker_block(ind_EEG_EMG_Go,2) marker_block(ind_EEG_EMG_Go,1)];
        %                       [401*ones(length(ind_EEG_EMG_Go_catch),1) marker_block(ind_EEG_EMG_Go_catch,1)]]);  
        % dlmwrite('BNBO_s4b2_markers.txt',markers,'delimiter','\t','precision','%.4f');

        marker_file_id = fopen([folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
            '_block' num2str(closeloop_Block_num) '_event_markers.txt'],'w');

        for i = 1:length(ind_move_onset)
            fprintf(marker_file_id,'EEG-GO-%d \t %d \n',m,marker_block(accurate_ind_EEG_Go(i),1));                         % Not-Accurate - To be replaced 
            fprintf(marker_file_id,'EMG-GO-%d \t %d \n',m,marker_block(ind_EEG_EMG_Go(i),1));            % Not-Accurate - To be replaced
            %fprintf(marker_file_id,'EEG-GO \t %6.3f \n',eeg_detected_times(i)); % More accurate, times occur after move_onset
            %fprintf(marker_file_id,'EMG-GO \t %6.3f \n',emg_detected_times(i)); % More accurate, times occur after move_onset
            fprintf(marker_file_id,'Move-Onset-%d \t %d \n',m,marker_block(ind_move_onset(i),1));         % Accurate
            fprintf(marker_file_id,'Target-Hit \t %d \n',target_reached(i));                                               % Accurate      
        end

        catch_indexes =  accurate_ind_EEG_Go(Intents_labels == 2);
        for i = 1:length(ind_move_onset_catch) % Bug found - multiple line prints
            %fprintf(marker_file_id,'catch-EEG-GO \t %d \n',marker_block(ind_EEG_Go_catch(i),1));                         % Not-Accurate - To be replaced 
            fprintf(marker_file_id,'catch-EEG-GO-%d \t %d \n',m,marker_block(catch_indexes(i),1));                          
            fprintf(marker_file_id,'catch-EMG-GO-%d \t %d \n',m,marker_block(ind_EEG_EMG_Go_catch(i),1));            
            %fprintf(marker_file_id,'catch-EEG-GO \t %6.3f \n',catch_eeg_detected_times(i));               % More accurate
            %fprintf(marker_file_id,'catch-EMG-GO \t %6.3f \n',catch_emg_detected_times(i));            % More accurate
            fprintf(marker_file_id,'catch-Move-Onset-%d \t %d \n',m,marker_block(ind_move_onset_catch(i),1));         % Accurate
            fprintf(marker_file_id,'catch-Target-Hit \t %d \n',target_reached_catch(i));                                                % Accurate
        end

        for j = 1:length(target_shown)
            fprintf(marker_file_id,'Target-Shown \t %d \n', target_shown(j));                                                                   % Accurate
        end

        for k= 1:length(target_shown_catch)
            fprintf(marker_file_id,'catch-Target-Shown \t %d \n', target_shown_catch(k));                                         % Accurate   
        end
            fclose(marker_file_id);

            % Import markers into EEGLAB
            if estimate_eeg_emg_delays == 1
                % Retrive old data set first 
                [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, EMG_dataset_no,'retrieve',EEG_dataset_no,'study',0); 
            end
            EEG = pop_importevent( EEG, 'event',[folder_path Subject_name '_ses' num2str(closeloop_Sess_num)...
            '_block' num2str(closeloop_Block_num) '_event_markers.txt'],'fields',{'type' 'latency'},'timeunit',NaN,'align',0);
            EEG = eeg_checkset( EEG );

         % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        %eeglab redraw;
        EEGLAB_dataset_to_merge = [EEGLAB_dataset_to_merge CURRENTSET];
     end
    
end

if ~isempty(EEGLAB_dataset_to_merge) && create_raster_plot == 0
    EEG = pop_mergeset( ALLEEG, EEGLAB_dataset_to_merge, 0);
    EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_all_blocks_eeg_raw'];
    EEG = eeg_checkset( EEG );
    % Update EEGLAB window
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;

    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_all_blocks_eeg_raw_modified.set'],...
   'filepath',folder_path);
    EEG = eeg_checkset( EEG );
end

%EEG_EMG_latency_calculated = [block1.EEG_EMG_latency_calculated; all_blocks.EEG_EMG_latency_calculated];

if estimate_eeg_emg_delays == 1 && create_raster_plot == 0
    save([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_eeg_kin_latencies.mat'],'EEG_EMG_latency_calculated',...
       'Total_EEG_EMG_latency','EEG_detected_times','EMG_detected_times','Accurate_ind_EEG_Go','EEG_kinematic_latency');
end

if create_raster_plot == 1
    
    for ch = 1:size(EEG_rastor,1)
        EEG_rastor_upsampled(ch,:) = interp1(emg_time_vector',EEG_rastor(ch,:)',kinematics(:,1),'pchip')';
    end
    for ch = 1:size(EMG_rms_resampled,1)
        EMG_rastor_upsampled(ch,:) = (interp1(emg_time_vector',EMG_rms_resampled(ch,:)',kinematics(:,1),'pchip'))';
    end
    marker_block(:,1) = marker_block(:,1) - (matlab_data_start_stop_samples(1) - eeg_time_correction);
    raster_data = [position_f,...
                   velocity_f,...
                   EMG_rastor_upsampled',...
                   mean(EEG_rastor_upsampled,1)',...
                   EEG_rastor_upsampled'...
                   ]; 
    
    raster_zscore = zscore(raster_data);
    raster_time = kinematics(:,1);
    raster_colors = ['k','k','k','k','k','k','k','k'];
    % Plot the rasters; Adjust parameters for plot
    [raster_row,raster_col] = size(raster_zscore);
    add_offset = 5.5;
    %raster_ylim1 = 0;
    %raster_ylim2 = (raster_col+1)*add_offset;
   
    figure('Position',[100 1500 3.5*116 3.5*116]);
    R_plot = tight_subplot(1,1,[0.05 0.05],[0.1 0.01],[0.01 0.1]);
    hold on;
    raster_zscore(:,1:4) = raster_zscore(:,1:4).*1;   % Invert Yaxis for EEG channels
    raster_zscore(:,5:raster_col) = -1.*raster_zscore(:,5:raster_col).*0.5;   
    original_raster_zscore = raster_zscore;
    for raster_index = 1:raster_col;
        raster_zscore(:,raster_index) = raster_zscore(:,raster_index) + add_offset*raster_index;  % Add offset to each channel of raster plot
        myhand(raster_index) = plot(R_plot,raster_time,raster_zscore(:,raster_index),'k','LineWidth',0.5);
        
        
    end
    axis([35 66 1 55]);
    %set(myhand(5),'LineWidth',2);
    %set(myhand(6),'LineWidth',2);
    %set(myhand(7),'LineWidth',2);
    %axis([pred_start_stop_times(1) pred_start_stop_times(2) 1 60]);
    %ylim([0 40]);
    myaxis = axis;
    for plot_ind3 = 1:length(indexB)
        line([marker_block(indexB(plot_ind3),1)/Fs_kin, marker_block(indexB(plot_ind3),1)/Fs_kin],[myaxis(3), myaxis(4)-3],...
            'Color','k','LineWidth',0.5,'LineStyle',':');
        %text(marker_block(indexB(plot_ind3),1)/Fs_kin,myaxis(4)+0.5,'E','Rotation',60,'FontSize',9);
        unfilled_tri = plot(marker_block(indexB(plot_ind3),1)/Fs_kin,myaxis(3),'^k','MarkerFaceColor','w','MarkerSize',4);
    end 
    
    for plot_ind4 = 1:length(ind_EEG_EMG_Go)
        line([marker_block(ind_EEG_EMG_Go(plot_ind4),1)./Fs_kin, marker_block(ind_EEG_EMG_Go(plot_ind4),1)./Fs_kin],[myaxis(3), myaxis(4)-3],...
            'Color','k','LineWidth',0.5,'LineStyle','-');
        %text(marker_block(ind_EEG_EMG_Go(plot_ind4),1)./Fs_kin+0.1,myaxis(4)-5,'GO','Rotation',0,'FontSize',9,'Color','r');
        filled_tri = plot(marker_block(ind_EEG_EMG_Go(plot_ind4),1)./Fs_kin,myaxis(3),'^k','MarkerFaceColor','k','MarkerSize',4);
    end
    
    for plot_ind2 = 1:size(block_performance,1);
        %line([block_performance(plot_ind2,2)./Fs_kin_raw, block_performance(plot_ind2,2)./Fs_kin_raw],[myaxis(3)-1, myaxis(4)],...
        %    'Color','k','LineWidth',0.5,'LineStyle','-');
        %text(pred_GO_times(plot_ind2),myaxis(4)+0.5,'EEG','Rotation',90,'FontSize',12);
        %line([block_performance(plot_ind2,3)./Fs_kin_raw, block_performance(plot_ind2,3)./Fs_kin_raw],[myaxis(3)-1, myaxis(4)],...
        %    'Color','g','LineWidth',0.5,'LineStyle','-');
        trial_time = block_performance(plot_ind2,2)/Fs_kin_raw:1/Fs_kin:block_performance(plot_ind2,3)/Fs_kin_raw;
        jbfill(trial_time,repmat(myaxis(4),1,length(trial_time)),repmat(myaxis(3)-0.5,1,length(trial_time)),[0.5 0.5 0.5],[0.5 0.5 0.5],0,0.3); % trial time must be row vector 
        text(trial_time(1)+0.5,myaxis(4)-1,'Attempt','Rotation',0,'FontSize',9,'Color','k');
        text(trial_time(end)+0.5,myaxis(4)-1,'Fixation','Rotation',0,'FontSize',9,'Color','k');
        %line([block_performance(plot_ind2,18)./Fs_kin_raw, block_performance(plot_ind2,18)./Fs_kin_raw],[myaxis(3)-1, myaxis(4)],...
        %    'Color','k','LineWidth',0.5,'LineStyle','-');
        %text(block_performance(plot_ind2,18)./Fs_kin_raw-1,myaxis(4)+0.5,'MO','Rotation',60,'FontSize',9,'Color','k');             
        %hold on;
    end
    
    text(myaxis(1)+0.5,myaxis(4)-1,'Fixation','Rotation',0,'FontSize',9,'Color','k');
    axes_pos = get(gca,'Position'); %[lower bottom width height]
    axes_ylim = get(gca,'Ylim');
    axes_xlim = get(gca,'Xlim');
    annotate_length = 5*(axes_pos(3))/(axes_xlim(2) - axes_xlim(1));
    annotation(gcf,'line', [axes_pos(1) (axes_pos(1) + annotate_length)],...
    [(axes_pos(2)) (axes_pos(2))],'LineWidth',1);
    %annotation('textbox',[axes_pos(1) 0 annotate_length 0.05],'String','5s','EdgeColor','none','FontSize',9);
    
    [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([unfilled_tri, filled_tri],{'EEG only','EMG-gated EEG (Intent)'},'ncol',2, 'ref',R_plot,...
                                            'anchor',[7 1],'buffer',[-10 -15],'box','off','xscale',0.5,'padding',[0 0 1]);
    
    for raster_index = 1:raster_col
        if raster_index == 1
            % Plot axes for elbow angle
            val_rom = 60; 
            y_upper = add_offset*raster_index + max(original_raster_zscore(:,raster_index)); 
            y_lower = add_offset*raster_index + (max(raster_data(:,raster_index)) - val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index));
            axes_y_upper = axes_pos(2) + (y_upper - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_y_lower = axes_pos(2) + (y_lower - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
                        [axes_y_lower axes_y_upper],'LineWidth',1);
            annotation('textbox',[0 axes_y_upper 0.9 0.05],'String',{'Elbow';'Position'},'EdgeColor','none','FontSize',9);
        end
        if raster_index == 2
            % Plot axes for elbow velocity
            val_rom = 10; 
            y_upper = add_offset*raster_index + (val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index)); 
            y_lower = add_offset*raster_index - (val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index));
            axes_y_upper = axes_pos(2) + (y_upper - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_y_lower = axes_pos(2) + (y_lower - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
                        [axes_y_lower axes_y_upper],'LineWidth',1);
            annotation('textbox',[0 axes_midline 0.9 0.05],'String',{'Velocity'},'EdgeColor','none','FontSize',9);
        end
        if raster_index == 3
            % Plot axes for biceps
            val_rom = 150; 
            y_upper = add_offset*raster_index + (min(raster_data(:,raster_index)) + val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index)); 
            y_lower = add_offset*raster_index + min(original_raster_zscore(:,raster_index));
            axes_y_upper = axes_pos(2) + (y_upper - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_y_lower = axes_pos(2) + (y_lower - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
                        [axes_y_lower axes_y_upper],'LineWidth',1);
%             annotation(gcf,'line', [(axes_pos(1)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
%                         [axes_midline axes_midline],'LineWidth',1);
            annotation('textbox',[0 axes_midline 0.9 0.05],'String',{'Biceps';'(rms)'},'EdgeColor','none','FontSize',9);
        end
        if raster_index == 4
            % Plot axes for triceps
%             val_rom = 150; 
%             y_upper = add_offset*raster_index + (min(raster_data(:,raster_index)) + val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index)); 
%             y_lower = add_offset*raster_index + min(original_raster_zscore(:,raster_index));
%             axes_y_upper = axes_pos(2) + (y_upper - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
%             axes_y_lower = axes_pos(2) + (y_lower - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
%             annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
%                         [axes_y_lower axes_y_upper],'LineWidth',1);
            annotation('textbox',[0 axes_midline 0.9 0.05],'String',{'Triceps';'(rms)'},'EdgeColor','none','FontSize',9);
        end
        if raster_index == 5
            % Plot axes for spatial average
            val_rom = 5; 
            y_upper = add_offset*raster_index + (val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index)); 
            y_lower = add_offset*raster_index - (val_rom - mean(raster_data(:,raster_index)))/std(raster_data(:,raster_index));
            axes_y_upper = axes_pos(2) + (y_upper - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_y_lower = axes_pos(2) + (y_lower - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.015) (axes_pos(1)+axes_pos(3)+0.015)],...
                        [axes_y_lower axes_y_upper],'LineWidth',1);
            annotation('textbox',[0 axes_midline+0.01 0.9 0.05],'String',{'Spatial';'Average'},'EdgeColor','none','FontSize',9);
        end
        
        if raster_index == 6
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation('textbox',[0 axes_midline+0.01 0.9 0.05],'String',{'C1'},'EdgeColor','none','FontSize',9);
        end
        
        if raster_index == 7
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation('textbox',[0 axes_midline+0.025 0.9 0.05],'String',{'Cz'},'EdgeColor','none','FontSize',9);
        end
                
        if raster_index == 8
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation('textbox',[0 axes_midline+0.01 0.9 0.05],'String',{'C2'},'EdgeColor','none','FontSize',9);
        end
        
        
        if raster_index == 9
            axes_midline = axes_pos(2) + (add_offset*raster_index - axes_ylim(1))*axes_pos(4)/(axes_ylim(2) - axes_ylim(1));
            annotation('textbox',[0 axes_midline 0.9 0.05],'String',{'CP2'},'EdgeColor','none','FontSize',9);
        end
        
    end
    set(gca,'Visible','off');
    %set(gca,'YTick', add_offset.*(1:raster_col),'Box','off');
    
    %set(gca,'YTickLabel',{{'Posi';'tion'}});%,{'Elbow';'Velocity'},'Triceps','Biceps',{'Spatial'; 'Avg'},'CP2','C2','Cz','C1'},'FontSize',9);
    %set(gca,'YTickLabel',{'Pos','Vel','Biceps','Triceps','Spatial Avg','C1','Cz','C2','CP2'},'FontSize',9);
    %xlabel('Time (s)', 'FontSize',9);
    %ylim([3 55]);
    %axis([230 295 0 50]);
    %export_fig 'Block8_results' '-png' '-transparent'
end


