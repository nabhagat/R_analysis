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
Subject_name = 'BNBO';      %change1
closeloop_Sess_num = 5;     %change2
folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3

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
 biceps_threshold = 31*ones(1,8);                  
 triceps_threshold = 21*ones(1,8);


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
%biceps_threshold = [8 8 8 10 8 8 8 8 8];                  
%triceps_threshold = [6.5 6.5 6.5 8 7.2 7.2 7.2 7.2 7.2];

% Flags for selecting parts of code
import_event_markers_into_eeglab = 1; % must select estimate_eeg_emg_delays; both are dependent
estimate_eeg_emg_delays = 1;

% Fixed variables
Fs_eeg = 500; 
resamp_Fs = 500;            
EEG_channels_to_import = 1:64;
EMG_channels = [17 22 41 46];
EEG_channels_to_import(EMG_channels) = [];
Total_EEG_EMG_latency = [];
EEG_EMG_latency_calculated = [];
%EEG_EMG_latency_observed = [];     - Not accurate
EEG_detected_times = [];
EMG_detected_times = [];
Accurate_ind_EEG_Go = [];

% Load cloop_statistics.csv file 
cl_ses_data = dlmread([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_cloop_statistics.csv'],',',7,1); 
unique_blocks = unique(cl_ses_data(:,1));
EEGLAB_dataset_to_merge = [];

for m = 1:length(unique_blocks)
    if (strcmp(Subject_name,'LSGR') && (closeloop_Sess_num == 4)) 
        if (m == 1) || (m==3) || (m==4)           % LSGR_ses4
            %continue
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
    EEG=pop_chanedit(EEG, 'lookup','C:\NRI_BMI_Mahi_Project_files\EEGLAB_13_1_1b\eeglab13_1_1b\plugins\dipfit2.2\standard_BESA\standard-10-5-cap385.elp');
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
    
    target_shown = eeg_stimulus_trig((block_performance(:,4) == 1));
    target_shown_catch = eeg_stimulus_trig((block_performance(:,4) == 2));
      
    %EEG_EMG_latency_observed = [EEG_EMG_latency_observed (marker_block(ind_EEG_EMG_Go,1)' - marker_block(ind_EEG_Go,1)')/Fs_eeg];
    %EEG_EMG_latency_observed = [EEG_EMG_latency_observed (marker_block(ind100-1,1)' - marker_block(ind100-2,1)')/Fs_eeg];
    
    if estimate_eeg_emg_delays == 1
       % Use main file for extracting EMG channels
       % [EEG, com] = pop_loadbv(path, hdrfile, srange, chans);
       if closeloop_Block_num > 9
            EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop00' num2str(closeloop_Block_num) '.vhdr'],...
                eeg_start_stop_samples, EMG_channels);
       else
            EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop000' num2str(closeloop_Block_num) '.vhdr'],...
                eeg_start_stop_samples, EMG_channels);
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
        
        EMG_rms_resampled = resample(EMG_rms',resamp_Fs,emg_Fs)';       % redundant bcoz resamp_Fs = emg_Fs
        emg_time_vector = 0:1/resamp_Fs:(length(EMG_rms_resampled)-1)/resamp_Fs;
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
            if isempty(biceps_sample_num)
                % if a is empty, return b
                first_sample(j) = triceps_sample_num;
            elseif isempty(triceps_sample_num)
                % if b is empty, return a
                first_sample(j) = biceps_sample_num;
            elseif isempty(biceps_sample_num) && isempty(triceps_sample_num)
                first_sample(j) = 7501; % Timeout = 15 sec
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
        figure; hold on;
        plot(emg_time_vector,EMG_rms_corrected');
        plot(emg_time_vector,100.*above_threshold_move_counts,'k','LineWidth',1);
        plot(emg_time_vector,100.*detected_above_threshold_move_counts,'or','LineWidth',2);
        line([emg_time_vector(1) emg_time_vector(end)],[biceps_threshold(m) biceps_threshold(m)],'Color','b',...
            'LineWidth',1.5','LineStyle','--');
        line([emg_time_vector(1) emg_time_vector(end)],[triceps_threshold(m) triceps_threshold(m)],'Color','g',...
            'LineWidth',1.5','LineStyle','--');     
        
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
            [[ind_EEG_Go; ind_EEG_Go_catch] accurate_ind_EEG_Go ind100]
            combined_latencies'
    end
    
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

if ~isempty(EEGLAB_dataset_to_merge)
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

if estimate_eeg_emg_delays == 1
    save([folder_path Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_eeg_emg_latencies.mat'],'EEG_EMG_latency_calculated',...
       'Total_EEG_EMG_latency','EEG_detected_times','EMG_detected_times','Accurate_ind_EEG_Go');
end

