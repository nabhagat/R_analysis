% Program to evaluate Relative Power Level (RPL) and correlate with BMI performance across all subjects and sessions of clinical study - NRI Project
% Author: Nikunj A. Bhagat, Graduate Student, University of Houston, 
% Contact: nbhagat08[at]gmail.com
% Date: February 18, 2016
%------------------------------------------------------------------------------------------------------------------------
% Revisions
%                    
%                    
%------------------------------------------------------------------------------------------------------------------------
clear all; 
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 


EEG_channel_nos = 1:64;
EMG_channel_nos = [17 22 41 42 45 46 51 55];
Scalp_plot_channels = EEG_channel_nos;
Scalp_plot_channels(EMG_channel_nos) = [ ];
Subject_name_all = {'S9014','S9012','S9011','S9010','S9009','S9007'};
Subject_number = [9014,9012,9011,9010,9009,9007];
Subject_labels = {'S6','S5','S4','S3','S2','S1'};
ACC_marker_color = {'-sk','-sk','-vr','-^m','--ok','-sb'};
Marker_face_color = {'k','k','r','m','w','b'};
Sig_line_color = {'--k','--k','--k','--k','--k','--k'};
h_acc = zeros(length(Subject_name_all),1);
directory = 'D:\NRI_Project_Data\Clinical_study_Data\';
poster_font_size = 12;

% Compute power in different bands 
    delta_range = [0.1 4];
    theta_range = [4 8];
    alpha_range = [8 13];
    beta_range = [13 30];
    gamma_range1 = [30 70];
    %gamma_range2 = [65 85];
    tot_range1 = [0.1 100];
        %tot_range2 = [0.1 85];

% Subject Details 
% Subject_name = 'S9009'; % change1
% subj_n = 6;
% Sess_num = '2';  % For calibration and classifier model             
% Cond_num = 1;  % 1 - Active/User-driven; 2 - Passive; 3 - Triggered/User-triggered; 4 - Observation 
% Block_num = 160;

%Gamma_RPL_channels = [1 2 33 34 28 35 36 37 4 38 5 39 6 40]; % Common
Gamma_RPL_channels = [1 2 33 34 28 35 36 37 4 38 5 39 6 40 43 9 32 10 44];
%Gamma_RPL_channels = [4 38 5 39 6 43 9 32 10 44 13 48 14 49 15];
%Gamma_RPL_channels = [1 2 33 34 28 35 4 38 5 39 6 9 32 10 44]; %S9010
%Gamma_RPL_channels = [1 2 33 36 38 5 39 40 9 32 10 44]; %S9007
subject_wise_correlation_values = zeros(1,length(Subject_name_all));
subject_wise_correlation_p_values = zeros(1,length(Subject_name_all));
subject_wise_average_gamma_RPL = zeros(1,length(Subject_name_all));
%Gamma_RPL_channels = [34 28 35 4 38 5 39 6];

% Flags to control the processing 
readbv_files = 0;
segment_eeg_baseline = 0;
use_saved_data = 1;
figure('Position',[-1500 200 1204 3*116]); 
Cplot = tight_subplot(1,1,[0.02 0.02],[0.15 0.05],[0.05 0.05]);


 for subj_n = 1:length(Subject_name_all)
     Subject_name = Subject_name_all{subj_n}; % change1
    switch Subject_name
        case 'S9007'
            blocks_nos_to_import = [2 1 1 1 1 1 1 1 1 1 1 1]; % S9007_start_blocks
            all_sessions_nos = 3:14;
        case 'S9009'
            blocks_nos_to_import = [1 1 1 1 1 1 2 1 1 1 1 1]; % S9009_start_blocks
            %blocks_nos_to_import = [3 3 3 4 4 4 4 4 4 4 4 4]; % S9009_mid_blocks
            all_sessions_nos = 4:15;
        case 'S9010'
            blocks_nos_to_import = [2 1 1 2 1 1 1 1 1 1 1 1]; % S9010_start_blocks
            %blocks_nos_to_import = [9 8 8 9 8 8 8 8 8 8 8 8]; % S9010_end_blocks
            all_sessions_nos = 3:14;
        case 'S9011'
            blocks_nos_to_import = [6 1 1 4 4 3 4 4 4 4 6 4]; % S9011_start_blocks
            all_sessions_nos = 3:14;
        case 'S9012'
            blocks_nos_to_import = [1 1 1 1 2 1 1 1 1 1 1 1]; % S9012_start_blocks
            all_sessions_nos = 3:14;
        case 'S9014'
            blocks_nos_to_import = [3 1 1 4 1 1 1 1 1 2 1 1]; % S9014_start_blocks
            all_sessions_nos = 3:14;
    end

    if use_saved_data == 1
        closeloop_folder_path = [directory 'Subject_' Subject_name '\']; % change3
        load([closeloop_folder_path Subject_name '_bmi_performance.mat']);
        load([closeloop_folder_path Subject_name '_all_sessions_beta_RPL.mat']);
        load([closeloop_folder_path Subject_name '_all_sessions_gamma_RPL.mat']);
    else
    
        % all_sessions_delta_RPL = [];
        % all_sessions_theta_RPL = [];
        % all_sessions_alpha_RPL = [];
        all_sessions_beta_RPL = [];
        all_sessions_gamma_RPL = [];

        % all_sessions_delta_norm = [];
        % all_sessions_theta_norm = [];
        % all_sessions_alpha_norm = [];
        % all_sessions_beta_norm = [];
        % all_sessions_gamma1_norm = [];


    
    
        for ses_n = 1:length(all_sessions_nos)
            closeloop_Sess_num = all_sessions_nos(ses_n); 
            closeloop_folder_path = ['D:\NRI_Project_Data\Clinical_study_Data\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3

            %% Import raw BrainVision files (.eeg, .vhdr, .vmrk)  to EEGLAB dataset
            if readbv_files == 1
                %total_no_of_trials = 0;
                %for block_index = 1:length(blocks_nos_to_import)
                block_index = blocks_nos_to_import(ses_n);
                fprintf('\nImporting block # %d of %d blocks...\n',blocks_nos_to_import(ses_n),length(blocks_nos_to_import));     
               if blocks_nos_to_import(ses_n) > 9
                    EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block00' num2str(blocks_nos_to_import(ses_n)) '.vhdr'], [], 1:64);
               else
                    EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' num2str(closeloop_Sess_num) '_closeloop_block000' num2str(blocks_nos_to_import(ses_n)) '.vhdr'], [], 1:64);
               end

                EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_raw'];
                EEG = eeg_checkset( EEG );

                EEG=pop_chanedit(EEG, 'lookup','D:\Nikunj_Data\MATLAB_libraries\EEGLAB_13_1_1b\eeglab13_1_1b\plugins\dipfit2.3\standard_BESA\standard-10-5-cap385.elp');
                EEG = eeg_checkset( EEG );
                EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_raw.set'],...
                    'filepath',closeloop_folder_path);
                EEG = eeg_checkset( EEG );
                % Update EEGLAB window
                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
                %    eeglab redraw;
            %   end
            end

            %% Segment raw EEG to determine baseline (30s)
            if segment_eeg_baseline == 1
                EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_raw.set'], closeloop_folder_path); 
                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                eeglab redraw;

                % Define Paramteres
                raw_eeg_Fs = EEG.srate;      
                raw_eeg = EEG.data;
                [eeg_nbchns,eeg_pnts] = size(EEG.data);

        % %         % 60 Hz - notch filter
        % %         wo = 60/(raw_eeg_Fs/2);
        % %         quality_factor = 35; 
        % %         [b_notch,a_notch] = iirnotch(wo,wo/quality_factor); % iirnotch(wo,bw); wo = 60/(Fs/2);  bw = wo/35;
        % %         %fvtool(b_notch,a_notch,'Analysis','freq','Fs',raw_eeg_Fs)
        % %         notch_filtered_eeg = zeros(size(raw_eeg));
        % %         for i = 1:eeg_nbchns
        % %             notch_filtered_eeg(i,:) = filtfilt(b_notch,a_notch,double(raw_eeg(i,:))); % filtering with zero-phase delay
        % %         end
        % %         EEG.data = notch_filtered_eeg;

                block_start_trigger_latency = [];
                for j=1:length(EEG.event)-1
                    if (strcmp(EEG.event(j).type,'S 42'))
                        block_start_trigger_latency = [block_start_trigger_latency; EEG.event(j).latency/raw_eeg_Fs];
                        break;
                    end
                end
                if isempty(block_start_trigger_latency)
                    errordlg('Trial start triggere "S 42" not found');
                end

                % Plot pre-trial start EEG data and 'manually' select interval
        %         eegplot(EEG.data(EEG_channel_nos,:), 'srate', EEG.srate, 'spacing', 200, 'eloc_file', EEG.chanlocs, 'limits', [EEG.xmin*1000 block_start_trigger_latency],...
        %             'winlength', 30, 'title', 'EEG channel activities using eegplot()'); 

                baseline_start = input('Enter start time for 20 sec baseline segment in seconds: ');
                baseline_stop = baseline_start + 20.00;
                EEG = pop_select(EEG,'time',[baseline_start baseline_stop]);
                EEG.setname=[Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_baseline'];
                EEG = eeg_checkset( EEG );
                EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_baseline.set'],...
                    'filepath',closeloop_folder_path);
                EEG = eeg_checkset( EEG );

                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                eeglab redraw;
            end

            %% Filter baseline EEG segment and compute RPL

            EEG = pop_loadset( [Subject_name '_ses' num2str(closeloop_Sess_num)  '_closeloop_block' num2str(blocks_nos_to_import(ses_n)) '_eeg_baseline.set'], closeloop_folder_path); 
            [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
            eeglab redraw;

            % Define Paramteres
            raw_eeg_Fs = EEG.srate;
            raw_eeg = EEG.data;
            [eeg_nbchns,eeg_pnts] = size(EEG.data);

        %     % 60 Hz - notch filter
            wo = 60/(raw_eeg_Fs/2);
            quality_factor = 35; 
            [b_notch,a_notch] = iirnotch(wo,wo/quality_factor); % iirnotch(wo,bw); wo = 60/(Fs/2);  bw = wo/35;
            %fvtool(b_notch,a_notch,'Analysis','freq','Fs',raw_eeg_Fs)
            notch_filtered_eeg = zeros(size(raw_eeg));
            for i = 1:eeg_nbchns
                notch_filtered_eeg(i,:) = filtfilt(b_notch,a_notch,double(raw_eeg(i,:))); % filtering with zero-phase delay
            end
            %EEG.data = notch_filtered_eeg;

            % Band pass filter between [0.1 - 100 Hz]
            [b_bpf,a_bpf] = butter(4,([0.1 100]./(raw_eeg_Fs/2)),'bandpass');
            BPFred_eeg = zeros(size(raw_eeg));
            %fvtool(b_bpf,a_bpf,'Analysis','freq','Fs',raw_eeg_Fs)
            for i = 1:eeg_nbchns
                BPFred_eeg(i,:) = filtfilt(b_bpf,a_bpf,notch_filtered_eeg(i,:)); % filtering with zero-phase delay
                %BPFred_eeg(i,:) = filtfilt(b_bpf,a_bpf,double(raw_eeg(i,:))); % filtering with zero-phase delay
            end

            % EEG.data = BPFred_eeg;
            % [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
            % eeglab redraw;
            % figure;
            % [spectra,freqs,speccomp,contrib,specstd] = spectopo(EEG.data,0,raw_eeg_Fs,'freq',[10 30 70],...
            %                                                                                 'chanlocs',EEG.chanlocs, 'nfft',512,'freqrange',[0.1 100], 'plotchans',EEG_channel_nos,...
            %                                                                                 'winsize', 512, 'overlap',256);
            % Equivalent to  [pxx,freqs] = pwelch(EEG.data(1,:),512,256,512,raw_eeg_Fs); figure; plot(freqs, 10*log10(pxx)) - % detrend makes no difference since band pass filtered                                                                           

            % Compute power spectrum density using pwelch, default : window length is length(data)/8, overlap = 50% - preferred for reducing spectral leakage
                NFFT = 512; % Freq_Res = Fs/NFFT - This is true !!
                PSD_eeg = zeros(eeg_nbchns,((NFFT/2)+1));
                PSD_f = zeros(1,((NFFT/2)+1));      % range = (-fs/2:0:fs/2). But PSD is real and symmetric. Hence only +ve frequencies plotted
                for psd_len = 1:eeg_nbchns
                    [PSD_eeg(psd_len,:),PSD_f] = pwelch(BPFred_eeg(psd_len,:),[],[],NFFT,raw_eeg_Fs);
                end
        %         figure; hold on; grid on;
        %         plot(PSD_f,10*log10(PSD_eeg(EEG_channel_nos,:)))
        %         xlabel('Frequency (Hz)');
        %         ylabel('PSD (dB)');
        %         title('EEG Baseline PSD');
        %         hold off

                %delta_power = trapz(PSD_f(PSD_f>=delta_range(1) & PSD_f<delta_range(2)),(PSD_eeg(:,PSD_f>=delta_range(1) & PSD_f<delta_range(2))'))';
                %theta_power = trapz(PSD_f(PSD_f>=theta_range(1) & PSD_f<theta_range(2)),(PSD_eeg(:,PSD_f>=theta_range(1) & PSD_f<theta_range(2))'))';
                %alpha_power = trapz(PSD_f(PSD_f>=alpha_range(1) & PSD_f<alpha_range(2)),(PSD_eeg(:,PSD_f>=alpha_range(1) & PSD_f<alpha_range(2))'))';
                beta_power = trapz(PSD_f(PSD_f>=beta_range(1) & PSD_f<beta_range(2)),(PSD_eeg(:,PSD_f>=beta_range(1) & PSD_f<beta_range(2))'))';
                gamma_power = trapz(PSD_f(PSD_f>=gamma_range1(1) & PSD_f<gamma_range1(2)),(PSD_eeg(:,PSD_f>=gamma_range1(1) & PSD_f<gamma_range1(2))'))';
                total_band_power = trapz(PSD_f(PSD_f>=tot_range1(1) & PSD_f<tot_range1(2)),(PSD_eeg(:,PSD_f>=tot_range1(1) & PSD_f<tot_range1(2))'))';

        %        total_band_power = tot_power1(EEG_channel_nos); 
        %         delta_norm = delta_power(EEG_channel_nos)./total_band_power;
        %         delta_RPL = delta_norm./sum(delta_norm);
        %         theta_norm = theta_power(EEG_channel_nos)./total_band_power;
        %         theta_RPL = theta_norm./sum(theta_norm);
        %         alpha_norm = alpha_power(EEG_channel_nos)./total_band_power;
        %         alpha_RPL = alpha_norm./sum(alpha_norm);

                %beta_norm = beta_power(EEG_channel_nos)./total_band_power;
                %beta_RPL = beta_norm./sum(beta_norm);
                beta_norm = beta_power./total_band_power; % Use all 64 channels to maintain consistency with EEGLAB
                beta_RPL = beta_norm./sum(beta_norm(EEG_channel_nos)); % Remove EMG channels and sum only the EEG channels

                %gamma_norm1 = gamma_power1(EEG_channel_nos)./total_band_power;  
                %gamma_RPL1 = gamma_norm1./sum(gamma_norm1);
                gamma_norm = gamma_power./total_band_power;  
                gamma_RPL = gamma_norm./sum(gamma_norm(EEG_channel_nos));

        %     figure; plot(EEG_channel_nos,delta_RPL,'-ok'); 
        %     hold on; plot(EEG_channel_nos,theta_RPL,'-og')
        %     hold on; plot(EEG_channel_nos,alpha_RPL,'-or')
        %     hold on; plot(EEG_channel_nos,beta_RPL,'-om')
        %     hold on; plot(EEG_channel_nos,gamma_RPL1,'-ob')

        %         all_sessions_delta_RPL = [all_sessions_delta_RPL delta_RPL];
        %         all_sessions_theta_RPL = [all_sessions_theta_RPL theta_RPL];
        %         all_sessions_alpha_RPL = [all_sessions_alpha_RPL alpha_RPL];
                all_sessions_beta_RPL = [all_sessions_beta_RPL beta_RPL];
                all_sessions_gamma_RPL = [all_sessions_gamma_RPL gamma_RPL];

        %         all_sessions_delta_norm = [all_sessions_delta_norm delta_norm];
        %         all_sessions_theta_norm = [all_sessions_theta_norm theta_norm];
        %         all_sessions_alpha_norm = [all_sessions_alpha_norm alpha_norm];
        %         all_sessions_beta_norm = [all_sessions_beta_norm beta_norm];
        %         all_sessions_gamma1_norm = [all_sessions_gamma1_norm gamma_norm];
        end

%% Analyze BMI accuracy from clinical study
        bmi_performance = [];
        bmi_performance_blockwise = [];

        fileid = dir([directory 'Subject_' Subject_name '\' Subject_name '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_name '\' fileid.name];
        subject_study_data = dlmread(results_filename,',',1,0);
        % Array fields - subject_study_data
        % 1 - Session_nos	2 - Block_number	3- Start_of_trial	4 - End_of_trial	5 - Valid_or_catch	
        % 6 - Intent_detected	7 - Time_to_trigger     8 -  Number_of_attempts     9 - EEG_decisions	10 - EEG_EMG_decisions	
        % 11 - MRCP_slope	12 - MRCP_neg_peak	13 - MRCP_AUC	14 - MRCP_mahalanobis	15 - feature_index	
        % 16 - Corrected_spatial_chan_avg_index     17 - Correction_applied_in_samples      18 - Likert_score	19 - Target     
        % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency 23 - EMG_decisions

        unique_session_nos = unique(subject_study_data(:,1));
        subject_intent_per_min = [];
            for ses_num = 1:length(unique_session_nos)
                session_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),:);

                ind_valid_trials = find(session_performance(:,5) == 1);  % col 5 - Valid(1) or Catch(2)
                ind_success_valid_trials = find((session_performance(:,5) == 1) & (session_performance(:,6) == 1)); % col 5 - Intent detected
                session_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                ind_catch_trials = find(session_performance(:,5) == 2);
                ind_success_catch_trials = find((session_performance(:,5) == 2) & (session_performance(:,6) == 0)); % True negative
                ind_failed_catch_trials = find((session_performance(:,5) == 2) & (session_performance(:,6) == 1)); % False positives
                session_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                Session_accuracy = (length(ind_success_valid_trials) + length(ind_success_catch_trials))/(length(ind_valid_trials) + length(ind_catch_trials));

                time_to_trigger_success_valid_trials = session_performance(ind_success_valid_trials,7); %col 7 - Time to Trigger
                Session_Intent_per_min = 60./time_to_trigger_success_valid_trials;

                subject_intent_per_min  = [subject_intent_per_min; [unique_session_nos(ses_num).*ones(length(Session_Intent_per_min),1) Session_Intent_per_min]];

                session_latencies = session_performance(ind_success_valid_trials,22);
                session_latencies(session_latencies < -1000 | session_latencies > 1000) = [];
                Session_detection_latency_mean = mean(session_latencies);
                Session_detection_latency_std = std(session_latencies);
                Session_likert_mean = mean(session_performance(:,18));
                Session_likert_std = std(session_performance(:,18));

                unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));
                % Block wise performance
                for block_num = 1:length(unique_block_nos)
                    block_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num) & ...
                                                                                                    subject_study_data(:,2) == unique_block_nos(block_num),:);
                    %ind_success_valid_trials = find((block_performance(:,5) == 1) & (block_performance(:,6) == 1));
                    %time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,7); %col 7 - Time to Trigger
                    %Block_Intent_per_min = 60./time_to_trigger_success_valid_trials;

                    block_ind_valid_trials = find(block_performance(:,5) == 1);  % col 5 - Valid(1) or Catch(2)
                    block_ind_success_valid_trials = find((block_performance(:,5) == 1) & (block_performance(:,6) == 1)); % col 5 - Intent detected
                    block_TPR = length(block_ind_success_valid_trials)/length(block_ind_valid_trials);      % TPR

                    block_ind_catch_trials = find(block_performance(:,5) == 2);
                    block_ind_failed_catch_trials = find((block_performance(:,5) == 2) & (block_performance(:,6) == 1));
                    block_ind_success_catch_trials = find((block_performance(:,5) == 2) & (block_performance(:,6) == 0));
                    block_FPR = length(block_ind_failed_catch_trials)/length(block_ind_catch_trials); %FPR

                    block_accuracy = (length(block_ind_success_valid_trials) + length(block_ind_success_catch_trials))/...
                                                     (length(block_ind_valid_trials) + length(block_ind_catch_trials));

                    bmi_performance_blockwise = [bmi_performance_blockwise;...
                                                                               [subj_n unique_session_nos(ses_num) unique_block_nos(block_num) block_TPR block_FPR block_accuracy] ]; % Added 10-18-2016

                end

                mean_Session_accuracy_blockwise = mean(bmi_performance_blockwise((bmi_performance_blockwise(:,1) == subj_n) & (bmi_performance_blockwise(:,2) == unique_session_nos(ses_num)),6));
                std_Session_accuracy_blockwise = std(bmi_performance_blockwise((bmi_performance_blockwise(:,1) == subj_n) & (bmi_performance_blockwise(:,2) == unique_session_nos(ses_num)),6));

                bmi_performance = [bmi_performance;...
                                                [subj_n unique_session_nos(ses_num) session_TPR session_FPR length(ind_valid_trials) length(ind_catch_trials)...
                                                                mean(Session_Intent_per_min) std(Session_Intent_per_min) median(Session_Intent_per_min) Session_likert_mean Session_likert_std...
                                                                Session_detection_latency_mean Session_detection_latency_std Session_accuracy...
                                                                mean_Session_accuracy_blockwise std_Session_accuracy_blockwise]];   
                %        1          2                 3                  4                 5                        6                                                    7 
                % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
                %                    8                                          9                                                       10                              11
                % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
                %                   12                                                                    13                                                      14
                % Session_detection_latency_mean    Session_detection_latency_std           Session_accuracy       
                %                   15                                                                    16
                % mean_Session_accuracy_blockwise    std_Session_accuracy_blockwise
            end
    end
    
%% Compute correlations between BMI accuracy and EEG RPL
    %        1          2                 3                  4                 5                        6                                                    7 
    % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
    %                    8                                          9                                                       10                              11
    % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
    %                   12                                                                    13                                                      14
    % Session_detection_latency_mean    Session_detection_latency_std           Session_accuracy       
    %                   15                                                                    16
    % mean_Session_accuracy_blockwise    std_Session_accuracy_blockwise

    bmi_accuracies =  bmi_performance(:,15)';
    bmi_accuracies_variance = bmi_performance(:,16)';
    %bmi_accuracies_CoV = bmi_accuracies_variance./bmi_accuracies;
    % X = [12 x 56]; Y = [12 x 1]; corr(X,Y) = [56 x 1]

    % [all_sessions_delta_RPL_corr,all_sessions_delta_RPL_corr_pvals] = corr(all_sessions_delta_RPL',bmi_accuracies_variance');
    % [all_sessions_theta_RPL_corr,all_sessions_theta_RPL_corr_pvals] = corr(all_sessions_theta_RPL',bmi_accuracies_variance');
    % [all_sessions_alpha_RPL_corr,all_sessions_alpha_RPL_corr_pvals] = corr(all_sessions_alpha_RPL',bmi_accuracies_variance');
    [all_sessions_beta_RPL_corr,all_sessions_beta_RPL_corr_pvals] = corr(all_sessions_beta_RPL',bmi_accuracies');   % correlate with BMI accuracy variance
    [all_sessions_gamma_RPL_corr,all_sessions_gamma_RPL_corr_pvals] = corr(all_sessions_gamma_RPL',bmi_accuracies'); % correlate with average BMI accuracy

    % [all_sessions_delta_CoV_corr,all_sessions_delta_norm_corr_pvals] = corr(all_sessions_delta_RPL',bmi_accuracies_CoV');
    % [all_sessions_theta_CoV_corr,all_sessions_theta_norm_corr_pvals] = corr(all_sessions_theta_RPL',bmi_accuracies_CoV');
    % [all_sessions_alpha_CoV_corr,all_sessions_alpha_norm_corr_pvals] = corr(all_sessions_alpha_RPL',bmi_accuracies_CoV');
    % [all_sessions_beta_CoV_corr,all_sessions_beta_norm_corr_pvals] = corr(all_sessions_beta_RPL',bmi_accuracies_CoV');
    % [all_sessions_gamma1_CoV_corr,all_sessions_gamma1_norm_corr_pvals] = corr(all_sessions_gamma1_RPL',bmi_accuracies_CoV');

    % Compute average RPL for frontal channels and compute correlation with BMI
    % accuracy
    all_channels_all_sessions_Frontal_gamma_RPL = all_sessions_gamma_RPL(Gamma_RPL_channels,:);
    Avg_all_sessions_frontal_gamma_RPL = mean(all_channels_all_sessions_Frontal_gamma_RPL);
    [corr_with_gamma_RPL,corr_with_gamma_RPL_p] = corr(Avg_all_sessions_frontal_gamma_RPL', bmi_accuracies');
    subject_wise_correlation_values(subj_n) = corr_with_gamma_RPL;
    subject_wise_correlation_p_values(subj_n) = corr_with_gamma_RPL_p;
    subject_wise_average_gamma_RPL(subj_n) = mean(Avg_all_sessions_frontal_gamma_RPL);
%% Plot normalized BMI accuracy across session

  
%     figure('Position',[-1100 700 7*116 2.5*116]); 
%     Cplot = tight_subplot(1,1,[0.05 0.02],[0.25 0.1],[0.1 0.05]);
%     axes(Cplot(1)); hold on;
    
    if subj_n == 1
        axes(Cplot(subj_n)); hold on;
        h_acc = plot(1:size(bmi_performance,1), bmi_accuracies./max(bmi_accuracies), ACC_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1,'MarkerSize',10);
        h_rpl = plot(1:size(bmi_performance,1), Avg_all_sessions_frontal_gamma_RPL./max(Avg_all_sessions_frontal_gamma_RPL), '--ob','LineWidth',1,'MarkerFaceColor','b','MarkerSize',10);
        ylim([0 1.1]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 0.5 1],'YtickLabel',{'0' '0.5' '1'},'FontSize',poster_font_size);
        %set(gca,'Xtick',1:12,'Xticklabel',{' '});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'},'FontSize',poster_font_size);
        xlabel('Therapy Sessions','FontSize',poster_font_size);
        %title(['Closed-loop BMI performance ' Subject_name ' corr = ' num2str(corr_with_gamma_RPL)],'FontSize',10);
        ylabel({'Normalized units'},'FontSize',poster_font_size);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        %[legend_h,object_h,plot_h,text_str] = legendflex([h_acc, h_rpl],{'Avg. BMI accuracy','Avg. Gamma RPL'},'ncol',2, 'nrow',1,'ref',Cplot(subj_n),'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
        legend([h_acc h_rpl],{'BMI accuracy','Avg. Gamma RPL'},'Orientation','Horizontal','FontSize',poster_font_size)
        hold off;
        annotation('textbox',[0.9 0 0.1 0.07],'String','\rho = 0.53','EdgeColor','none','FontSize',poster_font_size);    
        
 
    %% Plot topographic images of EEG bands for correlation with RPL
    
% %     EEG = pop_loadset( [Subject_name '_ses14_closeloop_block1_eeg_baseline.set'], [closeloop_folder_path '\S9014_Session14\']); 
% %     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
% %     eeglab redraw;
% % 
% %     Total_num_sessions = 1:12;
% %     Channels_used = Scalp_plot_channels; 
% %     emarker_chans = [];
% %     for ch_sel_cnt = 1:length(Gamma_RPL_channels)
% %         emarker_chans(ch_sel_cnt) = find(Channels_used == Gamma_RPL_channels(ch_sel_cnt));
% %     end
% %     
% %     figure('Position',[-1100 50 9*116 3*116]); 
% %     Scalp_plot = tight_subplot(1,length(Total_num_sessions),[0],[0],[0 0]);
% %     
% %     %freq_intervals = {'\delta', '\theta', '\alpha', '\beta', '\gamma1'};
% %     %Scalp_plot = tight_subplot(1,length(freq_intervals),[0.02 0.02],[0.05 0.1],[0.05 0.05]);
% %     %ScalpData = [delta_RPL, theta_RPL, alpha_RPL, beta_RPL, gamma_RPL1];
% %     %ScalpData = [all_sessions_beta_RPL, all_sessions_gamma_RPL];
% % 
% %     for ses_num = 1:length(Total_num_sessions)
% %        axes(Scalp_plot(ses_num));
% %        topoplot(all_sessions_gamma_RPL(EEG_channel_nos,ses_num), EEG.chanlocs(EEG_channel_nos),'maplimits', [0 0.02],'style','both',...    
% %             'electrodes','off','plotchans',Scalp_plot_channels,'plotrad', 0.55,'headrad',0.5,...
% %             'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'on','conv','off',...
% %             'emarker2',{emarker_chans,'.','k',6,0.5});
% %         %topoplot(ScalpData(:,tpt),EEG.chanlocs(EEG_channel_nos),'plotrad', 0.55);
% % %         title(ses_num,'FontSize',18);
% %         colormap(hot);
% % %         if ses_num == 0
% % %             ylabel('\gamma','FontSize',18);
% % %         end
% %         
% % %        axes(Scalp_plot(ses_num+12));
% % %        topoplot(all_sessions_beta_RPL(EEG_channel_nos,ses_num), EEG.chanlocs(EEG_channel_nos),'maplimits', [0 0.02],'style','map',...    
% % %             'electrodes','on','plotchans',[],'plotrad', 0.55,'headrad',0.5,...
% % %             'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off','conv','on');
% % %             %'emarker2',{emarker_chans,'.','k',16,2});
% % %         %topoplot(ScalpData(:,tpt),EEG.chanlocs(EEG_channel_nos),'plotrad', 0.55);
% % %         colormap(hot);
% % %         if ses_num == 1
% % %             ylabel('\gamma','FontSize',18);
% % %         end
% %         
% %     end
% %     
% %     %mtit('Bandwise RPL across 12 sessions from resting state EEG','FontSize',12);
% %     cbar_axes = colorbar('location','SouthOutside','XTick',[0 0.01 0.02],'XTickLabel',{'0','0.01', '0.02'});
% %     set(cbar_axes,'Position',[0.75 0.28 0.2 0.05]);
% %     ylabel(cbar_axes,'Gamma relative power level','FontSize',10);
% %     set(get(cbar_axes,'YLabel'),'Rotation',0); 
% %     
    end
%% Plot topographic images of EEG bands for correlation with BMI accuracy and variance

    % Channels_used = [4 5 6 9 10 13 14 15 19 20 24 25 26 32 ...
    %                  38 39 43 44 48 49 52 53 54 57 58];
    % emarker_chans = [];
    % for ch_sel_cnt = 1:length(Channels_sel)
    %     emarker_chans(ch_sel_cnt) = find(Channels_used == Channels_sel(ch_sel_cnt));
    % end

% %     %freq_intervals = {'\delta', '\theta', '\alpha', '\beta', '\gamma1'};
% %     freq_intervals = {'\beta', '\gamma'};
% %     figure('Position',[-1700 100 4*116 3*116]); 
% %     Corr_plot = tight_subplot(1,length(freq_intervals),[0.02 0.02],[0.05 0.1],[0.05 0.05]);
% %     %ScalpData = [delta_RPL, theta_RPL, alpha_RPL, beta_RPL, gamma_RPL1];
% %     %ScalpData = [all_sessions_delta_CoV_corr, all_sessions_theta_CoV_corr, all_sessions_alpha_CoV_corr, all_sessions_beta_CoV_corr, all_sessions_gamma1_CoV_corr];
% %     %ScalpData = [all_sessions_delta_RPL, all_sessions_theta_RPL, all_sessions_alpha_RPL, all_sessions_beta_RPL, all_sessions_gamma1_RPL,all_sessions_gamma2_RPL];
% %     ScalpData = [all_sessions_beta_RPL_corr, all_sessions_gamma_RPL_corr];
% % 
% %     for tpt = 1:length(freq_intervals)
% %        axes(Corr_plot(tpt));
% %        topoplot(ScalpData(EEG_channel_nos,tpt), EEG.chanlocs(EEG_channel_nos),'maplimits', [-1 1],'style','both',...    
% %             'electrodes','ptsnumbers','plotchans',1:56,'plotrad', 0.55,'headrad',0.5,...
% %             'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off','conv','on');
% %             %'emarker2',{emarker_chans,'.','k',16,2});
% %         %topoplot(ScalpData(:,tpt),EEG.chanlocs(EEG_channel_nos),'plotrad', 0.55);
% %         title(freq_intervals(tpt),'FontSize',18);
% %         %axis([-0.55 0.55 -1.82 1.82]);
% %     end
% %     cbar_axes = colorbar('location','SouthOutside','XTick',[-0.5 0 0.5],'XTickLabel',{'-0.5','0','0.5'});
% %     xlabel(cbar_axes,'Bandwise RPL corr','FontSize',12);
% %     mtit(Subject_name,'FontSize',12);


 end                          

display(subject_wise_correlation_values)
display(subject_wise_correlation_p_values)
subject_wise_average_bmi_accuracy = [91 85 80 92 65 77];

%% Plot correlation with Gamma RPL for all subjects
figure('Position',[-1500 200 4.5*116 2*116]); 
%plot(1:6,flip(subject_wise_correlation_values),'ok','MarkerFaceColor','k')
bar(flip(subject_wise_correlation_values),'FaceColor',[0 0 1],'EdgeColor',[0 0 0],'LineWidth',1,'barwidth',0.5)
ylim([-0.6 0.6]); 
xlim([0.5 6.5]);
set(gca,'Ytick',[-0.5 0 0.5],'YtickLabel',{'-0.5' '0' '0.5'},'FontSize',poster_font_size);
set(gca,'Xtick',1:6,'Xticklabel',{'S1' 'S2' 'S3' 'S4' 'S5' 'S6'},'FontSize',poster_font_size);
set(gca,'Xgrid','on');
set(gca,'Ygrid','on');
xlabel('All subjects','FontSize',poster_font_size);
title({'Correlation (\rho) of BMI accuracy with frontal gamma RPL'},'FontSize',poster_font_size); 







