% Program to evaluate BMI performance across all subjects and sessions of clinical study - NRI Project
% Author: Nikunj A. Bhagat, Graduate Student, University of Houston, 
% Contact: nbhagat08[at]gmail.com
% Date: October 12, 2015
%------------------------------------------------------------------------------------------------------------------------
% Revisions
% 10/12/15 - Evolved from import_markers_for_source_localization.m and acc_err_plots.m
%                    
%                    
%------------------------------------------------------------------------------------------------------------------------
clear;
paper_font_size = 11;
x_axis_deviation = [0.15,0.05,-0.05,-0.15];
directory = 'D:\NRI_Project_Data\Clinical_study_Data\';
Subject_names = {'S9023','S9020','S9018','S9017','S9014','S9012','S9011','S9010','S9009','S9007'};
Impaired_hand = {'R', 'L', 'L','R', 'R', 'R','L', 'L', 'L', 'R'};
Subject_numbers = [9023,9020,9018,9017,9014,9012,9011,9010,9009,9007];
Subject_labels = {'S10','S9','S8','S7','S6','S5','S4','S3','S2','S1'};
Subject_velocity_threshold = [1.44,1.5,1.5,1.1,1.17,1.28,1.16,1.03,1.99,1.19];
Cond_num = [1, 3, 1, 1, 1, 1, 1, 1, 3, 1];
Block_num = [160, 160, 160, 160, 160, 160, 170, 160, 150, 160];
blue_color = [0 0.64  1];%[0, 114, 178]./255;
orange_color = [1 0.3 0]; %[213, 94, 0]./255;
purple_color = [0.3   0   0.6];
green_color = [0   0.5    0];
pink_color = [255 28 213]./255;
dark_blue_color = [53 0 255]./255;

% Subject groupings based on pre- and post- clinical scores
Subjects_all = 1:length(Subject_numbers);
% Subjects_severe_moderate = [2,7,9];
% Subjects_moderate_mild = [1,3:6, 8, 10];
% Subjects_FMA_above_MCID = [10, 8, 6, 3, 2, 1];
% Subjects_FMA_below_MCID = [9, 7, 5];
% Subjects_ARAT_above_MCID = [10, 8, 5, 4, 3]; %[10, 5, 4, 3];
% Subjects_ARAT_below_MCID = [9, 7, 6, 2, 1]; %[9, 8, 7, 6, 2, 1];

ACC_marker_color = {'-sy','-^m','--ok','-sb','--ob','-sk','-vr','--^m','--ok','-sb'};
Latency_marker_color = {'^m','ok','sb','ob','sk','vr','^m','ok','sb'};
TPR_marker_color = {'--ok','--sk','--ok','--*k','--^k'};
FPR_marker_color = {'--ok','-sk','-ok','*k','^k'};
Marker_face_color = {'y','m','k','b','b','k','r','m','k','b'};
Sig_line_color = {'--k','--k','--k','--k','--k','--k'};
%likert_marker_color = {'-ok','-sk','-ok','--^k','-vk'};
h_acc = zeros(length(Subject_names),1);
h_tpr = zeros(length(Subject_names),1);
h_fpr = zeros(length(Subject_names),1);
h_intent = zeros(length(Subject_names),1);
h_likert = zeros(length(Subject_names),1);

 EMG_channel_nos = [17 22 41 42 45 46 51 55];
% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 

baseline_int = [-2.5 -2.25]; 
segment_resting_eegdata = 0; 
plot_c3i_poster_plot = 0; 
plot_sfn_poster_plot = 0;  % use for paper
plot_movement_smoothness = 0;
compute_statistics = 0;
perform_posthoc_RP_analysis = 0;
perform_posthoc_EMG_analysis = 0; 
correlate_RP_changes = 1;
compute_trial_averages = 0;
compute_statistics_on_kinematics = 0;
plot_bilateral_EMG_traces =  0;

if segment_resting_eegdata == 1
   for subj_n = 1:length(Subject_names)               
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
        subject_study_data = dlmread(results_filename,',',1,0);
        % Array fields - subject_study_data
        % 1 - Session_nos	2 - Block_number	3- Start_of_trial	4 - End_of_trial	5 - Valid_or_catch	
        % 6 - Intent_detected	7 - Time_to_trigger     8 -  Number_of_attempts     9 - EEG_decisions	10 - EEG_EMG_decisions	
        % 11 - MRCP_slope	12 - MRCP_neg_peak	13 - MRCP_AUC	14 - MRCP_mahalanobis	15 - feature_index	
        % 16 - Corrected_spatial_chan_avg_index     17 - Correction_applied_in_samples      18 - Likert_score	19 - Target     
        % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency 23 - EMG_decisions
        
        unique_session_nos = unique(subject_study_data(:,1));
        unique_session_nos = [1; 2; unique_session_nos]; 
        
        for ses_num = 11:length(unique_session_nos)
            eegfilesave_location = ['D:\NRI_Project_Data\Clinical_study_Data\Resting_EEG_data_Stroke\Subject_' Subject_names{subj_n} '\'];
            eegfile_location = ['D:\NRI_Project_Data\Clinical_study_Data\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(unique_session_nos(ses_num)) '\'];
            
            if (unique_session_nos(ses_num) == 1) || (unique_session_nos(ses_num) == 2)
                
                if (unique_session_nos(ses_num) == 1) 
                    unique_block_nos = [1 2 3 4]';  % change 
                elseif (unique_session_nos(ses_num) == 2)
                    unique_block_nos = [3 4 5 6]';  % change                    
                end
                    
                Cond_num = 1;   % change
                for block_num = 1:length(unique_block_nos)
                      eegfile_name = [Subject_names{subj_n} '_ses' num2str(unique_session_nos(ses_num))  '_cond' num2str(Cond_num) '_block' num2str(unique_block_nos(block_num))]; 
                      [ALLEEG, EEG, CURRENTSET, ALLCOM] = SegmentEEGRestData(Subject_names{subj_n},unique_session_nos(ses_num), unique_block_nos(block_num), 0, EMG_channel_nos, ...
                                                                                              eegfile_location, eegfile_name, eegfilesave_location, ALLEEG, EEG, CURRENTSET, ALLCOM); 
                      [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                      eeglab redraw;
                end
            else
                unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));
                for block_num = 1:length(unique_block_nos)
                        if unique_block_nos(block_num) > 9
                            eegfile_name = [Subject_names{subj_n} '_ses' num2str(unique_session_nos(ses_num))  '_closeloop_block00' num2str(unique_block_nos(block_num))]; 
                       else
                            eegfile_name = [Subject_names{subj_n} '_ses' num2str(unique_session_nos(ses_num))  '_closeloop_block000' num2str(unique_block_nos(block_num))]; 
                       end
                      
                      [ALLEEG, EEG, CURRENTSET, ALLCOM] = SegmentEEGRestData(Subject_names{subj_n},unique_session_nos(ses_num), unique_block_nos(block_num), 1, EMG_channel_nos, ...
                                                                                              eegfile_location, eegfile_name, eegfilesave_location, ALLEEG, EEG, CURRENTSET, ALLCOM); 
                      [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                      eeglab redraw;
                end
                
            end
        end
   end
end

if plot_sfn_poster_plot == 1
    all_subjects_bmi_performance = zeros(12,16,length(Subject_names));
    BMI_offline_performance = zeros(length(Subject_names),2); 
    Offline_decision_latency  = [];
    for subj_n = 1:length(Subject_names)
        classifier_filename = [directory    'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session2\' ...
                 Subject_names{subj_n} '_ses2_cond' num2str(Cond_num(subj_n)) '_block' num2str(Block_num(subj_n))...
                 '_performance_optimized_conventional_smart.mat'];
        
        load(classifier_filename);
        % Classifier with best accuracy
        [max_acc_val,max_acc_index] = max(Performance.eeg_accur); 
        
        BMI_offline_performance(subj_n,:) = [Performance.eeg_accur(max_acc_index), 100*(1 - Performance.eeg_specificity(max_acc_index))];                
        for cv_index = 1:Performance.CVO.NumTestSets
           Offline_decisions = Performance.All_eeg_decision{Performance.smart_opt_wl_ind}(2,cv_index); 
           Offline_decision_estimates = Performance.All_eeg_prob_estimates{Performance.smart_opt_wl_ind}(2,cv_index);
           Offline_decision_latency = [Offline_decision_latency; Offline_decision_estimates{1}(Offline_decisions{1}==1,3)];
        end
        
        bmi_performance = [];
%         bmi_performance_blockwise = [];
               
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
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
            ind_failed_catch_trials = find((session_performance(:,5) == 2) & (session_performance(:,6) == 1));
            ind_success_catch_trials = find((session_performance(:,5) == 2) & (session_performance(:,6) == 0));
            session_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR
            
            Session_accuracy = (length(ind_success_valid_trials) + length(ind_success_catch_trials))/(length(ind_valid_trials) + length(ind_catch_trials));
            
            time_to_trigger_success_valid_trials = session_performance(ind_success_valid_trials,7); %col 7 - Time to Trigger
            Session_Intent_per_min = 60./time_to_trigger_success_valid_trials;
            
            subject_intent_per_min  = [subject_intent_per_min; [unique_session_nos(ses_num).*ones(length(Session_Intent_per_min),1) Session_Intent_per_min]];
            
            session_latencies = session_performance(ind_success_valid_trials,22);
            session_latencies(session_latencies>1000) = [];
            session_latencies(session_latencies < -1000) = [];
            
            %Session_detection_latency_mean = mean(session_performance(ind_success_valid_trials,22));
            Session_detection_latency_mean = mean(session_latencies);
            %Session_detection_latency_std = std(session_performance(ind_success_valid_trials,22));
            Session_detection_latency_num_trials = length(session_latencies);
            
            Session_likert_mean = mean(session_performance(:,18));
            %Session_likert_std = std(session_performance(:,18));
            Session_likert_num_trials = size(session_performance,1);
            
            session_num_attempts = session_performance(ind_success_valid_trials,8);
%             disp(length(find(session_num_attempts == 0)))
            Session_num_attempts_mean = mean(session_num_attempts);
            Session_num_attempts_num_trials = length(session_num_attempts);
            
            unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));
            % Block wise performance
% %             for block_num = 1:length(unique_block_nos)                                                
% %                 block_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num) & ...
% %                                                                                                 subject_study_data(:,2) == unique_block_nos(block_num),:);
% %                 %ind_success_valid_trials = find((block_performance(:,5) == 1) & (block_performance(:,6) == 1));
% %                 %time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,7); %col 7 - Time to Trigger
% %                 %Block_Intent_per_min = 60./time_to_trigger_success_valid_trials;
% %                 
% %                 block_ind_valid_trials = find(block_performance(:,5) == 1);  % col 5 - Valid(1) or Catch(2)
% %                 block_ind_success_valid_trials = find((block_performance(:,5) == 1) & (block_performance(:,6) == 1)); % col 5 - Intent detected
% %                 block_TPR = length(block_ind_success_valid_trials)/length(block_ind_valid_trials);      % TPR
% % 
% %                 block_ind_catch_trials = find(block_performance(:,5) == 2);
% %                 block_ind_failed_catch_trials = find((block_performance(:,5) == 2) & (block_performance(:,6) == 1));
% %                 block_ind_success_catch_trials = find((block_performance(:,5) == 2) & (block_performance(:,6) == 0));
% %                 block_FPR = length(block_ind_failed_catch_trials)/length(block_ind_catch_trials); %FPR
% % 
% %                 block_accuracy = (length(block_ind_success_valid_trials) + length(block_ind_success_catch_trials))/...
% %                                                  (length(block_ind_valid_trials) + length(block_ind_catch_trials));
% %                                              
% %                 bmi_performance_blockwise = [bmi_performance_blockwise;...
% %                                                                            [subj_n unique_session_nos(ses_num) unique_block_nos(block_num) block_TPR block_FPR block_accuracy] ];
% %             end

%             bmi_performance = [bmi_performance;...
%                                             [subj_n unique_session_nos(ses_num) session_TPR session_FPR length(ind_valid_trials) length(ind_catch_trials)...
%                                                             mean(Session_Intent_per_min) std(Session_Intent_per_min) median(Session_Intent_per_min) Session_likert_mean Session_likert_std...
%                                                             Session_detection_latency_mean Session_detection_latency_std Session_accuracy]];   
            %        1          2                 3                  4                 5                        6                                                    7 
            % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
            %                    8                                          9                                                       10                              11
            % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
            %                   12                                                                    13                                                    
            % Session_detection_latency_mean    Session_detection_latency_std
            %                    14
            %       Session_accuracy
            
            bmi_performance = [bmi_performance;...
                                            [subj_n unique_session_nos(ses_num) session_TPR session_FPR length(ind_valid_trials) length(ind_catch_trials)...
                                                            mean(Session_Intent_per_min) std(Session_Intent_per_min) median(Session_Intent_per_min) Session_likert_mean Session_likert_num_trials...
                                                            Session_detection_latency_mean Session_detection_latency_num_trials Session_accuracy ...
                                                            Session_num_attempts_mean  Session_num_attempts_num_trials]];   

        end
        
        all_subjects_bmi_performance(:,:,subj_n) = bmi_performance;
    end
    
  
    %%
        % Determine t-statistics for 95% C.I.
        %http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s
        deg_freedom = length(Subject_numbers) - 1;
        t_value = tinv(1 - 0.05/2, deg_freedom);
    
        % all_subjects_bmi_performance has size (#of sessions, #variables, #subjects)
        all_subjects_session_wise_accuracy_mean = mean(squeeze(all_subjects_bmi_performance(:,14,Subjects_all)),2);
        all_subjects_session_wise_accuracy_std = std(squeeze(all_subjects_bmi_performance(:,14,Subjects_all)),[],2); 
%         all_subjects_session_wise_accuracy_SE = (t_value.*std(squeeze(all_subjects_bmi_performance(:,14,Subjects_all)),[],2))/sqrt( length(Subject_numbers)); 
        
        all_subjects_session_wise_FPR_mean = mean(squeeze(all_subjects_bmi_performance(:,4,Subjects_all)),2);
        all_subjects_session_wise_FPR_std = std(squeeze(all_subjects_bmi_performance(:,4,Subjects_all)),[],2);
        
        % http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/homepage.htm
        % http://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
              
        All_session_latencies_mean = squeeze(all_subjects_bmi_performance(:,12,Subjects_all));
        All_session_latencies_num_trials = squeeze(all_subjects_bmi_performance(:,13,Subjects_all));
        %all_subjects_sessions_wise_latency_mean = sum(All_session_latencies_mean.*All_session_latencies_num_trials,2)./sum(All_session_latencies_num_trials,2);
        %all_subjects_sessions_wise_latency_std = std(squeeze(all_subjects_bmi_performance(:,13,:)),[],2);
        all_subjects_sessions_wise_latency_mean = [];
        all_subjects_sessions_wise_latency_std = [];
        for ses_num = 1:size(All_session_latencies_mean,1)
            means = All_session_latencies_mean(ses_num,:);
            %means = All_session_latencies_mean(:);
            trials = All_session_latencies_num_trials(ses_num,:);
            %trials = All_session_latencies_num_trials(:);
            N = length(trials);
            weighted_means = sum(means.*trials)/sum(trials);
            weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));
            all_subjects_sessions_wise_latency_mean = [all_subjects_sessions_wise_latency_mean; weighted_means];
            all_subjects_sessions_wise_latency_std = [all_subjects_sessions_wise_latency_std; weighted_standard_deviation];
        end
        
        All_session_likert_mean = squeeze(all_subjects_bmi_performance(:,10,Subjects_all));
        All_session_likert_num_trials = squeeze(all_subjects_bmi_performance(:,11,Subjects_all));
%         all_subjects_sessions_wise_likert_mean = sum(All_session_likert_mean.*All_session_likert_num_trials,2)./sum(All_session_likert_num_trials,2);
%         all_subjects_sessions_wise_likert_std = std(squeeze(all_subjects_bmi_performance(:,13,:)),[],2);
        all_subjects_sessions_wise_likert_mean = [];
        all_subjects_sessions_wise_likert_std = [];
         for ses_num = 1:size(All_session_likert_mean,1)
            means = All_session_likert_mean(ses_num,:);
            %means = All_session_likert_mean(:);
            trials = All_session_likert_num_trials(ses_num,:);
            %trials = All_session_likert_num_trials(:);
            N = length(trials);
            weighted_means = sum(means.*trials)/sum(trials);
            weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));
            all_subjects_sessions_wise_likert_mean = [all_subjects_sessions_wise_likert_mean weighted_means];
            all_subjects_sessions_wise_likert_std = [all_subjects_sessions_wise_likert_std weighted_standard_deviation];
         end
        
        All_session_num_attempts_mean = squeeze(all_subjects_bmi_performance(:,15,Subjects_all));
        All_session_num_attempts_num_trials = squeeze(all_subjects_bmi_performance(:,16,Subjects_all));
        %all_subjects_sessions_wise_latency_mean = sum(All_session_latencies_mean.*All_session_latencies_num_trials,2)./sum(All_session_latencies_num_trials,2);
        %all_subjects_sessions_wise_latency_std = std(squeeze(all_subjects_bmi_performance(:,13,:)),[],2);
        all_subjects_sessions_wise_num_attempts_mean = [];
        all_subjects_sessions_wise_num_attempts_std = [];
        for ses_num = 1:size(All_session_num_attempts_mean,1)
            means = All_session_num_attempts_mean(ses_num,:);
            %means = All_session_latencies_mean(:);
            trials = All_session_num_attempts_num_trials(ses_num,:);
            %trials = All_session_latencies_num_trials(:);
            N = length(trials);
            weighted_means = sum(means.*trials)/sum(trials);
            weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));
            all_subjects_sessions_wise_num_attempts_mean = [all_subjects_sessions_wise_num_attempts_mean; weighted_means];
            all_subjects_sessions_wise_num_attempts_std = [all_subjects_sessions_wise_num_attempts_std; weighted_standard_deviation];
        end
        
        % Calculate mean+/-s.d. of accuracy, fpr, latency, likert scores overall subjects and sessions
        % Added on 11th July, 2018
        Ovracc = squeeze(all_subjects_bmi_performance(:,14,:));
        disp(['Overall BMI accuracy = ', num2str(mean(Ovracc(:))*100), ' +/- ', num2str(std(Ovracc(:))*100), ' %'])  
        
        Ovrfpr = squeeze(all_subjects_bmi_performance(:,4,:));
        disp(['Overall false positive rate = ', num2str(mean(Ovrfpr(:))*100), ' +/- ', num2str(std(Ovrfpr(:))*100), ' %']) 
        
        OvrLatency_mean = sum(sum(All_session_latencies_mean.*All_session_latencies_num_trials))/sum(All_session_latencies_num_trials(:)); %weighted mean
        N = length(All_session_latencies_num_trials(:));
        OvrLatency_std = sqrt(sum(sum(((All_session_latencies_mean - OvrLatency_mean).^2).*All_session_latencies_num_trials))/(((N-1)/N)*sum(All_session_latencies_num_trials(:))));
        disp(['Overall latency = ', num2str(OvrLatency_mean), ' +/- ', num2str(OvrLatency_std), ' ms']) 
        
        OvrLikert_mean = sum(sum(All_session_likert_mean.*All_session_likert_num_trials))/sum(All_session_likert_num_trials(:)); %weighted mean
        N = length(All_session_likert_num_trials(:));
        OvrLikert_std = sqrt(sum(sum(((All_session_likert_mean - OvrLikert_mean).^2).*All_session_likert_num_trials))/(((N-1)/N)*sum(All_session_likert_num_trials(:))));
        disp(['Overall likert score = ', num2str(OvrLikert_mean), ' +/- ', num2str(OvrLikert_std)])     
        
        OvrNumAttempts_mean = sum(sum(All_session_num_attempts_mean.*All_session_num_attempts_num_trials))/sum(All_session_num_attempts_num_trials(:)); %weighted mean
        N = length(All_session_num_attempts_num_trials(:));
        OvrNumAttempts_std = sqrt(sum(sum(((All_session_num_attempts_mean - OvrNumAttempts_mean).^2).*All_session_num_attempts_num_trials))/(((N-1)/N)*sum(All_session_num_attempts_num_trials(:))));
        disp(['Overall Num Attempts = ', num2str(OvrNumAttempts_mean), ' +/- ', num2str(OvrNumAttempts_std), ' ']) 
        
        Num_of_valid_trials = squeeze(all_subjects_bmi_performance(:,5,:));
        mean(Num_of_valid_trials(:));
        std(Num_of_valid_trials(:));
        
 %%       
        figure('Position',[300 5 5*116 6*116]); 
        Cplot = tight_subplot(4,1,[0.05 0.05],[0.1 0.05],[0.1 0.05]);
        
        axes(Cplot(1)); hold on;
        %h_acc = plot(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean,'-sk','MarkerFaceColor','k','LineWidth',1);
        h_acc = errorbar(1:size(all_subjects_bmi_performance,1),...
            100*all_subjects_session_wise_accuracy_mean, 100*all_subjects_session_wise_accuracy_std,'^k','MarkerFaceColor','w','LineWidth',1,'MarkerSize',6);
        h_p2  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,14,4),'-','Color',orange_color,'LineWidth',0.5);
        h_p1  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,14,2),'-','Color',blue_color,'LineWidth',0.5);
%         h_p3  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,14,10),'-r','LineWidth',0.5);
        
        ylim([0 110]); 
        xlim([0.5 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{' '});
%         set(gca,'Xtick',0:12,'Xticklabel',{'calibration', '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
%         set(gca,'Xtick',0,'Xticklabel','calibration');
        %title('Closed-loop BMI performance for ongoing clinical study','FontSize',paper_font_size);
        %title(Subject_names{subj_n},'FontSize',paper_font_size);
        %ylabel({'True Positive'; 'Rate (%)'},'FontSize',paper_font_size);
%         ylabel({'BMI Accuracy', 'Mean \pm S.D. (%)'},'FontSize',paper_font_size);
        ylabel('BMI Accuracy (%)','FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        %xlabel('Sessions','FontSize',10);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        mlr_acc = LinearModel.fit(1:size(all_subjects_bmi_performance,1),100*all_subjects_session_wise_accuracy_mean);
        corr_acc = corr((1:size(all_subjects_bmi_performance,1))',100*all_subjects_session_wise_accuracy_mean); 
        if mlr_acc.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(all_subjects_bmi_performance,1)]]*mlr_acc.Coefficients.Estimate;
            plot(gca,[1  size(all_subjects_bmi_performance,1)+0.5],line_regress,'--k','LineWidth',0.5); hold on;
%             text(9,20,{sprintf('y = (%.2f)x + %.1f %%',mlr_acc.Coefficients.Estimate(2),mlr_acc.Coefficients.Estimate(1))},'FontSize',paper_font_size,'Color','k');
%             text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf('r = %.1f*',corr_acc)},'FontSize',paper_font_size,'Color','r');
        end
        title('A. Session-wise BMI performance','FontSize',paper_font_size);
        hold off;
        
        axes(Cplot(2)); hold on;
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        h_acc2 = errorbar(1:size(all_subjects_bmi_performance,1),...
            100*all_subjects_session_wise_FPR_mean,100*all_subjects_session_wise_FPR_std,'vk','MarkerFaceColor','w','LineWidth',1,'MarkerSize',6);
        h_p2  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,4,4),'-','Color',orange_color,'LineWidth',0.5);
        h_p1  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,4,2),'-','Color',blue_color,'LineWidth',0.5);
%         h_p3  = plot(1:size(all_subjects_bmi_performance,1),100*all_subjects_bmi_performance(1:12,4,10),'-r','LineWidth',0.5);
        
        ylim([0 110]); 
        xlim([0.5 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{' '});
        ylabel({'False Positives (%)'},'FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        
        mlr_fpr = LinearModel.fit(1:size(all_subjects_bmi_performance,1),100*all_subjects_session_wise_FPR_mean);
        cor_fpr = corr((1:size(all_subjects_bmi_performance,1))',100*all_subjects_session_wise_FPR_mean);
        if mlr_fpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_fpr.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.5],line_regress,'--r','LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_fpr.Coefficients.Estimate(2))},'FontSize',paper_font_size,'Color','r');
            text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf('r = %.1f',corr_fpr)},'FontSize',paper_font_size,'Color','r');
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        axes(Cplot(3)); hold on;
        h_latency = errorbar(1:size(all_subjects_bmi_performance,1),...
                             all_subjects_sessions_wise_latency_mean, all_subjects_sessions_wise_latency_std,'ok','MarkerFaceColor','w','LineWidth',1,'MarkerSize',6);
        h_p2  = plot(1:size(all_subjects_bmi_performance,1),All_session_latencies_mean(:,4),'-','Color',orange_color,'LineWidth',0.5);
        h_p1  = plot(1:size(all_subjects_bmi_performance,1),All_session_latencies_mean(:,2),'-','Color',blue_color,'LineWidth',0.5);
        
        ylim([-200 200]); 
        xlim([0.5 12.5]);
        set(gca,'Ytick',[-200 0 200],'YtickLabel',{'-200' '0' '200'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{' '});
        ylabel('Detection Time (ms)','FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        %xlabel('Sessions','FontSize',10);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
         mlr_latency = LinearModel.fit(1:size(all_subjects_bmi_performance,1),all_subjects_sessions_wise_latency_mean/1000);
        if mlr_latency.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(all_subjects_bmi_performance,1)]]*mlr_latency.Coefficients.Estimate;
            plot(gca,[0  size(all_subjects_bmi_performance,1)+0.5],line_regress,'--r','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_latency.Coefficients.Estimate(2))},'FontSize',paper_font_size,'Color','r');
        end
         hold off;
               
        axes(Cplot(4));hold on;
        h_likert(subj_n) = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_likert_mean, all_subjects_sessions_wise_likert_std, 'ok','MarkerFaceColor','w','LineWidth',1,'MarkerSize',6);
        h_p2  = plot(1:size(all_subjects_bmi_performance,1),All_session_likert_mean(:,4),'-','Color',orange_color,'LineWidth',0.5);
        h_p1  = plot(1:size(all_subjects_bmi_performance,1),All_session_likert_mean(:,2),'-','Color',blue_color,'LineWidth',0.5);
        ylim([0 4]); 
        xlim([0.5 12.5]);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        set(gca,'Ytick',[1 2 3],'YtickLabel',{'1', '2', '3'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'},'FontSize',paper_font_size);
        ylabel('Subject Rating','FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        xlabel('Therapy sessions','FontSize',paper_font_size);
        mlr_likert = LinearModel.fit(1:size(all_subjects_bmi_performance,1),all_subjects_sessions_wise_likert_mean);
        corr_likert = corr((1:size(all_subjects_bmi_performance,1))',all_subjects_sessions_wise_likert_mean');
        if mlr_likert.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(all_subjects_bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
            plot(gca,[1  size(all_subjects_bmi_performance,1)+0.5],line_regress,'--k','LineWidth',0.5); hold on;
%             text(9,1,{sprintf('y = (%.2f)x + %.1f',mlr_likert.Coefficients.Estimate(2),mlr_likert.Coefficients.Estimate(1))},'FontSize',paper_font_size,'Color','k');
%             text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf('r = %.1f',corr_likert)},'FontSize',paper_font_size,'Color','r');
%               text(9,1,{sprintf('y = (%.2f)x + %.1f',mlr_likert.Coefficients.Estimate(2),mlr_likert.Coefficients.Estimate(1))},'FontSize',paper_font_size,'Color','k');
        end
        legend([h_p1, h_p2],{'P9', 'P7'},'Orientation','horizontal','Location','southeastoutside');
        hold off;
        
%         annotation('textbox',[0.9 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none','FontSize',paper_font_size);    
%         line([0.2, 0.2], [0 1], 'Color','k','LineWidth',1.5);
    
    
% %  commented on 4-28-16 ---for presentation 
% %      axes(Cplot(1)); hold on;
% %      [legend_h,object_h,plot_h,text_str] = ...
% %               legendflex([h_tpr(1), h_tpr(2)],{'S1 (132 trials/session)','S2 (121 trials/session)'},'ncol',2, 'ref',Cplot(1),...
% %                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
% %     hold off
% %                               
% %      axes(Cplot(2)); hold on;
% %      [legend_h,object_h,plot_h,text_str] = ...
% %               legendflex([h_fpr(1), h_fpr(2)],{'S1 (13 trials/session)','S2 (13 trials/session)'},'ncol',2, 'ref',Cplot(2),...
% %                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
% %       hold off


%      [legend_h,object_h,plot_h,text_str] = ...
%               legendflex([h_intent(1), h_intent(2)],{'S1','S2'},'ncol',2, 'ref',Cplot(2),...
%                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);

%%       Avg comparison plots
        figure('Position',[900 5 2.5*116 6*116]); 
        Cplot = tight_subplot(4,1,[0.05 0.05],[0.1 0.05],[0.2 0.05]);
        
        axes(Cplot(1)); hold on;
        bar1 = bar([1], [mean(BMI_offline_performance(:,1))],'BarWidth',0.5);
        bar1.FaceColor = [0.6 0.6 0.6];
        
        bar2 = bar([2], [mean(Ovracc(:))*100],'BarWidth',0.5);
        bar2.FaceColor = [1 1 1];
        
        [p,h] = ranksum(BMI_offline_performance(:,1),Ovracc(:)*100)
        errorbar([1,2],[mean(BMI_offline_performance(:,1)), mean(Ovracc(:))*100],...
                 [std(BMI_offline_performance(:,1)), std(Ovracc(:))*100],'.k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',4);
        ylim([0 110]); 
        xlim([0 3]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',[1 2],'XtickLabel',{' '},'FontSize',paper_font_size);
        ylabel('BMI Accuracy (%)','FontSize',paper_font_size);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        title('B. Overall BMI performance','FontSize',paper_font_size);
        hold off;
        
        axes(Cplot(2)); hold on;
        bar1 = bar([1], [mean(BMI_offline_performance(:,2))],'BarWidth',0.5);
        bar1.FaceColor = [0.6 0.6 0.6];
        
        bar2 = bar([2], [mean(Ovrfpr(:))*100],'BarWidth',0.5);
        bar2.FaceColor = [1 1 1];
        [p,h] = ranksum(BMI_offline_performance(:,2),Ovrfpr(:)*100)
        
        errorbar([1,2],[mean(BMI_offline_performance(:,2)), mean(Ovrfpr(:))*100],...
                 [std(BMI_offline_performance(:,2)), std(Ovrfpr(:))*100],'.k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',4);
        ylim([0 110]); 
        xlim([0 3]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',[1 2],'XtickLabel',{' '},'FontSize',paper_font_size);
        ylabel('False Positives (%)','FontSize',paper_font_size);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;                                
                                       
        axes(Cplot(3)); hold on;
        bar1 = bar([1], [1E3*mean(Offline_decision_latency)],'BarWidth',0.5);
        bar1.FaceColor = [0.6 0.6 0.6];
        
        bar2 = bar([2], [OvrLatency_mean],'BarWidth',0.5);
        bar2.FaceColor = [1 1 1];
        
        [p,h] = ranksum(Offline_decision_latency.*1E3,All_session_latencies_mean(:))
        
        errorbar([1,2],[1E3*mean(Offline_decision_latency), OvrLatency_mean],...
                 [1E3*std(Offline_decision_latency), OvrLatency_std],'.k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',4);
             
        ylim([-1600 100]); 
        xlim([0 3]);
        set(gca,'Ytick',[-1500 -1000 -500 0],'YtickLabel',{'-1.5' '-1' '-0.5' '0'},'FontSize',paper_font_size);
        set(gca,'Xtick',[1 2],'XtickLabel',{' '},'FontSize',paper_font_size);
        ylabel('Detection Time (s)','FontSize',paper_font_size);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;                      
               
        axes(Cplot(4));hold on;
        bar2 = bar([2], [OvrLikert_mean],'BarWidth',0.5);
        bar2.FaceColor = [1 1 1];        
        errorbar([2],[OvrLikert_mean],OvrLikert_std,'.k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',4);
             
        ylim([0 4]); 
        xlim([0 3]);
        set(gca,'Ytick',[1 2 3],'YtickLabel',{'1', '2', '3'},'FontSize',paper_font_size);
        set(gca,'Xtick',[1 2],'XtickLabel',{'calibration','online'},'FontSize',paper_font_size);
        ylabel('Subject Rating','FontSize',paper_font_size);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');                        
        text(0.9,2,'NA','FontSize',paper_font_size,'Color','k');
        hold off;
        
%         annotation('textbox',[0.9 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none','FontSize',paper_font_size);    
%         line([0.2, 0.2], [0 1], 'Color','k','LineWidth',1.5);
    
    
% %  commented on 4-28-16 ---for presentation 
% %      axes(Cplot(1)); hold on;
% %      [legend_h,object_h,plot_h,text_str] = ...
% %               legendflex([h_tpr(1), h_tpr(2)],{'S1 (132 trials/session)','S2 (121 trials/session)'},'ncol',2, 'ref',Cplot(1),...
% %                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
% %     hold off
% %                               
% %      axes(Cplot(2)); hold on;
% %      [legend_h,object_h,plot_h,text_str] = ...
% %               legendflex([h_fpr(1), h_fpr(2)],{'S1 (13 trials/session)','S2 (13 trials/session)'},'ncol',2, 'ref',Cplot(2),...
% %                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
% %       hold off


%      [legend_h,object_h,plot_h,text_str] = ...
%               legendflex([h_intent(1), h_intent(2)],{'S1','S2'},'ncol',2, 'ref',Cplot(2),...
%                                   'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
 %% Condition B plots 
       
%         axes(Cplot(1)); hold on;
%         %h_acc = plot(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean,'-sk','MarkerFaceColor','k','LineWidth',1);
%         h_acc = errorbar(1-0.25:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean, 100*all_subjects_session_wise_accuracy_std,'^k','MarkerFaceColor','w','LineWidth',1,'MarkerSize',8);
%                
%         axes(Cplot(2)); hold on;
%         %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
%         h_acc2 = errorbar(1-0.25:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_FPR_mean, 100*all_subjects_session_wise_FPR_std,'vk','MarkerFaceColor','w','LineWidth',1,'MarkerSize',8);
%                 
%         axes(Cplot(3)); hold on;
%         h_latency = errorbar(1-0.25:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_latency_mean, all_subjects_sessions_wise_latency_std,'ok','MarkerFaceColor','w','LineWidth',1,'MarkerSize',8);
%                       
%         axes(Cplot(4));hold on;
%         h_likert(subj_n) = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_likert_mean, all_subjects_sessions_wise_likert_std, 'ok','MarkerFaceColor','w','LineWidth',1,'MarkerSize',8);
%        
    
%% Correlation plots 

%    Change_BMI_accuracy = flip((squeeze(all_subjects_bmi_performance(12,14,:)) - squeeze(all_subjects_bmi_performance(1,14,:))).*100); % Outlier leads to significance
%    Change_BMI_accuracy_last_6ses = flip((squeeze(all_subjects_bmi_performance(12,14,:)) - squeeze(all_subjects_bmi_performance(7,14,:))).*100);
%    Change_BMI_accuracy_last_5ses = flip((squeeze(all_subjects_bmi_performance(12,14,:)) - squeeze(all_subjects_bmi_performance(8,14,:))).*100);
%    
%    Avg_BMI_accuracy = flip(mean(squeeze(all_subjects_bmi_performance(1:12,14,:)))'*100);
%    Avg_BMI_accuracy_last_6ses = flip(mean(squeeze(all_subjects_bmi_performance(7:12,14,:)))'*100);
%    Avg_BMI_accuracy_last_5ses = flip(mean(squeeze(all_subjects_bmi_performance(8:12,14,:)))'*100);
% 
%    Change_FMA_1wk = [5, 1, -1, 3, 8, 4, 5, 4, 7, -1]'; % S9007 to 9023
%    Change_ARAT_1wk = [12, 0, 9, 0, 5, 6, 7, 10, 1, 2]'; % S9007 to 9023
%    
%    FMA_responders = [1, 4, 5, 6, 7, 8, 9];
%    ARAT_responders = [1, 3, 5, 6, 7, 8];
%    
%    FMA_responders_MCID = [1, 5, 7, 9];
%    ARAT_responders_MCID = [1, 3, 5, 6, 7, 8];
%    
%    Change_Avg_speed = [9.09, 11.41, 9.83, -1.20, 7.70, 2.90, 7.52, -4.24, 6.30, -2.55]'; % S9007 to 9023
%    Change_SAL = [0.18, 2.20, 0.11, -0.06, 0.20, 0.18, 0.07, -0.64, 0.68, 0.06]'; % S9007 to 9023

   %%
%    Independent_variable = Change_Avg_speed;    
%    %Independent_var_name =  '\DeltaBMI Accuracy = difference in BMI accuracies at start and end of therapy';
%    Independent_var_name =  '\DeltaAvg. Speed (deg/s) = difference in joint speed at start and end of therapy';
%    Ind_var_min = min(Independent_variable) - 0.1*min(Independent_variable);     
%    Ind_var_max = max(Independent_variable) + 0.1*max(Independent_variable);
%    Ind_var_range = [Ind_var_min  Ind_var_max]; % range for Xlim
%    [FMA_r_corr, FMA_p_corr] = corrcoef(Change_FMA_1wk(FMA_responders), Independent_variable(FMA_responders));
%    [ARAT_r_corr, ARAT_p_corr] = corrcoef(Change_ARAT_1wk(ARAT_responders), Independent_variable(ARAT_responders));
%    
%    
%    
%    figure;    
%    subplot(1,2,1); hold on; grid on;
%    plot(Independent_variable(FMA_responders), Change_FMA_1wk(FMA_responders), 'or'); 
%    ylim([0 10]); 
%    xlim(Ind_var_range);
%    ylabel('Change in Fugl-Meyer 1-wk post-tt from baseline2');
%    xlabel(Independent_var_name);
%    title(['Corrcoef = ' num2str(FMA_r_corr(1,2))  ' , p = ' num2str(FMA_p_corr(1,2))]);
%    
%    subplot(1,2,2); hold on; grid on;
%    plot(Independent_variable(ARAT_responders_MCID), Change_ARAT_1wk(ARAT_responders_MCID), 'or'); 
%    ylim([-1 12]); 
%    xlim(Ind_var_range);
%    ylabel('Change in ARAT 1-wk post-tt');
%    xlabel(Independent_var_name);
%    title(['Corrcoef = ' num2str(ARAT_r_corr(1,2))  ' , p = ' num2str(ARAT_p_corr(1,2))]);
%    
%    
% %    mlr_fma = LinearModel.fit(Independent_variable, Change_FMA_1wk);
% %    line_regress = [ones(2,1) [-10; 50]]*mlr_fma.Coefficients.Estimate;
% %    plot([-10  50],line_regress,'--b','LineWidth',0.5); hold on;
   
 
end

if plot_c3i_poster_plot == 1
     figure('Position',[700 100 7*116 2.5*116]); 
     Cplot = tight_subplot(1,1,[0.05 0.02],[0.25 0.1],[0.1 0.05]);
     Subject_wise_performance = [];
     
    for subj_n = 1:length(Subject_names)
        bmi_performance = [];
        bmi_performance_blockwise = [];
               
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
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
                
        axes(Cplot(1)); hold on;
        h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),ACC_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
        %plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),'-ok','LineWidth',1);
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor','none','LineWidth',1);
        ylim([0 110]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 90 100],'YtickLabel',{'0' '90' '100'});
        %set(gca,'Xtick',1:12,'Xticklabel',{' '});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        xlabel('Sessions','FontSize',10);
        title('Closed-loop BMI performance for ongoing clinical study (6 subjects)','FontSize',paper_font_size);
        ylabel({'BMI Accuracy (%)'},'FontSize',paper_font_size);
         %xlabel('Sessions','FontSize',10);
         
        mlr_acc = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,14));
        if mlr_acc.coefTest <= 0.05
            %line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_tpr.Coefficients.Estimate;
            %plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
            %text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_tpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
            Subject_labels{subj_n} = [Subject_labels{subj_n} '*'];
        end
  
%         if subj_n == 1
%             %legend([h_tpr h_fpr],'TPR','FPR','Location','SouthEast');
%              [legend_h,object_h,plot_h,text_str] = ...
%                             legendflex([h_tpr(1), h_fpr(1)],{'True Positives','False Positives'},'ncol',2, 'ref',Cplot(1),...
%                                                 'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
%     %         set(object_h(3),'FaceAlpha',0.5);
%     %         set(object_h(4),'FaceAlpha',0.5);
%             set(gca,'Xgrid','on');
%             set(gca,'Ygrid','on');
%         end
        set(gca,'Xgrid','off');
        set(gca,'Ygrid','on');
        hold off;
        
%         axes(Cplot(2)); hold on;
%         h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
%         ylim([-10 100]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
%         set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
%         ylabel({'False Positive'; 'Rate (%)'},'FontSize',paper_font_size);
%          xlabel('Sessions','FontSize',10);
%         mlr_tpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,3));
%         
%         mlr_fpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,4));
%         if mlr_fpr.coefTest <= 0.05
% %             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_fpr.Coefficients.Estimate;
% %             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
% %             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_fpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%             Subject_labels{subj_n} = [Subject_labels{subj_n} '^'];
%         end
% %         if subj_n == 1
% %             %legend([h_tpr h_fpr],'TPR','FPR','Location','SouthEast');
% %              [legend_h,object_h,plot_h,text_str] = ...
% %                             legendflex([h_tpr(1), h_fpr(1)],{'True Positives','False Positives'},'ncol',2, 'ref',Cplot(1),...
% %                                                 'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
% %     %         set(object_h(3),'FaceAlpha',0.5);
% %     %         set(object_h(4),'FaceAlpha',0.5);
% %             set(gca,'Xgrid','on');
% %             set(gca,'Ygrid','on');
% %         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;
        
%         axes(Cplot(3)); hold on;
% %         hbox_axes = boxplot(subject_intent_per_min(:,2),subject_intent_per_min(:,1),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o','colors',boxplot_color{subj_n}); % symbol - Outliers take same color as box
% %         set(hbox_axes(6,1:size(bmi_performance,1)),'Color',boxplot_color{subj_n});
% %         set(hbox_axes(7,1:size(bmi_performance,1)),'MarkerSize',4);
% %         set(hbox_axes,'LineWidth',1);
%         h_intent(subj_n) = plot(1:size(bmi_performance,1), bmi_performance(:,9),likert_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
%         ylim([0 20]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[0 10 20],'YtickLabel',{'0' '10' '20'});
%         set(gca,'Xtick',1:12,'Xticklabel',' ');
%        ylabel({'No. of Intents'; 'detected per min.'; '(median)'},'FontSize',paper_font_size);
%         plot1_pos = get(Cplot(1),'Position');
%         boxplot_pos = get(gca,'Position');
%         set(gca,'Position',[plot1_pos(1) boxplot_pos(2) plot1_pos(3) boxplot_pos(4)]);
%         mlr_intents = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,9));
%         if mlr_intents.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_intents.Coefficients.Estimate;
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_intents.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;
        
% Commented here on 9-8-2016
%         axes(Cplot(2));hold on;
%         h_latency = errorbar([1:size(bmi_performance,1)]+x_axis_deviation(subj_n), bmi_performance(:,12)./1000, bmi_performance(:,13)./1000, Latency_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
%         ylim([-1 1]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[-1 -0.5 0 0.5 1],'YtickLabel',{'-1' '-0.5' '0' '0.5' '1'});
%         set(gca,'Xtick',1:12,'Xticklabel',{' '});
%         ylabel({'Intent detection';'latency (sec.)'},'FontSize',paper_font_size);
%         %xlabel('Sessions','FontSize',10);
%         plot1_pos = get(Cplot(1),'Position');
%         latencyplot_pos = get(gca,'Position');
%         set(gca,'Position',[plot1_pos(1) latencyplot_pos(2) plot1_pos(3) latencyplot_pos(4)]);
%         mlr_latency = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,12));
%         if mlr_latency.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_latency.Coefficients.Estimate;
%             %plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
%             %text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_latency.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','off');
%         set(gca,'Ygrid','on');
%         hold off;
%         
% %         figure;
% %         
% %         [f1,x1] = hist(subject_study_data(:,22),-500:10:500);
% %         h_patch(1) = jbfill(x1,f1./trapz(x1,f1), zeros(1,length(-500:10:500)),'r','r',1,0.5);
%         
%         axes(Cplot(3));hold on;
%         h_likert(subj_n) = errorbar([1:size(bmi_performance,1)]+x_axis_deviation(subj_n), bmi_performance(:,10), bmi_performance(:,11), Latency_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
%         ylim([0 4]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[1 2 3],'YtickLabel',{'1' '2' '3'});
%         set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
%         ylabel({'Subject rating'; '(Likert scale)'},'FontSize',paper_font_size);
%         xlabel('Sessions','FontSize',10);
%         mlr_likert = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,10));
%         if mlr_likert.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
%             %plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
%             %text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_likert.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','off');
%         set(gca,'Ygrid','on');
%         hold off;
             
        
        
        
     Subject_wise_performance = [Subject_wise_performance; bmi_performance];   
    end
    
%    legendflex(fliplr(h_tpr')',fliplr(Subject_labels),'ncol',3, 'ref',Cplot(1),'anchor',[3 1],'buffer',[0 5],'box','on','xscale',1,'padding',[2 1 1], 'title', 'Subject labels');
    %annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
    Group_mean = mean(Subject_wise_performance);
    Group_std = std(Subject_wise_performance);
    mean_Subject_wise_perf_mean = [];
    mean_Subject_wise_perf_std = [];
    mean_Subject_wise_FPR_mean = [];
    max_Subject_wise_intrasession_variability = [];
     for subj_n = 1:length(Subject_names)
         mean_Subject_wise_perf_mean = [mean_Subject_wise_perf_mean;...
                                                                mean(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,15))]; % Accuracy
         mean_Subject_wise_perf_std = [mean_Subject_wise_perf_std;...
                                                                std(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,15))];
         %max_Subject_wise_intrasession_variability = [max_Subject_wise_intrasession_variability;
         %                                                       max(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,16))];
         mean_Subject_wise_FPR_mean = [mean_Subject_wise_FPR_mean;
                                                                     mean(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,4))];    % FPR
     end
    %figure; boxplot(Subject_wise_performance(:,4),Subject_wise_performance(:,1))
end

if plot_movement_smoothness == 1
    load('C:\NRI_BMI_Mahi_Project_files\All_Subjects\Smoothness_coefficient_S9007-9010.mat');
    kin_smoothness = s7_s9_s10_smth;
    figure('Position',[700 100 7*116 2.5*116]);
    Cplot = tight_subplot(1,1,[0.05 0.02],[0.15 0.05],[0.1 0.05]);
    axes(Cplot(1));hold on;
    for subj_n = 2: length(Subject_names)
        kin_performance = kin_smoothness(kin_smoothness(:,1) == Subject_numbers(subj_n),:);
        errorbar([0:12]+x_axis_deviation(subj_n), kin_performance(end-12:end,7), kin_performance(end-12:end,8), Latency_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
    end
    ylim([-1 1.2]); 
    xlim([-0.5 12.5]);
    set(gca,'Ytick',[-1 0 1],'YtickLabel',{'-1' '0' '1'});
    set(gca,'Xtick',0:12,'Xticklabel',{'Baseline' '1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
    %ylabel({'Subject rating'; '(Likert scale)'},'FontSize',paper_font_size);
    xlabel('Sessions','FontSize',paper_font_size);
    set(gca,'Xgrid','off');
    set(gca,'Ygrid','on');
    hold off;       
end

if compute_statistics_on_kinematics == 1
        
%    Avg_speed = [22.60	11.80	12.97	16.52	10.14	16.18	9.47	27.74	14.81	11.55;
%                 31.69	23.21	22.80	24.22	13.03	23.71	15.78	25.19	13.61	7.31];
%             
%    SAL = [-2.10	-4.89	-2.40	-2.12	-2.31	-2.30	-2.72	-1.92	-2.38	-2.78;
%           -1.91	-2.68	-2.30	-1.92	-2.13	-2.23	-2.04	-1.86	-2.44	-3.42];
%       
%    Numpeaks = [1.58	2.68	2.62	1.59	2.29	2.03	2.63	1.16	2.39	2.39;
%                1.03	1.83	1.98	1.13	1.85	1.54	1.23	1.05	2.56	3.63];
%            
%    time_to_peak = [0.42	0.18	0.25	0.43	0.38	0.39	0.30	0.48	0.27	0.27;
%                    0.60	0.30	0.33	0.53	0.44	0.45	0.63	0.59	0.27	0.14];
   
   load('D:\NRI_Project_Data\R_analysis\Kinematic Analysis\MasterAnalysis.mat');            
   Avg_speed = zeros(2,10);
   SAL = zeros(2,10);
   Numpeaks = zeros(2,10);
   time_to_peak = zeros(2,10);
   
   % Recompute metrics based on sessions 1 and 2
   Table_KinbySess = KinbySess.Avgs;
   Avg_KinbySess = Table_KinbySess((Table_KinbySess.calc == 'avg'),:);
   
   for subj_n = 1:length(Subject_numbers)
       
       subjectwise_kin_pre = Avg_KinbySess((Avg_KinbySess.subj_num == num2str(Subject_numbers(subj_n))) & ((Avg_KinbySess.TR_sess == 1) | (Avg_KinbySess.TR_sess == 2)),:);
       subjectwise_kin_post = Avg_KinbySess((Avg_KinbySess.subj_num == num2str(Subject_numbers(subj_n))) & ((Avg_KinbySess.TR_sess == 11) | (Avg_KinbySess.TR_sess == 12)),:);

       Avg_speed(1,subj_n) = sum(subjectwise_kin_pre.avg_spd.*subjectwise_kin_pre.n_trls)/sum(subjectwise_kin_pre.n_trls);
       Avg_speed(2,subj_n) = sum(subjectwise_kin_post.avg_spd.*subjectwise_kin_post.n_trls)/sum(subjectwise_kin_post.n_trls);

       SAL(1,subj_n) = sum(subjectwise_kin_pre.sal.*subjectwise_kin_pre.n_trls)/sum(subjectwise_kin_pre.n_trls);
       SAL(2,subj_n) = sum(subjectwise_kin_post.sal.*subjectwise_kin_post.n_trls)/sum(subjectwise_kin_post.n_trls);

       Numpeaks(1,subj_n) = sum(subjectwise_kin_pre.num_pks.*subjectwise_kin_pre.n_trls)/sum(subjectwise_kin_pre.n_trls);
       Numpeaks(2,subj_n) = sum(subjectwise_kin_post.num_pks.*subjectwise_kin_post.n_trls)/sum(subjectwise_kin_post.n_trls);

       time_to_peak(1,subj_n) = sum(subjectwise_kin_pre.t_pk1.*subjectwise_kin_pre.n_trls)/sum(subjectwise_kin_pre.n_trls);
       time_to_peak(2,subj_n) = sum(subjectwise_kin_post.t_pk1.*subjectwise_kin_post.n_trls)/sum(subjectwise_kin_post.n_trls);
   end
   
   mean(Avg_speed,2)
   std(Avg_speed,[],2)
   median(Avg_speed,2)
   [p,h] = signrank(Avg_speed(2,:)', Avg_speed(1,:)', 'tail','right')

   mean(SAL,2)
   std(SAL,[],2)
   median(SAL,2)
   [p,h] = signrank(SAL(2,:)', SAL(1,:)', 'tail','right')
   
   mean(Numpeaks,2)
   std(Numpeaks,[],2)
   median(Numpeaks,2)
   [p,h] = signrank(Numpeaks(2,:)', Numpeaks(1,:)', 'tail','left')

   mean(time_to_peak,2)
   std(time_to_peak,[],2)
   median(time_to_peak,2)
   [p,h] = signrank(time_to_peak(2,:)', time_to_peak(1,:)', 'tail','right')

   figure('Position',[300 5 6*116 2.5*116]); 
   Splot = tight_subplot(1,4,[0.1 0.05],[0.15 0.1],[0.1 0.1]);
   group_names = {'Start';'End'};
   group_positions = [1 2];
   
   axes(Splot(1)); hold on; grid on;
   hbox = boxplot(Avg_speed','labels', group_names, 'positions',group_positions,'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
   set(hbox,'LineWidth',1);
   set(hbox(5,1),'Color',blue_color);         % Box
   set(hbox(5,2),'Color',orange_color);
   set(hbox(6,1),'Color',blue_color);          % Median
   set(hbox(6,2),'Color',orange_color);
   set(hbox(1,1),'LineStyle','-','Color',blue_color);      % Top whisker line
   set(hbox(1,2),'LineStyle','-','Color',orange_color);      % Top whisker line
   set(hbox(2,1),'LineStyle','-','Color',blue_color);      % Bottom whisker line
   set(hbox(2,2),'LineStyle','-','Color',orange_color);      % Bottom whisker line
   set(hbox(3,1),'Color',blue_color);          % Top whisker bar
   set(hbox(3,2),'Color',orange_color);
   set(hbox(4,1),'Color',blue_color);          % Bottom whisker bar
   set(hbox(4,2),'Color',orange_color);
   if size(hbox,1) > 6
        set(hbox(7,1),'MarkerEdgeColor',blue_color,'MarkerSize',6); 
        set(hbox(7,2),'MarkerEdgeColor',orange_color,'MarkerSize',6);
   end
   set(gca,'YLim',[0 40], 'YTick', [0 15 30], 'YTickLabel', {'0' '15' '30'}, 'FontSize', paper_font_size-1);
   title('Avg. Speed (deg/s)','FontSize',paper_font_size-1,'FontWeight','normal');
   sigstar({group_positions},[0.05]); 
   
   axes(Splot(2)); hold on; grid on;
   hbox = boxplot(SAL','labels', group_names, 'positions',group_positions,'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
   set(hbox,'LineWidth',1);
   set(hbox(5,1),'Color',blue_color);         % Box
   set(hbox(5,2),'Color',orange_color);
   set(hbox(6,1),'Color',blue_color);          % Median
   set(hbox(6,2),'Color',orange_color);
   set(hbox(1,1),'LineStyle','-','Color',blue_color);      % Top whisker line
   set(hbox(1,2),'LineStyle','-','Color',orange_color);      % Top whisker line
   set(hbox(2,1),'LineStyle','-','Color',blue_color);      % Bottom whisker line
   set(hbox(2,2),'LineStyle','-','Color',orange_color);      % Bottom whisker line
   set(hbox(3,1),'Color',blue_color);          % Top whisker bar
   set(hbox(3,2),'Color',orange_color);
   set(hbox(4,1),'Color',blue_color);          % Bottom whisker bar
   set(hbox(4,2),'Color',orange_color);
   if size(hbox,1) > 6
        set(hbox(7,1),'MarkerEdgeColor',blue_color,'MarkerSize',6); 
        set(hbox(7,2),'MarkerEdgeColor',orange_color,'MarkerSize',6);
   end
   set(gca,'YLim',[-6 0.25], 'YTick', [-5 -2.5 0], 'YTickLabel', {'-5' '-2.5' '0'}, 'FontSize', paper_font_size-1);
   title({'Spectral Arc Length'},'FontSize',paper_font_size-1,'FontWeight','normal');   
   sigstar({group_positions},[0.05]); 
   
   axes(Splot(3)); hold on; grid on;
   hbox = boxplot(Numpeaks','labels', group_names, 'positions',group_positions,'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
   set(hbox,'LineWidth',1);
   set(hbox(5,1),'Color',blue_color);         % Box
   set(hbox(5,2),'Color',orange_color);
   set(hbox(6,1),'Color',blue_color);          % Median
   set(hbox(6,2),'Color',orange_color);
   set(hbox(1,1),'LineStyle','-','Color',blue_color);      % Top whisker line
   set(hbox(1,2),'LineStyle','-','Color',orange_color);      % Top whisker line
   set(hbox(2,1),'LineStyle','-','Color',blue_color);      % Bottom whisker line
   set(hbox(2,2),'LineStyle','-','Color',orange_color);      % Bottom whisker line
   set(hbox(3,1),'Color',blue_color);          % Top whisker bar
   set(hbox(3,2),'Color',orange_color);
   set(hbox(4,1),'Color',blue_color);          % Bottom whisker bar
   set(hbox(4,2),'Color',orange_color);
   if size(hbox,1) > 6
        set(hbox(7,1),'MarkerEdgeColor',blue_color,'MarkerSize',6); 
        set(hbox(7,2),'MarkerEdgeColor',orange_color,'MarkerSize',6);
   end
   set(gca,'YLim',[0 5], 'YTick', [0 2 4], 'YTickLabel', {'0' '2' '4'}, 'FontSize', paper_font_size-1);
   title('Num. of Peaks','FontSize',paper_font_size-1,'FontWeight','normal');
   sigstar({group_positions},[0.01]); 
   
   axes(Splot(4)); hold on; grid on;
   hbox = boxplot(time_to_peak','labels', group_names, 'positions',group_positions,'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o'); % symbol - Outliers take same color as box
   set(hbox,'LineWidth',1);
   set(hbox(5,1),'Color',blue_color);         % Box
   set(hbox(5,2),'Color',orange_color);
   set(hbox(6,1),'Color',blue_color);          % Median
   set(hbox(6,2),'Color',orange_color);
   set(hbox(1,1),'LineStyle','-','Color',blue_color);      % Top whisker line
   set(hbox(1,2),'LineStyle','-','Color',orange_color);      % Top whisker line
   set(hbox(2,1),'LineStyle','-','Color',blue_color);      % Bottom whisker line
   set(hbox(2,2),'LineStyle','-','Color',orange_color);      % Bottom whisker line
   set(hbox(3,1),'Color',blue_color);          % Top whisker bar
   set(hbox(3,2),'Color',orange_color);
   set(hbox(4,1),'Color',blue_color);          % Bottom whisker bar
   set(hbox(4,2),'Color',orange_color);
   if size(hbox,1) > 6
        set(hbox(7,1),'MarkerEdgeColor',blue_color,'MarkerSize',6); 
        set(hbox(7,2),'MarkerEdgeColor',orange_color,'MarkerSize',6);
   end
   set(gca,'YLim',[0 0.8], 'YTick', [0 0.25 0.5 0.75 1], 'YTickLabel', {'0' '0.25' '0.5' '0.75' '1'}, 'FontSize', paper_font_size-1);
   title({'Time to 1st Peak(s)'},'FontSize',paper_font_size-1,'FontWeight','normal');
   sigstar({group_positions},[0.01]); 
      
   annotation('textbox',[0.7 0 0.25 0.05],'String','*\itp\rm < 0.05, **\itp\rm < 0.001','EdgeColor','none','FontSize',paper_font_size-2);    
end
if compute_statistics == 1  
    %% Plot clinical assessemnt results
ARAT_scores = [43 55 53 55;
                             4  4    6   5;
                             45 54 42 49;
                             4   4   4 4;               % (4,3), 
                             39 44 42 40;          % (5,3), interpolated interp1([1 2 4],[39 44 40],3)
                             30 36 38 42];

 FMA_scores = [53	51	56	58	58;
                            24	26	27	26	21;
                            49	48	47	50	54;
                            18	21	24	24	24;     % interp1([1 2 3 5],[18	21 24 24],4)
                            37	44	51	47	43;
                            47	45	49	49	49];
FMA_scores(:,1) = [];

 JebsenTaylor_scores = [74.97	76.48	77.9	91.38;
                                           600		600		600		600;
                                           119.15	104.7	95.36	90.19;
                                           600		600			600     600;
                                           123.48	161.03  148.5250   136.02;
                                           93.58	92.28	104.68	81.11];


ARAT_scores = ARAT_scores - repmat(ARAT_scores(:,1),[1 4]);
FMA_scores = FMA_scores - repmat(FMA_scores(:,1),[1 4]);
JebsenTaylor_scores = JebsenTaylor_scores - repmat(JebsenTaylor_scores(:,1),[1 4]);

    figure('Position',[-1500 200 6*116 3*116]); 
    hold on;
    xrange = 0:2;
    h_arat = errorbar(xrange-0.1, mean(ARAT_scores(:,[1 2 4])), std(ARAT_scores(:,[1 2 4])),'-sb','MarkerFaceColor','b','LineWidth',1,'MarkerSize',10);
    h_fma = errorbar(xrange+0.1, mean(FMA_scores(:,[1 2 4])), std(FMA_scores(:,[1 2 4])),'-or','MarkerFaceColor','w','LineWidth',1,'MarkerSize',10);
    ylim([-4 11]); 
    xlim([-0.1 2.5]);
    set(gca,'Ytick',[-2 0 2 4 6 8 10],'FontSize',paper_font_size);
    %set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
    %set(gca,'Xtick',0:3,'Xticklabel',{'Baseline', 'Post-tt', 'Post-2wks', 'Post-2mon'},'FontSize',paper_font_size);
    set(gca,'Xtick',0:2,'Xticklabel',{'Baseline', 'Post-treat.', 'Post-2 mon.'},'FontSize',paper_font_size);
    legend([h_arat h_fma],{'Action Research Arm Test','Fugl-Meyer Upper Ext.'},'Orientation','Vertical','Location','NorthEastOutside')
    line([-0.5 4],[0 0],'LineStyle','--','Color','k')
    %title('Closed-loop BMI performance for ongoing clinical study','FontSize',paper_font_size);
    %title(Subject_names{subj_n},'FontSize',paper_font_size);
    %ylabel({'True Positive'; 'Rate (%)'},'FontSize',paper_font_size);
    %         ylabel({'BMI Accuracy', 'Mean \pm S.D. (%)'},'FontSize',paper_font_size);
    %         set(get(gca,'YLabel'),'Rotation',0); 
    %xlabel('Sessions','FontSize',10);
    set(gca,'Xgrid','off');
    set(gca,'Ygrid','off');
    annotation('textbox',[0.9 0 0.1 0.07],'String','*','EdgeColor','none','FontSize',24);    
    %% Repeated measures anova 
%http://vassarstats.net/anova1u.html
FMA_rm_anova = [FMA_scores(:,1) FMA_scores(:,2) FMA_scores(:,4)];
ARAT_rm_anova = [ARAT_scores(:,1) ARAT_scores(:,2) ARAT_scores(:,4)];

%[p_ARAT,tbl_ARAT,stats_ARAT] = friedman(ARAT_rm_anova)

% 1 Factor Repeated Measures ANOVA, Chp 11, Pg 275 
alzheimer = [ 7 8 8 7;
                      9 8 10 11;
                      5 6 8 7;
                      10 9 10 11;
                      4 5 7 8;
                      5 4 5 5;
                      6 5 6 7;
                      8 9 9 10];
          
%X = alzheimer;
X = ARAT_rm_anova;
J = size(X,1);  % Number of participants
I = size(X,2);  % Number of treatment levels/factors

Xi_dot = sum(X,1);      % Sigma_X for each level of treatment

X_dotdot = sum(Xi_dot);     % Total Sigma_X

Xi_mean = Xi_dot/J;

CM = (X_dotdot.^2)/(I*J);       % Correction to the mean

SSTr = sum(Xi_dot.^2)/J - CM;
SST = sum(sum(X.^2)) - CM;

% Obtain Sum of Squares for Subjects (SSS)
SSS = sum(sum(X,2).^2)/I - CM;

SSE = SST - SSS - SSTr;

% Calculate degrees of freedom 
df_SSTr = I-1;
df_SST = I*J - 1;
df_SSS = J - 1;
df_SSE = df_SST - df_SSS - df_SSTr;

F_statistic = (SSTr/df_SSTr)/(SSE/df_SSE);
display(F_statistic);
p_value = 1-fcdf(F_statistic,df_SSTr,df_SSE);
display(p_value);

% Determine Effect Size
MSSE = (SSE/df_SSE);
omega_sq = (SSTr - (I-1)*MSSE)/(SST + MSSE);
display(omega_sq);
end

if perform_posthoc_RP_analysis == 1
   for subj_n = 1:length(Subject_names)
        All_Smart_Features = cell(1,13);
        All_EMG_Epochs = cell(1,13);
        
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
        subject_study_data = dlmread(results_filename,',',1,0);
        % Array fields - subject_study_data
        % 1 - Session_nos	2 - Block_number	3- Start_of_trial	4 - End_of_trial	5 - Valid_or_catch	
        % 6 - Intent_detected	7 - Time_to_trigger     8 -  Number_of_attempts     9 - EEG_decisions	10 - EEG_EMG_decisions	
        % 11 - MRCP_slope	12 - MRCP_neg_peak	13 - MRCP_AUC	14 - MRCP_mahalanobis	15 - feature_index	
        % 16 - Corrected_spatial_chan_avg_index     17 - Correction_applied_in_samples      18 - Likert_score	19 - Target     
        % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency 23 - EMG_decisions
                
        unique_session_nos = unique(subject_study_data(:,1));
        classifier_filename = [directory    'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session2\' ...
                 Subject_names{subj_n} '_ses2_cond' num2str(Cond_num(subj_n)) '_block' num2str(Block_num(subj_n))...
                 '_performance_optimized_conventional_smart.mat'];
        
        load(classifier_filename);         
        RP_filename = [directory    'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session2\' ...
                 Subject_names{subj_n} '_ses2_cond' num2str(Cond_num(subj_n)) '_block' num2str(Block_num(subj_n))...
                 '_average_causal.mat'];        
        load(RP_filename);         
        
        All_Smart_Features{1} = Performance.Smart_Features(1:length(Performance.Smart_Features)/2,:);
        All_RPs{1} = [Average.move_avg_channels; Average.move_erp_time];
        
        for ses_num = 1:length(unique_session_nos)
            session_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),:);                                                         
            
            Session_info = session_performance(:,[2,7,19]); %col 2 - start of trial, 7 - Time to Trigger, 19 - Target movement                        
            unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));            
            readbv_files = 1;
            process_raw_emg = 1;
            process_raw_eeg = 1;
            extract_epochs = 1;
            remove_corrupted_epochs = [];
            Posthoc_Average = Calculate_RPs_sessionwise(Subject_names{subj_n}, unique_session_nos(ses_num), readbv_files, unique_block_nos, remove_corrupted_epochs, Impaired_hand{subj_n}, Session_info, Performance, process_raw_emg, process_raw_eeg, extract_epochs);
            Averaged_EMG_epochs = Process_All_EMG_signals(Subject_names{subj_n}, unique_session_nos(ses_num), 0, unique_block_nos, [], Impaired_hand{subj_n}, process_raw_emg, extract_epochs, session_performance,'EMGonset');
            All_Smart_Features{ses_num+1} = Posthoc_Average.Smart_Features;            
            All_RPs{ses_num+1} = [Posthoc_Average.move_avg_channels; Posthoc_Average.move_erp_time];                                    
            All_EMG_Epochs{ses_num+1} = Averaged_EMG_epochs;
        end            
        save([directory 'Subject_' Subject_names{subj_n} '\' 'All_Smart_Features_RPs_EMGs.mat'],'All_Smart_Features','All_RPs','All_EMG_Epochs');
        
        figure;
        channels = [48, 14, 49];
        chanlabels = {'Contralateral', 'Midline (Cz)', 'Ipsilateral'};
        for chn = 1:length(channels)            
            mean_features = zeros(13,4);
            std_features = zeros(13,4);
             subplot(1,3,chn); hold on; grid on;
            for cell_num = 1:13
               mean_features(cell_num,:) = mean(All_Smart_Features{cell_num});
               std_features(cell_num,:) = std(All_Smart_Features{cell_num});
               if cell_num == 3
                   plot(All_RPs{cell_num}(65,:),All_RPs{cell_num}(channels(chn),:),'-r');
                   ylabel('Grand averaged RP (\muV)');
               elseif cell_num == 13    % 9014 use 12, 9009 use 13
                   plot(All_RPs{cell_num}(65,:),All_RPs{cell_num}(channels(chn),:),'-b'); 
               else
                   %plot(All_RPs{cell_num}(65,:),All_RPs{cell_num}(channels(chn),:),'Color',[1 1 1]*(14 - cell_num)/14); 
               end
            end
            title(chanlabels(chn));
            set(gca,'YDir','reverse');
            ylim([-10    10])
            xlim([-2.25 1])
        end
        legend('Therapy start','Therapy end')
        
        figure;                
        M_plot = tight_subplot(4,12,0.01,0.01,0.01); 
        
        if strcmp(Impaired_hand{subj_n},'L')
           channel_order = [5,6,7,8];
        elseif strcmp(Impaired_hand{subj_n},'R')
           channel_order = [7,8,5,6];
        end
        
        for n = 1:4
            for ses_num = 1:length(unique_session_nos)            
                axes(M_plot(12*(n-1)+ses_num));                 
%                 plot(All_EMG_Epochs{ses_num+1}.move_erp_time,All_EMG_Epochs{ses_num+1}.move_avg_channels(channel_order(n),:)./max(max(All_EMG_Epochs{ses_num+1}.move_avg_channels(5:8,:))))
                plot(All_EMG_Epochs{ses_num+1}.move_erp_time,All_EMG_Epochs{ses_num+1}.move_avg_channels(channel_order(n),:))
%                 ylim([0 max(max(All_EMG_Epochs{ses_num+1}.move_avg_channels(5:8,:)))]);
            end
        end 
   end
end

%% Run this analysis by first setting correlate_RP_changes = 1 and compute_trial_averages = 1. Then run this section again 
%  by setting correlate_RP_changes = 1 and compute_trial_averages = 0.

if correlate_RP_changes == 1
    
    
    NumElements = zeros(10,12);
    Subject_wise_RP_metrics = zeros(64,6,10);
    figure('Position',[300 300 2.5*116 1.5*116]);
    C12_plot = tight_subplot(2,5,0.05,[0.05 0.05],[0.05 0.05]);
    figure('Position',[300 50 2.5*116 1.5*116]);
    FC12_plot = tight_subplot(2,5,0.05,[0.05 0.05],[0.05 0.05]);

    for subj_n = 1:length(Subject_names)   
        

        
        if compute_trial_averages == 1
            fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
            results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
            if ~exist(results_filename,'file')
                continue
            end
            subject_study_data = dlmread(results_filename,',',1,0);
            % Array fields - subject_study_data
            % 1 - Session_nos	2 - Block_number	3- Start_of_trial	4 - End_of_trial	5 - Valid_or_catch	
            % 6 - Intent_detected	7 - Time_to_trigger     8 -  Number_of_attempts     9 - EEG_decisions	10 - EEG_EMG_decisions	
            % 11 - MRCP_slope	12 - MRCP_neg_peak	13 - MRCP_AUC	14 - MRCP_mahalanobis	15 - feature_index	
            % 16 - Corrected_spatial_chan_avg_index     17 - Correction_applied_in_samples      18 - Likert_score	19 - Target     
            % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency 23 - EMG_decisions

            unique_session_nos = unique(subject_study_data(:,1));        
            unique_session_nos = unique_session_nos([1,2,end-1,end]); % Only look at first 2 and last 2 sessions
            
            Therapy_start_move_epochs = [];
            Therapy_end_move_epochs = [];
            for ses_num = 1:length(unique_session_nos)            
                unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2)); 
                RP_filename = [directory    'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(unique_session_nos(ses_num)) '\'...
                    Subject_names{subj_n} '_ses' num2str(unique_session_nos(ses_num)) '_closeloop_block' num2str(20*length(unique_block_nos))...
                     '_posthoc_average.mat'];        
                load(RP_filename);   
    %             NumElements(subj_n,ses_num) = size(Posthoc_Average.move_epochs,1);
                [no_epochs,no_timepoints,no_channels] = size(Posthoc_Average.move_epochs);
                LRP_move_epochs = zeros(no_epochs,no_timepoints,2); % without baseline correction
                LRP_mean_baseline = zeros(2,no_epochs);
                for epoch_cnt = 1:no_epochs % Added on 01-04-2020
                    LRP_move_epochs(epoch_cnt,:,1) = Posthoc_Average.move_epochs(epoch_cnt,:,13) - Posthoc_Average.move_epochs(epoch_cnt,:,15);
                    LRP_mean_baseline(1,epoch_cnt) = mean(LRP_move_epochs(epoch_cnt,find(Posthoc_Average.move_erp_time == (baseline_int(1))):find(Posthoc_Average.move_erp_time == (baseline_int(2))),1));
                    LRP_move_epochs(epoch_cnt,:,1) = LRP_move_epochs(epoch_cnt,:,1) - LRP_mean_baseline(1,epoch_cnt);
                    LRP_move_epochs(epoch_cnt,:,2) = Posthoc_Average.move_epochs(epoch_cnt,:,15) - Posthoc_Average.move_epochs(epoch_cnt,:,13);
                    LRP_mean_baseline(2,epoch_cnt) = mean(LRP_move_epochs(epoch_cnt,find(Posthoc_Average.move_erp_time == (baseline_int(1))):find(Posthoc_Average.move_erp_time == (baseline_int(2))),2));
                    LRP_move_epochs(epoch_cnt,:,2) = LRP_move_epochs(epoch_cnt,:,2) - LRP_mean_baseline(2,epoch_cnt);
                end
                
                
                for epoch_cnt = 1:no_epochs
                    for channel_cnt = 1:no_channels
                        Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt) = Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt) ...
                            - Posthoc_Average.move_mean_baseline(channel_cnt,epoch_cnt);                    
                    end
                end            
                Posthoc_Average.move_epochs(:,:,13) = LRP_move_epochs(:,:,1); % Added on 01-04-2020
                Posthoc_Average.move_epochs(:,:,15) = LRP_move_epochs(:,:,2); % Added on 01-04-2020
                Posthoc_Average.move_epochs(Posthoc_Average.epochs_with_artefacts, :, :) = []; %Decided to keep it  on 01-04-2020

                switch (ses_num)
                    case 1
                        Therapy_start_move_epochs = Posthoc_Average.move_epochs;                    

                    case 2
                        [no_epochs,~,no_channels] = size(Posthoc_Average.move_epochs);            
                        for epoch_cnt = 1:no_epochs
                            Therapy_start_move_epochs(end+1,:,1) = Posthoc_Average.move_epochs(epoch_cnt,:,1);
                            for channel_cnt = 2:no_channels
                                Therapy_start_move_epochs(end,:,channel_cnt) = Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt);
                            end
                        end

                    case 3                    
                        Therapy_end_move_epochs = Posthoc_Average.move_epochs;                    

                    case 4
                        [no_epochs,~,no_channels] = size(Posthoc_Average.move_epochs);            
                        for epoch_cnt = 1:no_epochs
                            Therapy_end_move_epochs(end+1,:,1) = Posthoc_Average.move_epochs(epoch_cnt,:,1);
                            for channel_cnt = 2:no_channels
                                Therapy_end_move_epochs(end,:,channel_cnt) = Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt);
                            end
                        end                    
                end
            end
            % Determine t-statistics for 95% C.I.
            %http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s    
            deg_freedom_start = size(Therapy_start_move_epochs,1) - 1;
            t_value_start = tinv(1 - 0.05/2, deg_freedom_start);
            deg_freedom_end = size(Therapy_end_move_epochs,1) - 1;
            t_value_end = tinv(1 - 0.05/2, deg_freedom_end);

            for channel_cnt = 1:no_channels
%                 if channel_cnt == 48
%                     Therapy_start_grand_average(channel_cnt,:) =  mean((Therapy_start_move_epochs(:,:,48) + Therapy_start_move_epochs(:,:,19))/2);
%                     Therapy_start_grand_average_SE(channel_cnt,:) = t_value_start.*std((Therapy_start_move_epochs(:,:,48) + Therapy_start_move_epochs(:,:,19))/2)/sqrt(size(Therapy_start_move_epochs,1));
%                     Therapy_end_grand_average(channel_cnt,:) =  mean((Therapy_end_move_epochs(:,:,48) + Therapy_end_move_epochs(:,:,19))/2);
%                     Therapy_end_grand_average_SE(channel_cnt,:) = t_value_end.*std((Therapy_end_move_epochs(:,:,48) + Therapy_end_move_epochs(:,:,19))/2)/sqrt(size(Therapy_end_move_epochs,1));
%                 elseif channel_cnt == 14
%                     Therapy_start_grand_average(channel_cnt,:) =  mean((Therapy_start_move_epochs(:,:,14) + Therapy_start_move_epochs(:,:,53))/2);
%                     Therapy_start_grand_average_SE(channel_cnt,:) = t_value_start.*std((Therapy_start_move_epochs(:,:,14) + Therapy_start_move_epochs(:,:,53))/2)/sqrt(size(Therapy_start_move_epochs,1));
%                     Therapy_end_grand_average(channel_cnt,:) =  mean((Therapy_end_move_epochs(:,:,14) + Therapy_end_move_epochs(:,:,53))/2);
%                     Therapy_end_grand_average_SE(channel_cnt,:) = t_value_end.*std((Therapy_end_move_epochs(:,:,14) + Therapy_end_move_epochs(:,:,53))/2)/sqrt(size(Therapy_end_move_epochs,1));
%                 elseif channel_cnt == 49
%                     Therapy_start_grand_average(channel_cnt,:) =  mean((Therapy_start_move_epochs(:,:,49) + Therapy_start_move_epochs(:,:,20))/2);
%                     Therapy_start_grand_average_SE(channel_cnt,:) = t_value_start.*std((Therapy_start_move_epochs(:,:,49) + Therapy_start_move_epochs(:,:,20))/2)/sqrt(size(Therapy_start_move_epochs,1));
%                     Therapy_end_grand_average(channel_cnt,:) =  mean((Therapy_end_move_epochs(:,:,49) + Therapy_end_move_epochs(:,:,20))/2);
%                     Therapy_end_grand_average_SE(channel_cnt,:) = t_value_end.*std((Therapy_end_move_epochs(:,:,49) + Therapy_end_move_epochs(:,:,20))/2)/sqrt(size(Therapy_end_move_epochs,1));
%                 else
                    Therapy_start_grand_average(channel_cnt,:) =  mean(Therapy_start_move_epochs(:,:,channel_cnt));
                    Therapy_start_grand_average_SE(channel_cnt,:) = t_value_start.*std(Therapy_start_move_epochs(:,:,channel_cnt))/sqrt(size(Therapy_start_move_epochs,1));
                    Therapy_end_grand_average(channel_cnt,:) =  mean(Therapy_end_move_epochs(:,:,channel_cnt));
                    Therapy_end_grand_average_SE(channel_cnt,:) = t_value_end.*std(Therapy_end_move_epochs(:,:,channel_cnt))/sqrt(size(Therapy_end_move_epochs,1));
%                 end
            end

            Channels_nos = [ 13, 48, 14, 49, 15, ...
                             52, 19, 53, 20, 54];                 
            figure('NumberTitle', 'off', 'Name', [Subject_names{subj_n}, ' , impaired = ', Impaired_hand{subj_n}]);
            %figure('units','normalized','outerposition',[0 0 1 1])
            T_plot = tight_subplot(numel(Channels_nos)/5,5,[0.01 0.01],[0.15 0.01],[0.1 0.1]);
            hold on;        

            for ind4 = 1:length(Channels_nos)
                axes(T_plot(ind4));    
                hold on;
                plot(Posthoc_Average.move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:),'r','LineWidth',1.5);
                plot(Posthoc_Average.move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:)+ (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);
                plot(Posthoc_Average.move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:) - (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);

                plot(Posthoc_Average.move_erp_time,Therapy_end_grand_average(Channels_nos(ind4),:),'b','LineWidth',1.5);
                plot(Posthoc_Average.move_erp_time,Therapy_end_grand_average(Channels_nos(ind4),:)+ (Therapy_end_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);
                plot(Posthoc_Average.move_erp_time,Therapy_end_grand_average(Channels_nos(ind4),:) - (Therapy_end_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);

                % Added 8/21/2015        
                text(-2,-4,int2str(Channels_nos(ind4)),'Color','k','FontWeight','normal','FontSize',paper_font_size-1); 
                set(gca,'YDir','reverse');
                axis([-2.5 1.25 -10 10]);                
                line([0 0],[-30 20],'Color','k','LineWidth',0.5,'LineStyle','--');  
                line([-2.5 4],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
    %             plot_ind4 = plot_ind4 + 1;
                grid on;

            end
            move_erp_time = Posthoc_Average.move_erp_time;
            save([directory 'Subject_' Subject_names{subj_n} '\' 'Pre_Post_RPs.mat'],'Therapy_start_grand_average','Therapy_start_grand_average_SE', 'Therapy_end_grand_average', 'Therapy_end_grand_average_SE','move_erp_time');
        else
            load([directory 'Subject_' Subject_names{subj_n} '\' 'Pre_Post_RPs.mat']);                         
            
            [Therapy_start_RP_peak, Therapy_start_RP_peakIndex] = min(Therapy_start_grand_average(:, find(move_erp_time == -1.0,1):find(move_erp_time == 1.0,1)), [], 2);                                
            Therapy_start_RP_peakIndex = Therapy_start_RP_peakIndex + 500;            
                        
            [Therapy_end_RP_peak, Therapy_end_RP_peakIndex] = min(Therapy_end_grand_average(:, find(move_erp_time == -1.0,1):find(move_erp_time == 1.0,1)), [], 2);
            Therapy_end_RP_peakIndex = Therapy_end_RP_peakIndex + 500;            
             
            %             switch (subj_n)
%                 case 1
%                     
%                 case 2
%                     Therapy_start_RP_peakIndex([48,49]) = [817, 805] - 500;
%                     
%                 case 3
%                     Therapy_start_RP_peakIndex([14,49]) = [778, 778] - 500;
%                     
%                 case 4
%                     Therapy_start_RP_peakIndex([48]) = [601] - 500;
%                     Therapy_end_RP_peakIndex([14]) = [848] - 500;
%                     
%                 case 5
%                     Therapy_start_RP_peakIndex([48]) = [811] - 500;
%                     Therapy_end_RP_peakIndex([49]) = [605] - 500;
%                     
%                 case 6
%                     Therapy_start_RP_peakIndex([48, 14]) = [800, 800] - 500;
%                     
%                 case 7
%                     Therapy_end_RP_peakIndex([48,14,49]) = [550, 775, 743] - 500;
%                     
%                 case 8
%                     
%                 case 9
%                     Therapy_start_RP_peakIndex([48, 14]) = [651, 531] - 500;
%                     Therapy_end_RP_peakIndex([49]) = [801] - 500;
%                     
%                 case 10
%                     Therapy_start_RP_peakIndex([14]) = [719] - 500;
%                     Therapy_end_RP_peakIndex([14,49]) = [738, 715] - 500;
%                     
%             end
            slope_begin = Therapy_start_RP_peakIndex - 200;
            slope_end = Therapy_start_RP_peakIndex;
            Therapy_start_RP_slope = zeros(64,1);
            for ch = 1:64
                Therapy_start_RP_slope(ch) = (Therapy_start_grand_average(ch,slope_begin(ch)) - Therapy_start_grand_average(ch,slope_end(ch)))/1.0;
            end
            
            slope_begin = Therapy_end_RP_peakIndex - 200; %200
            slope_end = Therapy_end_RP_peakIndex;
            Therapy_end_RP_slope = zeros(64,1);
            for ch = 1:64
                Therapy_end_RP_slope(ch) = (Therapy_end_grand_average(ch,slope_begin(ch)) - Therapy_end_grand_average(ch,slope_end(ch)))/1.0;
            end                                   
            
            Therapy_start_RP_onset_latency = zeros(64,1);
            Therapy_end_RP_onset_latency = zeros(64,1);
            for ch_no = 1:64
                onset_latency = find(Therapy_start_grand_average(ch_no, find(move_erp_time == -2.0,1):find(move_erp_time == 1.0,1))...
                    <= Therapy_start_RP_peak(ch_no)*0.3,1); % Threshold = 30% of max.
                if isempty(onset_latency)
                    onset_latency = 1;
                end
               Therapy_start_RP_onset_latency(ch_no) = onset_latency + 300; 
               
               onset_latency = find(Therapy_end_grand_average(ch_no, find(move_erp_time == -2.0,1):find(move_erp_time == 1.0,1))...
                    <= Therapy_end_RP_peak(ch_no)*0.3,1); % Threshold = 30% of max.
                if isempty(onset_latency)
                    onset_latency = 1;
                end
               Therapy_end_RP_onset_latency(ch_no) = onset_latency + 300; 
            end
            
             if strcmp(Impaired_hand(subj_n),'R')
               Channels_nos = [48,  19];
%                  Channels_nos = [19, 53, 20];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement               
            else
               Channels_nos = [49, 20];
%                  Channels_nos = [20, 53, 19];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement                 
            end 
%             figure('NumberTitle', 'off', 'Name', [Subject_names{subj_n}], 'Position',[300 5 3*116 2*116]);
%             T_plot = tight_subplot(1,2,0.01,[0.15 0.1],[0.1 0.1]);
%             for ind4 = 1:length(Channels_nos)
%                 axes(T_plot(ind4));
%                 hold on;
%                 hstart = plot(move_erp_time, Therapy_start_grand_average(Channels_nos(ind4),:),'Color',blue_color,'LineWidth',1);                
% %                 plot(move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:)+ (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);
% %                 plot(move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:) - (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);
%                 
%                 hend = plot(move_erp_time, Therapy_end_grand_average(Channels_nos(ind4),:),'Color',orange_color, 'LineWidth',1);
% %                 plot(move_erp_time,Therapy_end_grand_average(Channels_nos(ind4),:)+ (Therapy_end_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);
% %                 plot(move_erp_time,Therapy_end_grand_average(Channels_nos(ind4),:) - (Therapy_end_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[0 0 1],'LineWidth',0.25);
%                 
%                 % Added 8/21/2015        
% %                 text(-2,-4,int2str(Channels_nos(ind4)),'Color','k','FontWeight','normal','FontSize',paper_font_size-1); 
%                 set(gca,'YDir','reverse');
% %                 axis([-2.5 1.25 -5.5 1]);                
%                 axis([-2.5 1.25 -3 2.5]);                
%                 line([0 0],[-30 20],'Color','k','LineWidth',0.5,'LineStyle','--');  
% %                 line([0 1000],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
% %                 line(-3.5 + [Therapy_start_RP_peakIndex(Channels_nos(ind4)), Therapy_start_RP_peakIndex(Channels_nos(ind4))]/200,[-30 20],'Color',blue_color,'LineWidth',0.5,'LineStyle','--');  
% %                 line(-3.5 + [Therapy_end_RP_peakIndex(Channels_nos(ind4)), Therapy_end_RP_peakIndex(Channels_nos(ind4))]/200,[-30 20],'Color',orange_color,'LineWidth',0.5,'LineStyle','--');  
% %                 line(-3.5+ [Therapy_start_RP_onset_latency(Channels_nos(ind4)), Therapy_start_RP_onset_latency(Channels_nos(ind4))]/200,[-30 20],'Color',blue_color,'LineWidth',0.5,'LineStyle','--');  
% %                 line(-3.5+ [Therapy_end_RP_onset_latency(Channels_nos(ind4)), Therapy_end_RP_onset_latency(Channels_nos(ind4))]/200,[-30 20],'Color',orange_color,'LineWidth',0.5,'LineStyle','--');  
%     %             plot_ind4 = plot_ind4 + 1;
%                 grid on;
%                 set(gca,'YTick',[-2.5 0 2.5], 'Xtick',[-2 -1 0 1], 'XTickLabel',{'-2', '-1', '0', '1'});
%                 switch (ind4)
%                     case 1
%                         ylabel('Avg. MRCP amplitude (\muV)');
%                         title('Contra. C_{1/2}','FontWeight','normal');                        
%                         set(gca,'YTickLabel',{'-2.5' '0' '2.5'});
%                         xlabel('Time (s)');
%                     case 2                    
% %                         ylabel('Avg. MRCP amplitude (\muV)');
%                         title('Contra. FC_{1/2}','FontWeight','normal');                        
%                         set(gca,'YTickLabel',{'-2.5' '0' '2.5'});
%                         xlabel('Time (s)');
% %                     case 3                    
% %                         title('Ipsi-');
%                         legend([hstart, hend],{'start', 'end'});
%                 end
%                 
%             end
            
            axes(C12_plot(11-subj_n));
            hold on;
            hstart = plot(move_erp_time, Therapy_start_grand_average(Channels_nos(1),:),'Color',blue_color,'LineWidth',1);                
            hend = plot(move_erp_time, Therapy_end_grand_average(Channels_nos(1),:),'Color',orange_color, 'LineWidth',1);                        
            set(gca,'YDir','reverse');
            if ((subj_n == 2)||(subj_n == 3))
                axis([-2.5 1 -3 2.5]);                 
                text(-2,-1.5,['P', int2str(11 - subj_n)],'Color','k','FontWeight','normal','FontSize',paper_font_size-3); 
                line([0 0],[-2.5 2.5],'Color','k','LineWidth',0.5,'LineStyle','--');  
            else
                axis([-2.5 1 -5.5 1]);                
                text(-2,-4,['P', int2str(11 - subj_n)],'Color','k','FontWeight','normal','FontSize',paper_font_size-3); 
                line([0 0],[-4 1],'Color','k','LineWidth',0.5,'LineStyle','--');  
            end            
            line([-3 2],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');                   
            set(gca,'Visible', 'off');
            
            axes(FC12_plot(11-subj_n));
            hold on;
            hstart = plot(move_erp_time, Therapy_start_grand_average(Channels_nos(2),:),'Color',blue_color,'LineWidth',1);                
            hend = plot(move_erp_time, Therapy_end_grand_average(Channels_nos(2),:),'Color',orange_color, 'LineWidth',1);                                   
            set(gca,'YDir','reverse');
            if ((subj_n == 6)||(subj_n == 7))
                axis([-2.5 1 -3 2.5]);                 
                text(-2,-1.5,['P', int2str(11 - subj_n)],'Color','k','FontWeight','normal','FontSize',paper_font_size-3); 
                line([0 0],[-2.5 2.5],'Color','k','LineWidth',0.5,'LineStyle','--');  
            else
                axis([-2.5 1 -5.5 1]);                
                text(-2,-4,['P', int2str(11 - subj_n)],'Color','k','FontWeight','normal','FontSize',paper_font_size-3); 
                line([0 0],[-4 1],'Color','k','LineWidth',0.5,'LineStyle','--');  
            end            
            line([-3 3],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');                   
            set(gca,'Visible', 'off');
            % Annotate line
            if subj_n == 1
                axes(FC12_plot(10));
                axes_pos = get(gca,'Position'); %[lower bottom width height]
                axes_ylim = get(gca,'Ylim');
                annotate_length = (3*axes_pos(4))/(axes_ylim(2) - axes_ylim(1));
                annotation(gcf,'line', [(axes_pos(1)+axes_pos(3)+0.025) (axes_pos(1)+axes_pos(3)+0.025)],...
                    [(axes_pos(2)+axes_pos(4) - annotate_length/5) (axes_pos(2)+axes_pos(4) - annotate_length - annotate_length/5)],'LineWidth',2);
            end
            
            Subject_wise_RP_metrics(:,:,subj_n) = [Therapy_start_RP_peak, Therapy_start_RP_slope, Therapy_start_RP_onset_latency...
            Therapy_end_RP_peak, Therapy_end_RP_slope, Therapy_end_RP_onset_latency];                               
        end               
    end
    
    if compute_trial_averages == 0        
        Subject_wise_RP_peak_therapy_start = zeros(10,2);
        Subject_wise_RP_peak_therapy_end = zeros(10,2);
        Subject_wise_RP_slope_therapy_start = zeros(10,2);
        Subject_wise_RP_slope_therapy_end = zeros(10,2);
        Subject_wise_RP_latency_therapy_start = zeros(10,2);
        Subject_wise_RP_latency_therapy_end = zeros(10,2);
        
        for subj_n = 1:length(Subject_names)
            if strcmp(Impaired_hand(subj_n),'R')
%                Channels_nos = [48,  14,  49];
%                  Channels_nos = [4, 43, 13, 52, 24, 38, 9, 48, 19,57,...
%                      5, 32, 14, 53, 25,...
%                      6, 44, 15, 54, 26, 39, 10, 49, 20, 38];
                 Channels_nos = [48, 19];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement               
            else
%                Channels_nos = [49,  14,  48];
%                Channels_nos = [6, 44, 15, 54, 26, 39, 10, 49, 20, 38,...
%                      5, 32, 14, 53, 25,...                     
%                      4, 43, 13, 52, 24, 38, 9, 48, 19,57];
                 Channels_nos = [49, 20];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement                 
            end
               Subject_wise_RP_peak_therapy_start(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,1,subj_n);
               Subject_wise_RP_peak_therapy_end(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,4,subj_n);
               Subject_wise_RP_slope_therapy_start(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,2,subj_n);
               Subject_wise_RP_slope_therapy_end(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,5,subj_n);               
               Subject_wise_RP_latency_therapy_start(subj_n,:,:)= Subject_wise_RP_metrics(Channels_nos,3,subj_n);               
               Subject_wise_RP_latency_therapy_end(subj_n,:,:)= Subject_wise_RP_metrics(Channels_nos,6,subj_n);               
        end
%         figure; 
%         for ch = 1:3
%             for i = 1:10
%                 subplot(2,3,ch); hold on;
% %             boxplot([Subject_wise_RP_peak_therapy_start(:,ch),Subject_wise_RP_peak_therapy_end(:,ch)]);            
%                 plot([1,2],[Subject_wise_RP_peak_therapy_start(i,ch), Subject_wise_RP_peak_therapy_end(i,ch)],'-o','MarkerSize',8);
%                 set(gca,'YDir','reverse');
%                 xlim([0.5 2.5]);
%                 
%                 subplot(2,3,ch + 3); hold on;
% %             boxplot([Subject_wise_RP_slope_therapy_start(:,ch),Subject_wise_RP_slope_therapy_end(:,ch)]);
%                 plot([1,2],[Subject_wise_RP_slope_therapy_start(i,ch), Subject_wise_RP_slope_therapy_end(i,ch)],'-o','MarkerSize',8);
%                 xlim([0.5 2.5]);
%             end            
%         end
%         
%         figure; 
%         for i = 1:10
%             for ch = 1:3            
%                 subplot(2,5,i); hold on;
% %             boxplot([Subject_wise_RP_peak_therapy_start(:,ch),Subject_wise_RP_peak_therapy_end(:,ch)]);            
% %               plot([1,2],[Subject_wise_RP_peak_therapy_start(i,ch), Subject_wise_RP_peak_therapy_end(i,ch)],'-o','MarkerSize',8);
% %               set(gca,'YDir','reverse');
%                 plot([1,2],[Subject_wise_RP_slope_therapy_start(i,ch), Subject_wise_RP_slope_therapy_end(i,ch)],'-o','MarkerSize',8);
%                 
%                 xlim([0.5 2.5]);                                
%             end            
%         end
        
        FMAchange = [-1     7     4     5     4     8     3    -1     1     5]';
%         FMAchange = [2     4     3     4     4     5     0    -1    -1     6]';
        ARATchange = [2     1    10     7     6     5     0     9     0    12]';
%         FMAchange = [6   NaN     6    10     4   NaN   NaN     2     0     7]';
%         ARATchange =[6   NaN     9    10     8   NaN   NaN    -3     2    10]';
        RPpeak_change = -1.*((Subject_wise_RP_peak_therapy_end - Subject_wise_RP_peak_therapy_start));
        RPslope_change = ((Subject_wise_RP_slope_therapy_end - Subject_wise_RP_slope_therapy_start));
        RPlatency_change = 1*((Subject_wise_RP_latency_therapy_end - Subject_wise_RP_latency_therapy_start));
        
        [RP_peak_FMA_corr,RP_peak_FMA_corr_p] = corr(RPpeak_change,FMAchange,'rows','complete');
        [RP_peak_ARAT_corr,RP_peak_ARAT_corr_p] = corr(RPpeak_change, ARATchange,'rows','complete');
        [RP_slope_FMA_corr,p] = corr(RPslope_change,FMAchange,'rows','complete');
        [RP_slope_ARAT_corr,p] = corr(RPslope_change,ARATchange,'rows','complete');
        [RP_latency_FMA_corr,p] = corr(RPlatency_change,FMAchange,'rows','complete');
        [RPlatency_ARAT_corr,p] = corr(RPlatency_change,ARATchange,'rows','complete');
        
        figure('Position',[300 5 5*116 2.5*116]);         
        subplot(1,2,2); hold on;
        plot(RPpeak_change(:,2), FMAchange, '.k');                    
        box on;
        text(RPpeak_change(:,2), FMAchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
        mlr_fit = LinearModel.fit(RPpeak_change(:,2),FMAchange);            
        line_regress = [ones(2,1) [min(RPpeak_change(:,2)) - 2; max(RPpeak_change(:,2)) + 2]]*mlr_fit.Coefficients.Estimate;
        plot([min(RPpeak_change(:,2)) - 2, max(RPpeak_change(:,2)) + 2],line_regress,'--r','LineWidth',1);                                    
        xlim([min(RPpeak_change(:,2)) - 0.75, max(RPpeak_change(:,2)) + 0.75]); 
        ylim([-2 10]);
        text(min(RPpeak_change(:,2))-0.5, 9.5, ['\rho = ', num2str(RP_peak_FMA_corr(2),2),', \itp\rm = ',num2str(RP_peak_FMA_corr_p(2),2)], 'FontSize', paper_font_size-2, 'Color', [1 0 0]);            
        xlabel('Amplitude change (\muV)');
        ylabel('FMA change post 1-week');
        title('Contralateral FC_{1/2}');
        
        
        subplot(1,2,1); hold on;
        plot(RPpeak_change(:,1), ARATchange, '.k');        
        box on;
        text(RPpeak_change(:,1), ARATchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
        mlr_fit2 = LinearModel.fit(RPpeak_change(:,1),ARATchange);            
        line_regress2 = [ones(2,1) [min(RPpeak_change(:,1)) - 2; max(RPpeak_change(:,1)) + 2]]*mlr_fit2.Coefficients.Estimate;
        plot([min(RPpeak_change(:,1)) - 2, max(RPpeak_change(:,1)) + 2],line_regress2,'--r','LineWidth',1);            
        xlim([min(RPpeak_change(:,1)) - 0.25, max(RPpeak_change(:,1)) + 0.75]); 
        ylim([-2 14]);            
        text(min(RPpeak_change(:,1)), 13.5, ['\rho = ', num2str(RP_peak_ARAT_corr(1),2),', \itp\rm < 0.05'], 'FontSize', paper_font_size-2, 'Color', [1 0 0]);            
        xlabel('Amplitude change (\muV)');        
        ylabel('ARAT change post 1-week');
        title('Contralateral C_{1/2}');

        
%         figure('Position',[300 5 6*116 4*116]); 
%         for i = 1:3
%             subplot(2,3,i); hold on;
%             plot(RPpeak_change(:,i), FMAchange, '.k');                    
%             box on;
%             text(RPpeak_change(:,i), FMAchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
%             mlr_fit = LinearModel.fit(RPpeak_change(:,i),FMAchange);            
%             line_regress = [ones(2,1) [min(RPpeak_change(:,i)) - 2; max(RPpeak_change(:,i)) + 2]]*mlr_fit.Coefficients.Estimate;
%             plot([min(RPpeak_change(:,i)) - 2, max(RPpeak_change(:,i)) + 2],line_regress,'--r','LineWidth',1);                                    
%             xlim([min(RPpeak_change(:,i)) - 0.75, max(RPpeak_change(:,i)) + 0.75]); 
%             ylim([-2 10]);
%             text(min(RPpeak_change(:,i))-0.5, 9.5, ['\rho = ', num2str(RP_peak_FMA_corr(i),2)], 'FontSize', paper_font_size-2, 'Color', [1 0 0]);            
%             xlabel('Amplitude change (\muV)');
%             switch (i)
%                 case 1
%                     ylabel('FMA change');
%                     title('Contralateral elec.');
%                 case 2                    
%                     title('Midline elec.');
%                 case 3                    
%                     title('Ipsilateral elec.');
%             end
%                     
% %             subplot(2,3,i+3); hold on;
% %             plot(RPslope_change(:,i), FMAchange, '.b');                    
% %             box on;
% %             text(RPslope_change(:,i), FMAchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 1]);
% %             mlr_fit = LinearModel.fit(RPslope_change(:,i),FMAchange);            
% %             line_regress = [ones(2,1) [min(RPslope_change(:,i)) - 2; max(RPslope_change(:,i)) + 2]]*mlr_fit.Coefficients.Estimate;
% %             plot([min(RPslope_change(:,i)) - 2, max(RPslope_change(:,i)) + 2],line_regress,'-b','LineWidth',1);                                                 
% %             xlim([min(RPslope_change(:,i)) - 0.75, max(RPslope_change(:,i)) + 0.75]); 
% %             ylim([-2 10]); 
% %             text(min(RPslope_change(:,i))-0.5, 9.5, ['\rho = ', num2str(RP_slope_FMA_corr(i),2)], 'FontSize', 8, 'Color', [0 0 0]);            
% 
%             subplot(2,3,i+3); hold on;
%             plot(RPpeak_change(:,i), ARATchange, '.k');        
%             box on;
%             text(RPpeak_change(:,i), ARATchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
%             mlr_fit2 = LinearModel.fit(RPpeak_change(:,i),ARATchange);            
%             line_regress2 = [ones(2,1) [min(RPpeak_change(:,i)) - 2; max(RPpeak_change(:,i)) + 2]]*mlr_fit2.Coefficients.Estimate;
%             plot([min(RPpeak_change(:,i)) - 2, max(RPpeak_change(:,i)) + 2],line_regress2,'--r','LineWidth',1);            
%             xlim([min(RPpeak_change(:,i)) - 0.25, max(RPpeak_change(:,i)) + 0.75]); 
%             ylim([-2 14]);            
%             text(min(RPpeak_change(:,i)), 13, ['\rho = ', num2str(RP_peak_ARAT_corr(i),2)], 'FontSize', paper_font_size-2, 'Color', [1 0 0]);            
%             xlabel('Amplitude change (\muV)');
%             switch (i)
%                 case 1
%                     ylabel('ARAT change');
%                     title('Contralateral elec.');
%                 case 2                    
%                     title('Midline elec.');
%                 case 3                    
%                     title('Ipsilateral elec.');
%             end
%             
%         end                
    end    
end

%% 

if perform_posthoc_EMG_analysis == 1
    for subj_n = 1:length(Subject_names)
        All_EMG_Epochs = cell(1,12);
               
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
        subject_study_data = dlmread(results_filename,',',1,0);
        % Array fields - subject_study_data
        % 1 - Session_nos	2 - Block_number	3- Start_of_trial	4 - End_of_trial	5 - Valid_or_catch	
        % 6 - Intent_detected	7 - Time_to_trigger     8 -  Number_of_attempts     9 - EEG_decisions	10 - EEG_EMG_decisions	
        % 11 - MRCP_slope	12 - MRCP_neg_peak	13 - MRCP_AUC	14 - MRCP_mahalanobis	15 - feature_index	
        % 16 - Corrected_spatial_chan_avg_index     17 - Correction_applied_in_samples      18 - Likert_score	19 - Target     
        % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency 23 - EMG_decisions
                
        unique_session_nos = unique(subject_study_data(:,1));
        
        for ses_num = 1:length(unique_session_nos) 
            session_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),:);                                                         
            unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));            
            readbv_files = 1;
            process_raw_emg = 1;            
            extract_epochs = 1;
            remove_corrupted_epochs = [];
            Averaged_EMG_epochs = Process_All_EMG_signals(Subject_names{subj_n}, unique_session_nos(ses_num), readbv_files, unique_block_nos, remove_corrupted_epochs, Impaired_hand{subj_n}, process_raw_emg, extract_epochs, session_performance);
            All_EMG_Epochs{ses_num} = Averaged_EMG_epochs;                        
        end            
        save([directory 'Subject_' Subject_names{subj_n} '\' 'All_Smart_Features.mat'],'All_Smart_Features','All_RPs');
        
        figure;                
        M_plot = tight_subplot(4,12,0.01,0.01,0.01); 
        
        if strcmp(Impaired_hand{subj_n},'L')
           channel_order = [5,6,7,8];
        elseif strcmp(Impaired_hand{subj_n},'R')
           channel_order = [7,8,5,6];
        end
        
        for n = 1:4
            for ses_num = 1:length(unique_session_nos)            
                axes(M_plot(12*(n-1)+ses_num)); 
                plot(All_EMG_Epochs{ses_num})
            end
        end
   end
end

%% Plot average bilateral EMG signals
if (plot_bilateral_EMG_traces == 1)
     for subj_n = 1:length(Subject_names)               
        fileid = dir([directory 'Subject_' Subject_names{subj_n} '\All_Smart_Features_RPs_EMGs.mat']);
        results_filename = [directory 'Subject_' Subject_names{subj_n} '\' fileid.name];
        if ~exist(results_filename,'file')
            continue
        end
        load(results_filename);
        Therapy_start_EMG_epochs_with_base_correct = All_EMG_Epochs{2}.move_epochs_with_base_correct;                    

        [no_epochs,~,no_channels] = size(All_EMG_Epochs{3}.move_epochs_with_base_correct);            
        for epoch_cnt = 1:no_epochs
            Therapy_start_EMG_epochs_with_base_correct(end+1,:,1) = All_EMG_Epochs{3}.move_epochs_with_base_correct(epoch_cnt,:,1);
            for channel_cnt = 2:no_channels
                Therapy_start_EMG_epochs_with_base_correct(end,:,channel_cnt) = All_EMG_Epochs{3}.move_epochs_with_base_correct(epoch_cnt,:,channel_cnt);
            end
        end
                        
        Therapy_end_EMG_epochs_with_base_correct = All_EMG_Epochs{12}.move_epochs_with_base_correct;
        [no_epochs,~,no_channels] = size(All_EMG_Epochs{13}.move_epochs_with_base_correct);            
        for epoch_cnt = 1:no_epochs
            Therapy_end_EMG_epochs_with_base_correct(end+1,:,1) = All_EMG_Epochs{13}.move_epochs_with_base_correct(epoch_cnt,:,1);
            for channel_cnt = 2:no_channels
                Therapy_end_EMG_epochs_with_base_correct(end,:,channel_cnt) = All_EMG_Epochs{13}.move_epochs_with_base_correct(epoch_cnt,:,channel_cnt);
            end
        end
    
        deg_freedom_start = size(Therapy_start_EMG_epochs_with_base_correct,1) - 1;
        t_value_start = tinv(1 - 0.05/2, deg_freedom_start);
        deg_freedom_end = size(Therapy_end_EMG_epochs_with_base_correct,1) - 1;
        t_value_end = tinv(1 - 0.05/2, deg_freedom_end);

        if strcmp(Impaired_hand{subj_n},'L')
           Therapy_start_flexion_max = max(mean(Therapy_start_EMG_epochs_with_base_correct(:,:,5)));
           Therapy_start_extension_max = max(mean(Therapy_start_EMG_epochs_with_base_correct(:,:,6)));
           Therapy_end_flexion_max = max(mean(Therapy_end_EMG_epochs_with_base_correct(:,:,5)));
           Therapy_end_extension_max = max(mean(Therapy_end_EMG_epochs_with_base_correct(:,:,6)));
        elseif strcmp(Impaired_hand{subj_n},'R')
           Therapy_start_flexion_max = max(mean(Therapy_start_EMG_epochs_with_base_correct(:,:,7)));
           Therapy_start_extension_max = max(mean(Therapy_start_EMG_epochs_with_base_correct(:,:,8)));
           Therapy_end_flexion_max = max(mean(Therapy_end_EMG_epochs_with_base_correct(:,:,7)));
           Therapy_end_extension_max = max(mean(Therapy_end_EMG_epochs_with_base_correct(:,:,8)));
        end
                
        % Normalize by the maximum
        Therapy_start_EMG_epochs_with_base_correct(:,:,5) = Therapy_start_EMG_epochs_with_base_correct(:,:,5)/Therapy_start_flexion_max;
        Therapy_start_EMG_epochs_with_base_correct(:,:,6) = Therapy_start_EMG_epochs_with_base_correct(:,:,6)/Therapy_start_extension_max;
        Therapy_start_EMG_epochs_with_base_correct(:,:,7) = Therapy_start_EMG_epochs_with_base_correct(:,:,7)/Therapy_start_flexion_max;
        Therapy_start_EMG_epochs_with_base_correct(:,:,8) = Therapy_start_EMG_epochs_with_base_correct(:,:,8)/Therapy_start_extension_max;
        
        Therapy_end_EMG_epochs_with_base_correct(:,:,5) = Therapy_end_EMG_epochs_with_base_correct(:,:,5)/Therapy_end_flexion_max;
        Therapy_end_EMG_epochs_with_base_correct(:,:,6) = Therapy_end_EMG_epochs_with_base_correct(:,:,6)/Therapy_end_extension_max;
        Therapy_end_EMG_epochs_with_base_correct(:,:,7) = Therapy_end_EMG_epochs_with_base_correct(:,:,7)/Therapy_end_flexion_max;
        Therapy_end_EMG_epochs_with_base_correct(:,:,8) = Therapy_end_EMG_epochs_with_base_correct(:,:,8)/Therapy_end_extension_max;        
        
        deg_freedom_start = (size(Therapy_start_EMG_epochs_with_base_correct,1) - 1)*ones(8,1);
        deg_freedom_end = (size(Therapy_end_EMG_epochs_with_base_correct,1) - 1)*ones(8,1);
        % Find trials with very high values of unimpaired EMG and remove
        % them from computing mean and SE
        if strcmp(Impaired_hand{subj_n},'L')
           Therapy_start_eliminate_unimpaired_flex_trials = find(max(Therapy_start_EMG_epochs_with_base_correct(:,:,7),[],2) >= 20);
           Therapy_start_eliminate_unimpaired_ext_trials = find(max(Therapy_start_EMG_epochs_with_base_correct(:,:,8),[],2) >= 20);
           Therapy_end_eliminate_unimpaired_flex_trials = find(max(Therapy_end_EMG_epochs_with_base_correct(:,:,7),[],2) >= 20);
           Therapy_end_eliminate_unimpaired_ext_trials = find(max(Therapy_end_EMG_epochs_with_base_correct(:,:,8),[],2) >= 20);
           % For trials to be eliminated, replace with NaN so that it can
           % omitted when calculating mean
           Therapy_start_EMG_epochs_with_base_correct(Therapy_start_eliminate_unimpaired_flex_trials,:,7) = NaN;
           Therapy_start_EMG_epochs_with_base_correct(Therapy_start_eliminate_unimpaired_ext_trials,:,8) = NaN;
           Therapy_end_EMG_epochs_with_base_correct(Therapy_end_eliminate_unimpaired_flex_trials,:,7) = NaN;
           Therapy_end_EMG_epochs_with_base_correct(Therapy_end_eliminate_unimpaired_ext_trials,:,8) = NaN;
           % Adjust degress of freedom
           deg_freedom_start(7) = deg_freedom_start(7) - length(Therapy_start_eliminate_unimpaired_flex_trials);
           deg_freedom_start(8) = deg_freedom_start(8) - length(Therapy_start_eliminate_unimpaired_ext_trials);
           deg_freedom_end(7) = deg_freedom_end(7) - length(Therapy_end_eliminate_unimpaired_flex_trials);
           deg_freedom_end(8) = deg_freedom_end(8) - length(Therapy_end_eliminate_unimpaired_ext_trials);
        else
           Therapy_start_eliminate_unimpaired_flex_trials = find(max(Therapy_start_EMG_epochs_with_base_correct(:,:,5),[],2) >= 20);
           Therapy_start_eliminate_unimpaired_ext_trials = find(max(Therapy_start_EMG_epochs_with_base_correct(:,:,6),[],2) >= 20);
           Therapy_end_eliminate_unimpaired_flex_trials = find(max(Therapy_end_EMG_epochs_with_base_correct(:,:,5),[],2) >= 20);
           Therapy_end_eliminate_unimpaired_ext_trials = find(max(Therapy_end_EMG_epochs_with_base_correct(:,:,6),[],2) >= 20);
           % For trials to be eliminated, replace with NaN so that it can
           % omitted when calculating mean
           Therapy_start_EMG_epochs_with_base_correct(Therapy_start_eliminate_unimpaired_flex_trials,:,5) = NaN;
           Therapy_start_EMG_epochs_with_base_correct(Therapy_start_eliminate_unimpaired_ext_trials,:,6) = NaN;
           Therapy_end_EMG_epochs_with_base_correct(Therapy_end_eliminate_unimpaired_flex_trials,:,5) = NaN;
           Therapy_end_EMG_epochs_with_base_correct(Therapy_end_eliminate_unimpaired_ext_trials,:,6) = NaN;
           % Adjust degrees of freedom
           deg_freedom_start(5) = deg_freedom_start(5) - length(Therapy_start_eliminate_unimpaired_flex_trials);
           deg_freedom_start(6) = deg_freedom_start(6) - length(Therapy_start_eliminate_unimpaired_ext_trials);
           deg_freedom_end(5) = deg_freedom_end(5) - length(Therapy_end_eliminate_unimpaired_flex_trials);
           deg_freedom_end(6) = deg_freedom_end(6) - length(Therapy_end_eliminate_unimpaired_ext_trials);
        end        
        t_value_start = tinv(1 - 0.05/2, deg_freedom_start);       
        t_value_end = tinv(1 - 0.05/2, deg_freedom_end);
        
        for channel_cnt = 1:no_channels                    
            Therapy_start_EMG_average(channel_cnt,:) =  mean(Therapy_start_EMG_epochs_with_base_correct(:,:,channel_cnt),'omitnan');
            Therapy_start_EMG_average_SE(channel_cnt,:) = t_value_start(channel_cnt).*std(Therapy_start_EMG_epochs_with_base_correct(:,:,channel_cnt),'omitnan')/sqrt(size(Therapy_start_EMG_epochs_with_base_correct,1));
            Therapy_end_EMG_average(channel_cnt,:) =  mean(Therapy_end_EMG_epochs_with_base_correct(:,:,channel_cnt),'omitnan');
            Therapy_end_EMG_average_SE(channel_cnt,:) = t_value_end(channel_cnt).*std(Therapy_end_EMG_epochs_with_base_correct(:,:,channel_cnt),'omitnan')/sqrt(size(Therapy_end_EMG_epochs_with_base_correct,1));
        end
        
        figure('Position',[300 5 4*116 3*116],'NumberTitle', 'off', 'Name', [Subject_names{subj_n}, ' , impaired = ', Impaired_hand{subj_n}]); 
        Splot = tight_subplot(2,2,[0.1 0.05],[0.1 0.05],[0.1 0.2]);    
        
        axes(Splot(1)); hold on; grid on; 
        if strcmp(Impaired_hand{subj_n},'L')
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(5,:),'Color',dark_blue_color,'LineWidth',1);
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(7,:),'Color',pink_color,'LineWidth',1);
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(5,:) + Therapy_start_EMG_average_SE(5,:),...
               Therapy_start_EMG_average(5,:) - Therapy_start_EMG_average_SE(5,:),...
               dark_blue_color,dark_blue_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(7,:) + Therapy_start_EMG_average_SE(7,:),...
               Therapy_start_EMG_average(7,:) - Therapy_start_EMG_average_SE(7,:),...
               pink_color,pink_color,true,0.5);
        else
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(7,:),'Color',dark_blue_color,'LineWidth',1);
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(5,:),'Color',pink_color,'LineWidth',1); 
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(7,:) + Therapy_start_EMG_average_SE(7,:),...
               Therapy_start_EMG_average(7,:) - Therapy_start_EMG_average_SE(7,:),...
               dark_blue_color,dark_blue_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(5,:) + Therapy_start_EMG_average_SE(5,:),...
               Therapy_start_EMG_average(5,:) - Therapy_start_EMG_average_SE(5,:),...
               pink_color,pink_color,true,0.5);
        end
        xlim([-1 3]);
        ylim([-0.1 1.3]);
        title('Therapy Start');
        ylabel('Flexor EMG (a.u.)');
        set(gca,'YTick',[0, 1],'YTickLabel',{'0', '1'},'XTick',[-2 -1 0 1 2 3],'XTickLabel',{'-2' '-1' '0' '1' '2' '3'});
        
        axes(Splot(2)); hold on; grid on; 
        if strcmp(Impaired_hand{subj_n},'L')
           f1 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(5,:),'Color',dark_blue_color,'LineWidth',1);
           f2 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(7,:),'Color',pink_color,'LineWidth',1);
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(5,:) + Therapy_end_EMG_average_SE(5,:),...
               Therapy_end_EMG_average(5,:) - Therapy_end_EMG_average_SE(5,:),...
               dark_blue_color,dark_blue_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(7,:) + Therapy_end_EMG_average_SE(7,:),...
               Therapy_end_EMG_average(7,:) - Therapy_end_EMG_average_SE(7,:),...
               pink_color,pink_color,true,0.5);
        else
           f1 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(7,:),'Color',dark_blue_color,'LineWidth',1);
           f2 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(5,:),'Color',pink_color,'LineWidth',1); 
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(7,:) + Therapy_end_EMG_average_SE(7,:),...
               Therapy_end_EMG_average(7,:) - Therapy_end_EMG_average_SE(7,:),...
               dark_blue_color,dark_blue_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(5,:) + Therapy_end_EMG_average_SE(5,:),...
               Therapy_end_EMG_average(5,:) - Therapy_end_EMG_average_SE(5,:),...
               pink_color,pink_color,true,0.5);
        end
        xlim([-1 3]);
        ylim([-0.1 1.3]);
        title('Therapy End');
        l1 = legend([f1, f2],{'Impaired', 'Unimpaired'},'Location','northeastoutside');
        l1.Position(1) = 0.78;
        set(gca,'YTick',[0, 1],'YTickLabel',{'0', '1'},'XTick',[-1 0 1 2 3],'XTickLabel',{'-1' '0' '1' '2' '3'});
        
        axes(Splot(3)); hold on; grid on; 
        if strcmp(Impaired_hand{subj_n},'L')
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(6,:),'Color',purple_color,'LineWidth',1);
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(8,:),'Color',green_color,'LineWidth',1);
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(6,:) + Therapy_start_EMG_average_SE(6,:),...
               Therapy_start_EMG_average(6,:) - Therapy_start_EMG_average_SE(6,:),...
               purple_color,purple_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(8,:) + Therapy_start_EMG_average_SE(8,:),...
               Therapy_start_EMG_average(8,:) - Therapy_start_EMG_average_SE(8,:),...
               green_color,green_color,true,0.5);
        else
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(8,:),'Color',purple_color,'LineWidth',1);
           plot(All_EMG_Epochs{2}.move_erp_time, Therapy_start_EMG_average(6,:),'Color',green_color,'LineWidth',1); 
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(8,:) + Therapy_start_EMG_average_SE(8,:),...
               Therapy_start_EMG_average(8,:) - Therapy_start_EMG_average_SE(8,:),...
               purple_color,purple_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_start_EMG_average(6,:) + Therapy_start_EMG_average_SE(6,:),...
               Therapy_start_EMG_average(6,:) - Therapy_start_EMG_average_SE(6,:),...
               green_color,green_color,true,0.5);
        end
        xlim([-1 3]);
        ylim([-0.1 1.3]);        
        ylabel('Extensor EMG (a.u.)');
        xlabel('Time (sec.)');
        set(gca,'YTick',[0, 1],'YTickLabel',{'0', '1'},'XTick',[-1 0 1 2 3],'XTickLabel',{'-1' '0' '1' '2' '3'});
        
        axes(Splot(4)); hold on; grid on; 
        if strcmp(Impaired_hand{subj_n},'L')
           e1 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(6,:),'Color',purple_color,'LineWidth',1);
           e2 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(8,:),'Color',green_color,'LineWidth',1);
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(6,:) + Therapy_end_EMG_average_SE(6,:),...
               Therapy_end_EMG_average(6,:) - Therapy_end_EMG_average_SE(6,:),...
               purple_color,purple_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(8,:) + Therapy_end_EMG_average_SE(8,:),...
               Therapy_end_EMG_average(8,:) - Therapy_end_EMG_average_SE(8,:),...
               green_color,green_color,true,0.5);
        else
           e1 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(8,:),'Color',purple_color,'LineWidth',1);
           e2 = plot(All_EMG_Epochs{2}.move_erp_time, Therapy_end_EMG_average(6,:),'Color',green_color,'LineWidth',1); 
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(8,:) + Therapy_end_EMG_average_SE(8,:),...
               Therapy_end_EMG_average(8,:) - Therapy_end_EMG_average_SE(8,:),...
               purple_color,purple_color,true,0.5);           
           jbfill(All_EMG_Epochs{2}.move_erp_time,...
               Therapy_end_EMG_average(6,:) + Therapy_end_EMG_average_SE(6,:),...
               Therapy_end_EMG_average(6,:) - Therapy_end_EMG_average_SE(6,:),...
               green_color,green_color,true,0.5);
        end
        xlim([-1 3]);
        ylim([-0.1 1.3]);
        xlabel('Time (sec.)');
        l2 = legend([e1, e2],{'Impaired', 'Unimpaired'},'Location','northeastoutside');
        l2.Position(1) = 0.78;
        set(gca,'YTick',[0, 1],'YTickLabel',{'0', '1'},'XTick',[-1 0 1 2 3],'XTickLabel',{'-1' '0' '1' '2' '3'});
     end
    
end