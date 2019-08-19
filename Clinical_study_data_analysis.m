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
paper_font_size = 12;
x_axis_deviation = [0.15,0.05,-0.05,-0.15];
directory = 'D:\NRI_Project_Data\Clinical_study_Data\';
Subject_names = {'S9023','S9020','S9018','S9017','S9014','S9012','S9011','S9010','S9009','S9007'};
Impaired_hand = {'R', 'L', 'L','R', 'R', 'R','L', 'L', 'L', 'R'};
Subject_numbers = [9023,9020,9018,9017,9014,9012,9011,9010,9009,9007];
Subject_labels = {'S10','S9','S8','S7','S6','S5','S4','S3','S2','S1'};
Subject_velocity_threshold = [1.44,1.5,1.5,1.1,1.17,1.28,1.16,1.03,1.99,1.19];
Cond_num = [1, 3, 1, 1, 1, 1, 1, 1, 3, 1];
Block_num = [160, 160, 160, 160, 160, 160, 170, 160, 150, 160];

% Subject groupings based on pre- and post- clinical scores
Subjects_all = 1:length(Subject_numbers);
Subjects_severe_moderate = [2,7,9];
Subjects_moderate_mild = [1,3:6, 8, 10];
Subjects_FMA_above_MCID = [10, 8, 6, 3, 2, 1];
Subjects_FMA_below_MCID = [9, 7, 5];
Subjects_ARAT_above_MCID = [10, 8, 5, 4, 3]; %[10, 5, 4, 3];
Subjects_ARAT_below_MCID = [9, 7, 6, 2, 1]; %[9, 8, 7, 6, 2, 1];

ACC_marker_color = {'-sy','-^m','--ok','-sb','--ob','-sk','-vr','-^m','--ok','-sb'};
Latency_marker_color = {'^m','ok','sb','ob','sk','vr','^m','ok','sb'};
TPR_marker_color = {'--ok','--sk','--ok','--*k','--^k'};
FPR_marker_color = {'--ok','-sk','-ok','*k','^k'};
Marker_face_color = {'w','k','r','m','w','b'};
Sig_line_color = {'--k','--k','--k','--k','--k','--k'};
%likert_marker_color = {'-ok','-sk','-ok','--^k','-vk'};
h_acc = zeros(length(Subject_names),1);
h_tpr = zeros(length(Subject_names),1);
h_fpr = zeros(length(Subject_names),1);
h_intent = zeros(length(Subject_names),1);
h_likert = zeros(length(Subject_names),1);

 EMG_channel_nos = [17 22 41 42 45 46 51 55];
% [ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 
 
segment_resting_eegdata = 0; 
plot_c3i_poster_plot = 0; 
plot_sfn_poster_plot = 0;
plot_movement_smoothness = 0;
compute_statistics = 0;
perform_posthoc_RP_analysis = 0;
correlate_RP_changes = 1;
compute_trial_averages = 0;

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
     all_subjects_bmi_performance = zeros(12,14,length(Subject_names));
     
    for subj_n = 1:length(Subject_names)
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
                                                            Session_detection_latency_mean Session_detection_latency_num_trials Session_accuracy]];   

        end
        
        all_subjects_bmi_performance(:,:,subj_n) =bmi_performance;
    end
    
  
    %%
        % Determine t-statistics for 95% C.I.
        %http://www.mathworks.com/matlabcentral/answers/20373-how-to-obtain-the-t-value-of-the-students-t-distribution-with-given-alpha-df-and-tail-s
        deg_freedom = length(Subject_numbers) - 1;
        t_value = tinv(1 - 0.05/2, deg_freedom);
    
        % all_subjects_bmi_performance has size (#of sessions, #variables, #subjects)
        all_subjects_session_wise_accuracy_mean = mean(squeeze(all_subjects_bmi_performance(:,14,Subjects_FMA_below_MCID)),2);
        all_subjects_session_wise_accuracy_std = std(squeeze(all_subjects_bmi_performance(:,14,Subjects_FMA_below_MCID)),[],2); 
%         all_subjects_session_wise_accuracy_SE = (t_value.*std(squeeze(all_subjects_bmi_performance(:,14,Subjects_all)),[],2))/sqrt( length(Subject_numbers)); 
        
        all_subjects_session_wise_FPR_mean = mean(squeeze(all_subjects_bmi_performance(:,4,Subjects_FMA_below_MCID)),2);
        all_subjects_session_wise_FPR_std = std(squeeze(all_subjects_bmi_performance(:,4,Subjects_FMA_below_MCID)),[],2);
        
        % http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/homepage.htm
        % http://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
              
        All_session_latencies_mean = squeeze(all_subjects_bmi_performance(:,12,Subjects_FMA_below_MCID));
        All_session_latencies_num_trials = squeeze(all_subjects_bmi_performance(:,13,Subjects_FMA_below_MCID));
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
        
        All_session_likert_mean = squeeze(all_subjects_bmi_performance(:,10,Subjects_FMA_below_MCID));
        All_session_likert_num_trials = squeeze(all_subjects_bmi_performance(:,11,Subjects_FMA_below_MCID));
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
        
        % Calculate mean+/-s.d. of accuracy, fpr, latency, likert scores overall subjects and sessions
        % Added on 11th July, 2018
        Ovracc = squeeze(all_subjects_bmi_performance(:,14,Subjects_FMA_below_MCID));
        disp(['Overall BMI accuracy = ', num2str(mean(Ovracc(:))*100), ' +/- ', num2str(std(Ovracc(:))*100), ' %'])  
        
        Ovrfpr = squeeze(all_subjects_bmi_performance(:,4,Subjects_FMA_below_MCID));
        disp(['Overall false positive rate = ', num2str(mean(Ovrfpr(:))*100), ' +/- ', num2str(std(Ovrfpr(:))*100), ' %']) 
        
        OvrLatency_mean = sum(sum(All_session_latencies_mean.*All_session_latencies_num_trials))/sum(All_session_latencies_num_trials(:)); %weighted mean
        N = length(All_session_latencies_num_trials(:));
        OvrLatency_std = sqrt(sum(sum(((All_session_latencies_mean - OvrLatency_mean).^2).*All_session_latencies_num_trials))/(((N-1)/N)*sum(All_session_latencies_num_trials(:))));
        disp(['Overall latency = ', num2str(OvrLatency_mean), ' +/- ', num2str(OvrLatency_std), ' ms']) 
        
        OvrLikert_mean = sum(sum(All_session_likert_mean.*All_session_likert_num_trials))/sum(All_session_likert_num_trials(:)); %weighted mean
        N = length(All_session_likert_num_trials(:));
        OvrLikert_std = sqrt(sum(sum(((All_session_likert_mean - OvrLikert_mean).^2).*All_session_likert_num_trials))/(((N-1)/N)*sum(All_session_likert_num_trials(:))));
        disp(['Overall likert score = ', num2str(OvrLikert_mean), ' +/- ', num2str(OvrLikert_std)])
        
        Num_of_valid_trials = squeeze(all_subjects_bmi_performance(:,5,:));
        mean(Num_of_valid_trials(:));
        std(Num_of_valid_trials(:));
        
 %%       
        figure('Position',[700 100 7*116 6*116]); 
        Cplot = tight_subplot(4,1,[0.02 0.02],[0.1 0.05],[0.1 0.05]);
        
        axes(Cplot(1)); hold on;
        %h_acc = plot(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean,'-sk','MarkerFaceColor','k','LineWidth',1);
        h_acc = errorbar(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean, 100*all_subjects_session_wise_accuracy_std,'^k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',8);
        ylim([0 110]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        %set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
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
        if mlr_acc.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(all_subjects_bmi_performance,1)]]*mlr_acc.Coefficients.Estimate;
            plot(gca,[0  size(all_subjects_bmi_performance,1)+0.5],line_regress,'--r','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_acc.Coefficients.Estimate(2))},'FontSize',paper_font_size,'Color','r');
        end
         hold off;
        
% %         axes(Cplot(1)); hold on;
% %         h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),ACC_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
% %         ylim([0 130]); 
% %         xlim([0 12.5]);
% %         set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
% %         set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
% %         %title('Closed-loop BMI performance for ongoing clinical study','FontSize',paper_font_size);
% %         title(Subject_names{subj_n},'FontSize',paper_font_size);
% %         %ylabel({'True Positive'; 'Rate (%)'},'FontSize',paper_font_size);
% %         ylabel({'BMI Accuracy (%)'},'FontSize',paper_font_size);
% %         xlabel('Sessions','FontSize',10);
% %         for ses_num = 1:length(unique_session_nos)
% %             plot_ses_performance = bmi_performance_blockwise(bmi_performance_blockwise(:,1) == subj_n &...
% %                                                                                                            bmi_performance_blockwise(:,2) == unique_session_nos(ses_num), :);
% %             no_of_blocks = size(plot_ses_performance,1);
% %             plot_x_values = linspace((ses_num - 1 + 0.1),(ses_num - 0.1),no_of_blocks);
% %             plot(plot_x_values, 100*plot_ses_performance(:,6), 'sk','MarkerSize', 4)
% %         end
        
        axes(Cplot(2)); hold on;
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        h_acc2 = errorbar(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_FPR_mean, 100*all_subjects_session_wise_FPR_std,'vk','MarkerFaceColor','k','LineWidth',1,'MarkerSize',8);
        ylim([0 110]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        ylabel({'False Positives (%)'},'FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        
        mlr_fpr = LinearModel.fit(1:size(all_subjects_bmi_performance,1),100*all_subjects_session_wise_FPR_mean);
        if mlr_fpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_fpr.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.5],line_regress,'--r','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.5,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_fpr.Coefficients.Estimate(2))},'FontSize',paper_font_size,'Color','r');
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        axes(Cplot(3)); hold on;
        h_latency = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_latency_mean, all_subjects_sessions_wise_latency_std,'ok','MarkerFaceColor','k','LineWidth',1,'MarkerSize',8);
        ylim([-200 200]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[-200 0 200],'YtickLabel',{'-200' '0' '200'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        ylabel('Latency (ms)','FontSize',paper_font_size);
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
        h_likert(subj_n) = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_likert_mean, all_subjects_sessions_wise_likert_std, 'ok','MarkerFaceColor','k','LineWidth',1,'MarkerSize',8);
        ylim([0 4]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[1 2 3],'YtickLabel',{'1', '2', '3'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'},'FontSize',paper_font_size);
        ylabel('Subject Rating','FontSize',paper_font_size);
%         set(get(gca,'YLabel'),'Rotation',0); 
        xlabel('Therapy sessions','FontSize',paper_font_size);
        mlr_likert = LinearModel.fit(1:size(all_subjects_bmi_performance,1),all_subjects_sessions_wise_likert_mean);
        if mlr_likert.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(all_subjects_bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
            plot(gca,[0  size(all_subjects_bmi_performance,1)+0.5],line_regress,'--r','LineWidth',0.5); hold on;
            text(size(all_subjects_bmi_performance,1)+0.5,line_regress(2),{sprintf(' %.2f*',mlr_likert.Coefficients.Estimate(2))},'FontSize',paper_font_size,'Color','r');
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        annotation('textbox',[0.9 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none','FontSize',paper_font_size);    
    
    
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
        h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),ACC_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
        %plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),'-ok','LineWidth',1);
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor','none','LineWidth',1);
        ylim([0 110]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
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
   for subj_n = 3:5 %length(Subject_names)
        All_Smart_Features = cell(1,13);
               
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
            
            Time_to_Trigger = session_performance(:,[2,7]); %col 7 - Time to Trigger                        
            unique_block_nos = unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2));            
            readbv_files = 1;
            process_raw_emg = 1;
            process_raw_eeg = 1;
            extract_epochs = 1;
            remove_corrupted_epochs = [];
            Posthoc_Average = Calculate_RPs_sessionwise(Subject_names{subj_n}, unique_session_nos(ses_num), readbv_files, unique_block_nos, remove_corrupted_epochs, Impaired_hand{subj_n}, Time_to_Trigger, Performance, process_raw_emg, process_raw_eeg, extract_epochs);
            All_Smart_Features{ses_num+1} = Posthoc_Average.Smart_Features;            
            All_RPs{ses_num+1} = [Posthoc_Average.move_avg_channels; Posthoc_Average.move_erp_time];                                    
        end            
        save([directory 'Subject_' Subject_names{subj_n} '\' 'All_Smart_Features.mat'],'All_Smart_Features','All_RPs');
        
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
               elseif cell_num == 13    % for 9007, 9014 use 12, 9009 use 13
                   plot(All_RPs{cell_num}(65,:),All_RPs{cell_num}(channels(chn),:),'-b'); 
               else
                   %plot(All_RPs{cell_num}(65,:),All_RPs{cell_num}(channels(chn),:),'Color',[1 1 1]*(14 - cell_num)/14); 
               end
            end
%             title(chanlabels(chn));
            set(gca,'YDir','reverse');
            ylim([-5    5])
            xlim([-2.25 1])
        end
        legend('Therapy start','Therapy end')
        
%         extract_emg_epochs = 1;        
%         if extract_emg_epochs 
%             figure; 
%             emg_colors = {'-b', '-r', '-k', '-m'};
%             for ses_num = 1:length(unique_session_nos)                
%                 filename = dir([directory, 'Subject_', Subject_names{subj_n}, '\', Subject_names{subj_n},'_Session', num2str(unique_session_nos(ses_num)), '\',...
%                     Subject_names{subj_n} '_ses' num2str(unique_session_nos(ses_num)) '_closeloop_block' num2str(length(unique_block_nos)*20) '_emg_*.set']);
%                 EEG = pop_loadset(filename.name, filename.folder);    
%                 [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
%                 eeglab redraw;        
% 
%                 EEG = pop_epoch( EEG, {  'EMGonset'  }, [-3.5 2], 'newname', 'EMG_epochs', 'epochinfo', 'yes');
%                 EEG = eeg_checkset( EEG );
%                 
%                 for sp = 1:4
%                     subplot(4,12,12*(sp-1)+ses_num); hold on; grid on;                    
%                     plot(squeeze(EEG.data(4+sp,:,:)),emg_colors{sp});
%                 end                
%             end
%         end
   end
end

%%

if correlate_RP_changes == 1
    
    
    NumElements = zeros(10,12);
    Subject_wise_RP_metrics = zeros(64,4,10);
    
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
                [no_epochs,~,no_channels] = size(Posthoc_Average.move_epochs);            
                for epoch_cnt = 1:no_epochs
                    for channel_cnt = 1:no_channels
                        Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt) = Posthoc_Average.move_epochs(epoch_cnt,:,channel_cnt) ...
                            - Posthoc_Average.move_mean_baseline(channel_cnt,epoch_cnt);                    
                    end
                end            
                Posthoc_Average.move_epochs(Posthoc_Average.epochs_with_artefacts, :, :) = [];

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
                Therapy_start_grand_average(channel_cnt,:) =  mean(Therapy_start_move_epochs(:,:,channel_cnt));
                Therapy_start_grand_average_SE(channel_cnt,:) = t_value_start.*std(Therapy_start_move_epochs(:,:,channel_cnt))/sqrt(size(Therapy_start_move_epochs,1));
                Therapy_end_grand_average(channel_cnt,:) =  mean(Therapy_end_move_epochs(:,:,channel_cnt));
                Therapy_end_grand_average_SE(channel_cnt,:) = t_value_end.*std(Therapy_end_move_epochs(:,:,channel_cnt))/sqrt(size(Therapy_end_move_epochs,1));
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
            switch (subj_n)
                case 1
                    
                case 2
                    Therapy_start_RP_peakIndex([48,49]) = [817, 805] - 500;
                    
                case 3
                    Therapy_start_RP_peakIndex([14,49]) = [778, 778] - 500;
                    
                case 4
                    Therapy_start_RP_peakIndex([48]) = [601] - 500;
                    Therapy_end_RP_peakIndex([14]) = [848] - 500;
                    
                case 5
                    Therapy_start_RP_peakIndex([48]) = [811] - 500;
                    Therapy_end_RP_peakIndex([49]) = [605] - 500;
                    
                case 6
                    Therapy_start_RP_peakIndex([48, 14]) = [800, 800] - 500;
                    
                case 7
                    Therapy_end_RP_peakIndex([48,14,49]) = [550, 775, 743] - 500;
                    
                case 8
                    
                case 9
                    Therapy_start_RP_peakIndex([48, 14]) = [651, 531] - 500;
                    Therapy_end_RP_peakIndex([49]) = [801] - 500;
                    
                case 10
                    Therapy_start_RP_peakIndex([14]) = [719] - 500;
                    Therapy_end_RP_peakIndex([14,49]) = [738, 715] - 500;
                    
            end
                    
                    
            
            Therapy_start_RP_peakIndex = Therapy_start_RP_peakIndex + 500;            
            slope_begin = Therapy_start_RP_peakIndex - 200;
            slope_end = Therapy_start_RP_peakIndex;
            Therapy_start_RP_slope = zeros(64,1);
            for ch = 1:64
                Therapy_start_RP_slope(ch) = (Therapy_start_grand_average(ch,slope_begin(ch)) - Therapy_start_grand_average(ch,slope_end(ch)))/1.0;
            end
            
            [Therapy_end_RP_peak, Therapy_end_RP_peakIndex] = min(Therapy_end_grand_average(:, find(move_erp_time == -1.0,1):find(move_erp_time == 1.0,1)), [], 2);
            Therapy_end_RP_peakIndex = Therapy_end_RP_peakIndex + 500;            
            slope_begin = Therapy_end_RP_peakIndex - 200; %200
            slope_end = Therapy_end_RP_peakIndex;
            Therapy_end_RP_slope = zeros(64,1);
            for ch = 1:64
                Therapy_end_RP_slope(ch) = (Therapy_end_grand_average(ch,slope_begin(ch)) - Therapy_end_grand_average(ch,slope_end(ch)))/1.0;
            end 
            
            
            Channels_nos = [48,  14,  49];                        
            figure;
             for ind4 = 1:length(Channels_nos)
                subplot(1,3,ind4);
                hold on;
                plot(Therapy_start_grand_average(Channels_nos(ind4),:),'r','LineWidth',1);                
                plot(Posthoc_Average.move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:)+ (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);
                plot(Posthoc_Average.move_erp_time,Therapy_start_grand_average(Channels_nos(ind4),:) - (Therapy_start_grand_average_SE(Channels_nos(ind4),:)),'--','Color',[1 0 0],'LineWidth',0.25);
                plot(Therapy_end_grand_average(Channels_nos(ind4),:),'b','LineWidth',1);
                
                % Added 8/21/2015        
                text(-2,-4,int2str(Channels_nos(ind4)),'Color','k','FontWeight','normal','FontSize',paper_font_size-1); 
                set(gca,'YDir','reverse');
                axis([201 951 -10 10]);                
%                 line([0 0],[-30 20],'Color','k','LineWidth',0.5,'LineStyle','--');  
                line([0 1000],[0 0],'Color','k','LineWidth',0.5,'LineStyle','--');  
                line([Therapy_start_RP_peakIndex(Channels_nos(ind4)), Therapy_start_RP_peakIndex(Channels_nos(ind4))],[-30 20],'Color','r','LineWidth',0.5,'LineStyle','--');  
                line([Therapy_end_RP_peakIndex(Channels_nos(ind4)) Therapy_end_RP_peakIndex(Channels_nos(ind4))],[-30 20],'Color','b','LineWidth',0.5,'LineStyle','--');  
    %             plot_ind4 = plot_ind4 + 1;
                grid on;
            end
        end
        Subject_wise_RP_metrics(:,:,subj_n) = [Therapy_start_RP_peak, Therapy_start_RP_slope, ...
            Therapy_end_RP_peak, Therapy_end_RP_slope];        
    end
    
    if compute_trial_averages == 0        
        Subject_wise_RP_peak_therapy_start = zeros(10,3);
        Subject_wise_RP_peak_therapy_end = zeros(10,3);
        Subject_wise_RP_slope_therapy_start = zeros(10,3);
        Subject_wise_RP_slope_therapy_end = zeros(10,3);
        
        for subj_n = 1:length(Subject_names)
            if strcmp(Impaired_hand(subj_n),'R')
               Channels_nos = [48,  14,  49];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement               
            else
               Channels_nos = [49,  14,  48];
               %[Contra-lateral,   Midline, Ipsilateral] - electrode arrangement                 
            end
               Subject_wise_RP_peak_therapy_start(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,1,subj_n);
               Subject_wise_RP_peak_therapy_end(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,3,subj_n);
               Subject_wise_RP_slope_therapy_start(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,2,subj_n);
               Subject_wise_RP_slope_therapy_end(subj_n,:) = Subject_wise_RP_metrics(Channels_nos,4,subj_n);               
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
        ARATchange = [2     1    10     7     6     5     0     9     0    12]';
        RPpeak_change = -1.*((Subject_wise_RP_peak_therapy_end - Subject_wise_RP_peak_therapy_start));
        RPslope_change = ((Subject_wise_RP_slope_therapy_end - Subject_wise_RP_slope_therapy_start));
        
        RP_peak_FMA_corr = corr(RPpeak_change,FMAchange);
        RP_peak_ARAT_corr = corr(RPpeak_change, ARATchange);
        RP_slope_FMA_corr = corr(RPslope_change,FMAchange);
        RP_slope_ARAT_corr = corr(RPslope_change,ARATchange);
        
        figure; 
        for i = 1:3
            subplot(2,3,i); hold on;
            plot(RPpeak_change(:,i), FMAchange, '.k');                    
            box on;
            text(RPpeak_change(:,i), FMAchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
            mlr_fit = LinearModel.fit(RPpeak_change(:,i),FMAchange);            
            line_regress = [ones(2,1) [min(RPpeak_change(:,i)) - 2; max(RPpeak_change(:,i)) + 2]]*mlr_fit.Coefficients.Estimate;
            plot([min(RPpeak_change(:,i)) - 2, max(RPpeak_change(:,i)) + 2],line_regress,'--k','LineWidth',1);                                    
            xlim([min(RPpeak_change(:,i)) - 0.75, max(RPpeak_change(:,i)) + 0.75]); 
            ylim([-2 10]);
            text(min(RPpeak_change(:,i))-0.5, 9.5, ['\rho = ', num2str(RP_peak_FMA_corr(i),2)], 'FontSize', paper_font_size-2, 'Color', [0 0 0]);            
            xlabel('Amplitude change (\muV)');
            switch (i)
                case 1
                    ylabel('FMA change');
                    title('Contralateral elec.');
                case 2                    
                    title('Midline elec.');
                case 3                    
                    title('Ipsilateral elec.');
            end
                    
%             subplot(2,3,i+3); hold on;
%             plot(RPslope_change(:,i), FMAchange, '.b');                    
%             box on;
%             text(RPslope_change(:,i), FMAchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 1]);
%             mlr_fit = LinearModel.fit(RPslope_change(:,i),FMAchange);            
%             line_regress = [ones(2,1) [min(RPslope_change(:,i)) - 2; max(RPslope_change(:,i)) + 2]]*mlr_fit.Coefficients.Estimate;
%             plot([min(RPslope_change(:,i)) - 2, max(RPslope_change(:,i)) + 2],line_regress,'-b','LineWidth',1);                                                 
%             xlim([min(RPslope_change(:,i)) - 0.75, max(RPslope_change(:,i)) + 0.75]); 
%             ylim([-2 10]); 
%             text(min(RPslope_change(:,i))-0.5, 9.5, ['\rho = ', num2str(RP_slope_FMA_corr(i),2)], 'FontSize', 8, 'Color', [0 0 0]);            

            subplot(2,3,i+3); hold on;
            plot(RPpeak_change(:,i), ARATchange, '.k');        
            box on;
            text(RPpeak_change(:,i), ARATchange, cellstr(num2str(flip(1:10)')), 'FontSize', 8, 'Color', [0 0 0]);
            mlr_fit2 = LinearModel.fit(RPpeak_change(:,i),ARATchange);            
            line_regress2 = [ones(2,1) [min(RPpeak_change(:,i)) - 2; max(RPpeak_change(:,i)) + 2]]*mlr_fit2.Coefficients.Estimate;
            plot([min(RPpeak_change(:,i)) - 2, max(RPpeak_change(:,i)) + 2],line_regress2,'--k','LineWidth',1);            
            xlim([min(RPpeak_change(:,i)) - 0.25, max(RPpeak_change(:,i)) + 0.75]); 
            ylim([-2 14]);            
            text(min(RPpeak_change(:,i)), 13, ['\rho = ', num2str(RP_peak_ARAT_corr(i),2)], 'FontSize', paper_font_size-2, 'Color', [0 0 0]);            
            xlabel('Amplitude change (\muV)');
            switch (i)
                case 1
                    ylabel('ARAT change');
                    title('Contralateral elec.');
                case 2                    
                    title('Midline elec.');
                case 3                    
                    title('Ipsilateral elec.');
            end
            
        end                
    end    
end