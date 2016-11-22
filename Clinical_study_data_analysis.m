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
Subject_names = {'S9014','S9012','S9011','S9010','S9009','S9007'};
Subject_numbers = [9014,9012,9011,9010,9009,9007];
Subject_labels = {'S6','S5','S4','S3','S2','S1'};
Subject_velocity_threshold = [1.17,1.28,1.16,1.03,1.99,1.19];
ACC_marker_color = {'--ob','-sk','-vr','-^m','--ok','-sb'};
Latency_marker_color = {'ob','sk','vr','^m','ok','sb'};
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

plot_c3i_poster_plot = 0; 
plot_sfn_poster_plot = 0;
plot_movement_smoothness = 0;

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
    
        % all_subjects_bmi_performance has size (#of sessions, #variables,
        % #subjects)
        all_subjects_session_wise_accuracy_mean = mean(squeeze(all_subjects_bmi_performance(:,14,:)),2);
        all_subjects_session_wise_accuracy_std = std(squeeze(all_subjects_bmi_performance(:,14,:)),[],2);
        all_subjects_session_wise_FPR_mean = mean(squeeze(all_subjects_bmi_performance(:,4,:)),2);
        all_subjects_session_wise_FPR_std = std(squeeze(all_subjects_bmi_performance(:,4,:)),[],2);
        
        % http://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/homepage.htm
        % http://stats.stackexchange.com/questions/6534/how-do-i-calculate-a-weighted-standard-deviation-in-excel
              
        All_session_latencies_mean = squeeze(all_subjects_bmi_performance(:,12,:));
        All_session_latencies_num_trials = squeeze(all_subjects_bmi_performance(:,13,:));
        %all_subjects_sessions_wise_latency_mean = sum(All_session_latencies_mean.*All_session_latencies_num_trials,2)./sum(All_session_latencies_num_trials,2);
        %all_subjects_sessions_wise_latency_std = std(squeeze(all_subjects_bmi_performance(:,13,:)),[],2);
        all_subjects_sessions_wise_latency_mean = [];
        all_subjects_sessions_wise_latency_std = [];
        for ses_num = 1:size(All_session_latencies_mean,1)
            means = All_session_latencies_mean(ses_num,:);
            trials = All_session_latencies_num_trials(ses_num,:);
            N = length(trials);
            weighted_means = sum(means.*trials)/sum(trials);
            weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));
            all_subjects_sessions_wise_latency_mean = [all_subjects_sessions_wise_latency_mean; weighted_means];
            all_subjects_sessions_wise_latency_std = [all_subjects_sessions_wise_latency_std; weighted_standard_deviation];
        end
        
        All_session_likert_mean = squeeze(all_subjects_bmi_performance(:,10,:));
        All_session_likert_num_trials = squeeze(all_subjects_bmi_performance(:,11,:));
%         all_subjects_sessions_wise_likert_mean = sum(All_session_likert_mean.*All_session_likert_num_trials,2)./sum(All_session_likert_num_trials,2);
%         all_subjects_sessions_wise_likert_std = std(squeeze(all_subjects_bmi_performance(:,13,:)),[],2);
        all_subjects_sessions_wise_likert_mean = [];
        all_subjects_sessions_wise_likert_std = [];
         for ses_num = 1:size(All_session_likert_mean,1)
            means = All_session_likert_mean(ses_num,:);
            trials = All_session_likert_num_trials(ses_num,:);
            N = length(trials);
            weighted_means = sum(means.*trials)/sum(trials);
            weighted_standard_deviation = sqrt(sum(((means - weighted_means).^2).*trials)/(((N-1)/N)*sum(trials)));
            all_subjects_sessions_wise_likert_mean = [all_subjects_sessions_wise_likert_mean weighted_means];
            all_subjects_sessions_wise_likert_std = [all_subjects_sessions_wise_likert_std weighted_standard_deviation];
        end
        
        
 %%       
        figure('Position',[700 100 7*116 6*116]); 
        Cplot = tight_subplot(4,1,[0.02 0.02],[0.1 0.05],[0.05 0.05]);
        
        axes(Cplot(1)); hold on;
        %h_acc = plot(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean,'-sk','MarkerFaceColor','k','LineWidth',1);
        h_acc = errorbar(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_accuracy_mean, 100*all_subjects_session_wise_accuracy_std,'^k','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);
        ylim([0 110]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        %set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        %title('Closed-loop BMI performance for ongoing clinical study','FontSize',paper_font_size);
        %title(Subject_names{subj_n},'FontSize',paper_font_size);
        %ylabel({'True Positive'; 'Rate (%)'},'FontSize',paper_font_size);
%         ylabel({'BMI Accuracy', 'Mean \pm S.D. (%)'},'FontSize',paper_font_size);
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
        h_acc2 = errorbar(1:size(all_subjects_bmi_performance,1), 100*all_subjects_session_wise_FPR_mean, 100*all_subjects_session_wise_FPR_std,'vk','MarkerFaceColor','w','LineWidth',1,'MarkerSize',10);
        ylim([0 110]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',' ');
%         ylabel({'False', 'Positives (%)'},'FontSize',paper_font_size);
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
        h_latency = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_latency_mean, all_subjects_sessions_wise_latency_std,'ok','MarkerFaceColor','k','LineWidth',1,'MarkerSize',10);
        ylim([-200 200]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[-200 0 200],'YtickLabel',{'-200' '0' '200'},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',' ');
%         ylabel({'Detection', 'Latency (msec.)'},'FontSize',paper_font_size);
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
        h_likert(subj_n) = errorbar(1:size(all_subjects_bmi_performance,1), all_subjects_sessions_wise_likert_mean, all_subjects_sessions_wise_likert_std, 'ok','MarkerFaceColor','w','LineWidth',1,'MarkerSize',10);
        ylim([0 4]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[1 2 3],'YtickLabel',{' '},'FontSize',paper_font_size);
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'},'FontSize',paper_font_size);
%         ylabel({'Subject Rating'},'FontSize',paper_font_size);
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
