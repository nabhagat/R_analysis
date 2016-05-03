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
paper_font_size = 10;
directory = 'C:\NRI_BMI_Mahi_Project_files\All_Subjects\';
Subject_name = {'S9007','S9009','S9010','S9011'};
Subject_labels = {'S1','S2','S3','S4'};
Subject_velocity_threshold = [1.19,1.99,1.03,1.16];
ACC_marker_color = {'-sk','-ok','-^k','-vk'};
TPR_marker_color = {'--sk','--ok','--*k','--^k'};
FPR_marker_color = {'-sk','-ok','*k','^k'};
Marker_face_color = {'w','k','w','k'};
Sig_line_color = {'--k','--k','--k','--k'};
boxplot_color = {'r','r'};
likert_marker_color = {'-sb','-or'};
h_acc = zeros(length(Subject_name),1);
h_tpr = zeros(length(Subject_name),1);
h_fpr = zeros(length(Subject_name),1);
h_intent = zeros(length(Subject_name),1);
h_likert = zeros(length(Subject_name),1);

plot_tpr_fpr = 1;
plot_blockwise_accuracy = 0;

if plot_blockwise_accuracy == 1
     figure('Position',[700 100 7*116 6*116]); 
     Cplot = tight_subplot(4,1,[0.05 0.02],[0.1 0.05],[0.1 0.05]);
     
    for subj_n = 1:length(Subject_name)
        bmi_performance = [];
        bmi_performance_blockwise = [];
               
        fileid = dir([directory 'Subject_' Subject_name{subj_n} '\' Subject_name{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_name{subj_n} '\' fileid.name];
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
            
            Session_detection_latency_mean = mean(session_performance(ind_success_valid_trials,22));
            Session_detection_latency_std = std(session_performance(ind_success_valid_trials,22));
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
                                                                           [subj_n unique_session_nos(ses_num) unique_block_nos(block_num) block_TPR block_FPR block_accuracy] ];
            end

            bmi_performance = [bmi_performance;...
                                            [subj_n unique_session_nos(ses_num) session_TPR session_FPR length(ind_valid_trials) length(ind_catch_trials)...
                                                            mean(Session_Intent_per_min) std(Session_Intent_per_min) median(Session_Intent_per_min) Session_likert_mean Session_likert_std...
                                                            Session_detection_latency_mean Session_detection_latency_std Session_accuracy]];   
            %        1          2                 3                  4                 5                        6                                                    7 
            % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
            %                    8                                          9                                                       10                              11
            % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
            %                   12                                                                    13                                                    
            % Session_detection_latency_mean    Session_detection_latency_std
            %                    14
            %       Session_accuracy
        end
                
        axes(Cplot(1)); hold on;
        %h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,3),TPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor','none','LineWidth',1);
        h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),TPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        ylim([0 130]); 
        xlim([0 12.5]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        title('Closed-loop BMI performance for ongoing clinical study','FontSize',paper_font_size);
        %ylabel({'True Positive'; 'Rate (%)'},'FontSize',paper_font_size);
        ylabel({'BMI Accuracy (%)'},'FontSize',paper_font_size);
        xlabel('Sessions','FontSize',10);
        for ses_num = 1:length(unique_session_nos)
            plot_ses_performance = bmi_performance_blockwise(bmi_performance_blockwise(:,1) == subj_n &...
                                                                                                           bmi_performance_blockwise(:,2) == unique_session_nos(ses_num), :);
            no_of_blocks = size(plot_ses_performance,1);
            plot_x_values = linspace((ses_num - 1 + 0.1),(ses_num - 0.1),no_of_blocks);
            plot(plot_x_values, 100*plot_ses_performance(:,6), 'sk','MarkerSize', 4)
        end
        
        mlr_tpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,3));
        if mlr_tpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_tpr.Coefficients.Estimate;
            %plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
            %text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_tpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
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
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        axes(Cplot(2)); hold on;
        h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        ylim([-10 120]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        ylabel({'False Positive'; 'Rate (%)'},'FontSize',paper_font_size);
        mlr_tpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,3));
        
        mlr_fpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,4));
        if mlr_fpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_fpr.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_fpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
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
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        axes(Cplot(3)); hold on;
%         hbox_axes = boxplot(subject_intent_per_min(:,2),subject_intent_per_min(:,1),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o','colors',boxplot_color{subj_n}); % symbol - Outliers take same color as box
%         set(hbox_axes(6,1:size(bmi_performance,1)),'Color',boxplot_color{subj_n});
%         set(hbox_axes(7,1:size(bmi_performance,1)),'MarkerSize',4);
%         set(hbox_axes,'LineWidth',1);
        h_intent(subj_n) = plot(1:size(bmi_performance,1), bmi_performance(:,9),likert_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        ylim([0 20]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 10 20],'YtickLabel',{'0' '10' '20'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
       ylabel({'No. of Intents'; 'detected per min.'; '(median)'},'FontSize',paper_font_size);
        plot1_pos = get(Cplot(1),'Position');
        boxplot_pos = get(gca,'Position');
        set(gca,'Position',[plot1_pos(1) boxplot_pos(2) plot1_pos(3) boxplot_pos(4)]);
        mlr_intents = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,9));
        if mlr_intents.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_intents.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_intents.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
%         axes(Cplot(3)); hold on;
%         CoV = bmi_performance(:,8)./bmi_performance(:,7);
%         h_cov = plot(1:size(bmi_performance,1), CoV,'-xk','MarkerFaceColor','none','LineWidth',1);
%         ylim([0 5]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[0 5],'YtickLabel',{'0' '5'});
%         set(gca,'Xtick',1:12,'Xticklabel',' ');
%         ylabel({'Coefficient of';'variation'},'FontSize',paper_font_size);
%         mlr_CoV = LinearModel.fit(1:size(bmi_performance,1),CoV);
%         if mlr_CoV.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_CoV.Coefficients.Estimate;
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_CoV.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;

%         axes(Cplot(3));hold on;
%         %h_latency = errorbar(1:size(bmi_performance,1), bmi_performance(:,12)./1000, bmi_performance(:,13)./1000, '-ok','MarkerFaceColor','k','LineWidth',1);
%         overall_latency = subject_study_data((subject_study_data (:,22) < 1000) & (subject_study_data (:,22) > -1000),22);
%         [f_lat,x_lat] = hist(overall_latency, -1000:20:1000);
%         h_latency_2 = bar(x_lat,f_lat);
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
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_latency.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;
        
%         figure;
%         
%         [f1,x1] = hist(subject_study_data(:,22),-500:10:500);
%         h_patch(1) = jbfill(x1,f1./trapz(x1,f1), zeros(1,length(-500:10:500)),'r','r',1,0.5);
        
        axes(Cplot(4));hold on;
        h_likert(subj_n) = errorbar(1:size(bmi_performance,1), bmi_performance(:,10), bmi_performance(:,11), likert_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
        ylim([0 4]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[1 2 3],'YtickLabel',{'1' '2' '3'});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        ylabel({'Subject rating'; '(Likert scale)'},'FontSize',paper_font_size);
        xlabel('Sessions','FontSize',10);
        mlr_likert = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,10));
        if mlr_likert.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_likert.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');    
    end
    
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

if plot_tpr_fpr == 1
     figure('Position',[700 100 7*116 2.5*116]); 
     Cplot = tight_subplot(1,1,[0.05 0.02],[0.2 0.05],[0.1 0.05]);
     Subject_wise_performance = [];
     
    for subj_n = 1:length(Subject_name)
        bmi_performance = [];
               
        fileid = dir([directory 'Subject_' Subject_name{subj_n} '\' Subject_name{subj_n} '_session_wise_results*']);
        results_filename = [directory 'Subject_' Subject_name{subj_n} '\' fileid.name];
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
            
            Session_detection_latency_mean = mean(session_performance(ind_success_valid_trials,22));
            Session_detection_latency_std = std(session_performance(ind_success_valid_trials,22));
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
            end

            bmi_performance = [bmi_performance;...
                                            [subj_n unique_session_nos(ses_num) session_TPR session_FPR length(ind_valid_trials) length(ind_catch_trials)...
                                                            mean(Session_Intent_per_min) std(Session_Intent_per_min) median(Session_Intent_per_min) Session_likert_mean Session_likert_std...
                                                            Session_detection_latency_mean Session_detection_latency_std Session_accuracy]];   
            %        1          2                 3                  4                 5                        6                                                    7 
            % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
            %                    8                                          9                                                       10                              11
            % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
            %                   12                                                                    13                                                      14
            % Session_detection_latency_mean    Session_detection_latency_std           Session_accuracy       
        end
                
        axes(Cplot(1)); hold on;
        h_tpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),ACC_marker_color{subj_n},'MarkerFaceColor',Marker_face_color{subj_n},'LineWidth',1);
        %plot(1:size(bmi_performance,1), 100*bmi_performance(:,14),'-ok','LineWidth',1);
        %h_fpr(subj_n) = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),FPR_marker_color{subj_n},'MarkerFaceColor','none','LineWidth',1);
        ylim([0 110]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        %title('Closed-loop BMI performance for ongoing clinical study (4 subjects)','FontSize',paper_font_size);
        ylabel({'BMI Accuracy (%)'},'FontSize',paper_font_size);
         xlabel('Sessions','FontSize',10);
         
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
        set(gca,'Xgrid','on');
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
        
%         axes(Cplot(3)); hold on;
%         CoV = bmi_performance(:,8)./bmi_performance(:,7);
%         h_cov = plot(1:size(bmi_performance,1), CoV,'-xk','MarkerFaceColor','none','LineWidth',1);
%         ylim([0 5]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[0 5],'YtickLabel',{'0' '5'});
%         set(gca,'Xtick',1:12,'Xticklabel',' ');
%         ylabel({'Coefficient of';'variation'},'FontSize',paper_font_size);
%         mlr_CoV = LinearModel.fit(1:size(bmi_performance,1),CoV);
%         if mlr_CoV.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_CoV.Coefficients.Estimate;
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_CoV.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;

%         axes(Cplot(3));hold on;
%         %h_latency = errorbar(1:size(bmi_performance,1), bmi_performance(:,12)./1000, bmi_performance(:,13)./1000, '-ok','MarkerFaceColor','k','LineWidth',1);
%         overall_latency = subject_study_data((subject_study_data (:,22) < 1000) & (subject_study_data (:,22) > -1000),22);
%         [f_lat,x_lat] = hist(overall_latency, -1000:20:1000);
%         h_latency_2 = bar(x_lat,f_lat);
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
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_latency.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;
        
%         figure;
%         
%         [f1,x1] = hist(subject_study_data(:,22),-500:10:500);
%         h_patch(1) = jbfill(x1,f1./trapz(x1,f1), zeros(1,length(-500:10:500)),'r','r',1,0.5);
        
%         axes(Cplot(4));hold on;
%         h_likert(subj_n) = errorbar(1:size(bmi_performance,1), bmi_performance(:,10), bmi_performance(:,11), likert_marker_color{subj_n},'MarkerFaceColor',boxplot_color{subj_n},'LineWidth',1);
%         ylim([0 4]); 
%         xlim([0.5 13]);
%         set(gca,'Ytick',[1 2 3],'YtickLabel',{'1' '2' '3'});
%         set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
%         ylabel({'Subject rating'; '(Likert scale)'},'FontSize',paper_font_size);
%         xlabel('Sessions','FontSize',10);
%         mlr_likert = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,10));
%         if mlr_likert.coefTest <= 0.05
%             line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
%             plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,Sig_line_color{subj_n},'LineWidth',0.5); hold on;
%             text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_likert.Coefficients.Estimate(2))},'FontSize',paper_font_size);
%         end
%         set(gca,'Xgrid','on');
%         set(gca,'Ygrid','on');
%         hold off;
             
        
        
        
     Subject_wise_performance = [Subject_wise_performance; bmi_performance];   
    end
    legendflex(h_tpr,Subject_labels,'ncol',2, 'ref',Cplot(1),'anchor',[5 5],'buffer',[0 5],'box','on','xscale',1,'padding',[2 1 1], 'title', 'Subject labels');
    %annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
    Subject_wise_mean = mean(Subject_wise_performance);
    Subject_wise_std = std(Subject_wise_performance);
    Subject_1_3_perf = Subject_wise_performance((Subject_wise_performance(:,1) == 1 | Subject_wise_performance(:,1) == 3),:);
    %figure; boxplot(Subject_wise_performance(:,4),Subject_wise_performance(:,1))
end
