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
x_axis_deviation = [0.15,0.05,-0.05,-0.15];
directory = 'D:\NRI_Project_Data\Clinical_study_Data\';
Subject_name = {'S9014'}; %{'S9014','S9012','S9011','S9010','S9009','S9007'}; %Modified 10-4-2016
Subject_number = [9014,9012,9011,9010,9009,9007];
Subject_labels = {'S6','S5','S4','S3','S2','S1'};
Subject_velocity_threshold = [1.17,1.28,1.16,1.03,1.99,1.19];
ACC_marker_color = {'--ob','-sk','-vr','-^m','--ok','-sb'};
Latency_marker_color = {'ob','sk','vr','^m','ok','sb'};
TPR_marker_color = {'--ok''--sk','--ok','--*k','--^k'};
FPR_marker_color = {'--ok''-sk','-ok','*k','^k'};
Marker_face_color = {'w','k','r','m','w','b'};
Sig_line_color = {'--k','--k','--k','--k','--k','--k'};
boxplot_color = {'r','r'};
%likert_marker_color = {'-ok','-sk','-ok','--^k','-vk'};
h_acc = zeros(length(Subject_name),1);
h_tpr = zeros(length(Subject_name),1);
h_fpr = zeros(length(Subject_name),1);
h_intent = zeros(length(Subject_name),1);
h_likert = zeros(length(Subject_name),1);

plot_tpr_fpr = 1;
plot_blockwise_accuracy = 0;
plot_movement_smoothness = 0;

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
     Cplot = tight_subplot(1,1,[0.05 0.02],[0.25 0.1],[0.1 0.05]);
     Subject_wise_performance = [];
     
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
        
        unique_session_nos = 11; %unique(subject_study_data(:,1)); %Modified 10-4-2016
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
            Test_BMI_detected = session_performance(ind_success_valid_trials,20) - session_performance(ind_success_valid_trials,4);
            session_latencies(session_latencies < -1000 | session_latencies > 1000) = [];
            Session_detection_latency_mean = mean(session_latencies);
            Session_detection_latency_std = std(session_latencies);
            Session_likert_mean = mean(session_performance(:,18));
            Session_likert_std = std(session_performance(:,18));
            
            unique_block_nos = 6; %unique(subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),2)); %Modified 10-4-2016
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
    
    legendflex(fliplr(h_tpr')',fliplr(Subject_labels),'ncol',3, 'ref',Cplot(1),'anchor',[3 1],'buffer',[0 5],'box','on','xscale',1,'padding',[2 1 1], 'title', 'Subject labels');
    %annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
    Group_mean = mean(Subject_wise_performance);
    Group_std = std(Subject_wise_performance);
    mean_Subject_wise_perf_mean = [];
    mean_Subject_wise_perf_std = [];
    max_Subject_wise_intrasession_variability = [];
     for subj_n = 1:length(Subject_name)
         mean_Subject_wise_perf_mean = [mean_Subject_wise_perf_mean;...
                                                                mean(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,4))];
         mean_Subject_wise_perf_std = [mean_Subject_wise_perf_std;...
                                                                std(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,4))];
         max_Subject_wise_intrasession_variability = [max_Subject_wise_intrasession_variability;
                                                                max(Subject_wise_performance(Subject_wise_performance(:,1) == subj_n,16))];
     end
    %figure; boxplot(Subject_wise_performance(:,4),Subject_wise_performance(:,1))
end

if plot_movement_smoothness == 1
    load('C:\NRI_BMI_Mahi_Project_files\All_Subjects\Smoothness_coefficient_S9007-9010.mat');
    kin_smoothness = s7_s9_s10_smth;
    figure('Position',[700 100 7*116 2.5*116]);
    Cplot = tight_subplot(1,1,[0.05 0.02],[0.15 0.05],[0.1 0.05]);
    axes(Cplot(1));hold on;
    for subj_n = 2: length(Subject_name)
        kin_performance = kin_smoothness(kin_smoothness(:,1) == Subject_number(subj_n),:);
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


%% Added 10-4-2016

    success_Intent_EEG_epochs_session = Intent_EEG_epochs_session(:,:,ind_success_valid_trials);
    % Further remove trials for which EEG epochs were unable to extract
    corrupt_trials = [];
    for trial_no = 1:size(success_Intent_EEG_epochs_session,3)
        if isempty(find(success_Intent_EEG_epochs_session(1,:,trial_no) == -2,1))
            corrupt_trials = [corrupt_trials, trial_no];
        end
    end
    if ~isempty(corrupt_trials)
        corrupt_response = input([num2str(length(corrupt_trials)) ' corrupted trail(s) will be deleted, Do you wish to proceed? [y/n]: '],'s');
        if strcmp(corrupt_response,'y')
            success_Intent_EEG_epochs_session(:,:,corrupt_trials) = [];
        end
    end
    
    cloop_ses_spatial_avgs = squeeze(success_Intent_EEG_epochs_session(2,find(success_Intent_EEG_epochs_session(1,:,2) == -2.5):find(success_Intent_EEG_epochs_session(1,:,2) == 1),:))';
    epoch_time = success_Intent_EEG_epochs_session(1,find(success_Intent_EEG_epochs_session(1,:,1) == -2.5):find(success_Intent_EEG_epochs_session(1,:,1) == 1),1);
    epoch_time = round(epoch_time*100)/100;
    
    % Baseline correction and compute confidence intervals
    epoch_baseline = mean(cloop_ses_spatial_avgs(:,find(epoch_time==-2.5):find(epoch_time == -2)),2); 
    base_corr_cloop_ses_spatial_avgs  = cloop_ses_spatial_avgs - repmat(epoch_baseline,1,size(cloop_ses_spatial_avgs,2));
    
    deg_freedom = size(cloop_ses_spatial_avgs,1) - 1;
    t_value = tinv(1 - 0.05/2, deg_freedom);
    cloop_spatial_avg_SE = t_value*std(base_corr_cloop_ses_spatial_avgs)/sqrt( size(cloop_ses_spatial_avgs,1));
    
    % Kinematic movement onset - EMG+EEG_GO
    time_interval = session_performance(ind_success_valid_trials,20) - session_performance(ind_success_valid_trials,4);
    time_interval(time_interval < -1000 | time_interval > 1000) = [];
    
    figure; hold on;
    plot(epoch_time,mean(base_corr_cloop_ses_spatial_avgs),'LineWidth',1,'Color',[0 0 0]);
    plot(epoch_time,mean(base_corr_cloop_ses_spatial_avgs)+ cloop_spatial_avg_SE,'--','Color',[0 0 0],'LineWidth',0.5);
    plot(epoch_time,mean(base_corr_cloop_ses_spatial_avgs)- cloop_spatial_avg_SE,'--','Color',[0 0 0],'LineWidth',0.5);
        
    axis([-1.5 1 -3 1]);
    set(gca,'Xtick',[-1.5 -1 0 1],'XtickLabel',{'-1.5' '-1' '0' '1'},'xgrid','on','FontSize',10);
    %text(-0.3,4.5,'detected');
    set(gca,'Ydir','reverse');
    set(gca,'Ytick',[-3 -1.5 0 1],'YtickLabel',{'-3','-1.5','0','1'},'ygrid','on','box','off');
    line([0 0],[-5.5 3],'Color','b','LineWidth',0.5,'LineStyle','--');
    line(-1*[mean(session_latencies) mean(session_latencies)]/1000,[-5.5 3],'Color','r','LineWidth',0.5,'LineStyle','--');
    line(-1*[mean(session_latencies)-mean(time_interval) mean(session_latencies)-mean(time_interval)]/1000,[-5.5 3],'Color','k','LineWidth',0.5,'LineStyle','--');
     xlabel('Time(s)','FontSize',11);
     ylabel('Mean Readiness Potential  \pm  95% C.I. (\muV)','FontSize',11);
     
     %% Channel locations
     Channels_used = [32    10    49    15    19    53    20    57];
     figure;
      topoplot([], EEG.chanlocs,'maplimits', [-6 6],'style','blank',...    
        'electrodes','labelpoint','chaninfo', EEG.chaninfo,'plotchans',Channels_used,'plotrad', 0.55,...
        'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off',...
        'conv','off');
    
    %% Single trial Readiness Potential detection by BMI
    corrected_spatial_chan_avg_index_sec = block_performance(block_ind_success_valid_trials,16)/20;
    time_exo_moves_sec = corrected_spatial_chan_avg_index_sec - block_performance(block_ind_success_valid_trials,22)/1000; % latency(ms) converted to sec
    time_BMI_detected_intention_sec = time_exo_moves_sec - (block_performance(block_ind_success_valid_trials,20) - block_performance(block_ind_success_valid_trials,4))/1000; 
    
    assitional_EEG_GO_times_sec = [68951 70001 74901 82601 86101 87151]/500; 
    
    start_of_trials_sec = double(marker_block(marker_block(:,2) == 2,1))/500; % now in seconds
    % Using help of kinematics, 
    % trial_duration = Target_is_hit(21) - Start_of_trial(3)
    trial_durations = (block_performance(:,21) - block_performance(:,3))/1000;
    end_of_trials_sec = start_of_trials_sec + trial_durations; 
    
    time_vector = 1/20:1/20:length(Overall_spatial_chan_avg)/20;
    figure('Position',[100 500 1000 300]); hold on;
    plot(time_vector, detrend(Overall_spatial_chan_avg), '-k','LineWidth',1);
    
    for i = 1:length(start_of_trials_sec)
        line([start_of_trials_sec(i) start_of_trials_sec(i)],[-20 20],'Color',[0.5 0.5 0.5],'LineWidth',0.5,'LineStyle',':');
        line([end_of_trials_sec(i) end_of_trials_sec(i)],[-20 20],'Color',[0.5 0.5 0.5],'LineWidth',0.5,'LineStyle',':');
    end
    
    for i = 1:length(corrected_spatial_chan_avg_index_sec)
        line([corrected_spatial_chan_avg_index_sec(i) corrected_spatial_chan_avg_index_sec(i)],[-20 20],'Color','b','LineWidth',0.5,'LineStyle','--');
        %line([time_BMI_detected_intention_sec(i) time_BMI_detected_intention_sec(i)],[-20 20],'Color','b','LineWidth',0.5,'LineStyle','-');
        line([time_exo_moves_sec(i) time_exo_moves_sec(i)],[-20 20],'Color','r','LineWidth',0.5,'LineStyle','--');
    end
    
    for i = 1:length(assitional_EEG_GO_times_sec)
        line([assitional_EEG_GO_times_sec(i) assitional_EEG_GO_times_sec(i)],[-20 20],'Color','b','LineWidth',0.5,'LineStyle','--');
    end
    set(gca,'YDIr','reverse');    
    axis([125 176.5 -5 5]);
     xlabel('Time(s)','FontSize',10);
     ylabel('Single-trial Readiness Potential (\muV)','FontSize',10);
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    