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
Subject_name = {'S9007'};
Subject_velocity_threshold = [1.19];

plot_tpr_fpr = 1;

if plot_tpr_fpr == 1
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
        % 20 - Kinematic_onset_sample_num	21 - Target_is_hit   22 - Detection latency
        
        unique_session_nos = unique(subject_study_data(:,1));
        subject_intent_per_min = [];
        for ses_num = 1:length(unique_session_nos)
            session_performance = subject_study_data(subject_study_data(:,1) == unique_session_nos(ses_num),:);
                        
            ind_valid_trials = find(session_performance(:,5) == 1);  % col 5 - Valid(1) or Catch(2)
            ind_success_valid_trials = find((session_performance(:,5) == 1) & (session_performance(:,6) == 1)); % col 5 - Intent detected
            session_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

            ind_catch_trials = find(session_performance(:,5) == 2);
            ind_failed_catch_trials = find((session_performance(:,5) == 2) & (session_performance(:,6) == 1));
            session_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR
            
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
                                                            Session_detection_latency_mean Session_detection_latency_std]];   
            %        1          2                 3                  4                 5                        6                                                    7 
            % [subj_n  ses_num  ses_TPR    ses_FPR   #valid_trials      #catch_trials          mean(session_intents/min)
            %                    8                                          9                                                       10                              11
            % std(session_intent/min)    median(session_intent/min)     Session_likert_mean      Session_likert_std                
            %                   12                                                                    13
            % Session_detection_latency_mean    Session_detection_latency_std
        end
        
        figure('Position',[700 100 7*116 6*116]); 
        Cplot = tight_subplot(4,1,[0.05 0.02],[0.1 0.05],[0.1 0.05]);
        
        axes(Cplot(1)); hold on;
        h_tpr = plot(1:size(bmi_performance,1), 100*bmi_performance(:,3),'-^k','MarkerFaceColor','k','LineWidth',1);
        h_fpr = plot(1:size(bmi_performance,1), 100*bmi_performance(:,4),'-vk','MarkerFaceColor','none','LineWidth',1);
        ylim([0 130]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 50 100],'YtickLabel',{'0' '50' '100'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        title('Closed-loop BMI performance (Subject # 9007)','FontSize',paper_font_size);
        ylabel({'Intent detection'; 'accuracy (%)'},'FontSize',paper_font_size);
        mlr_tpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,3));
        if mlr_tpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_tpr.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_tpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        mlr_fpr = LinearModel.fit(1:size(bmi_performance,1),100*bmi_performance(:,4));
        if mlr_fpr.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_fpr.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_fpr.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        %legend([h_tpr h_fpr],'TPR','FPR','Location','SouthEast');
         [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([h_tpr, h_fpr],{'TPR','FPR'},'ncol',2, 'ref',Cplot(1),...
                                            'anchor',[1 1],'buffer',[0 0],'box','off','xscale',1,'padding',[2 1 1]);
%         set(object_h(3),'FaceAlpha',0.5);
%         set(object_h(4),'FaceAlpha',0.5);
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        
        axes(Cplot(3)); hold on;
        hbox_axes = boxplot(subject_intent_per_min(:,2),subject_intent_per_min(:,1),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o','colors','k'); % symbol - Outliers take same color as box
        set(hbox_axes(6,1:size(bmi_performance,1)),'Color','k');
        set(hbox_axes(7,1:size(bmi_performance,1)),'MarkerSize',4);
        set(hbox_axes,'LineWidth',1);
        ylim([0 50]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[0 25 50],'YtickLabel',{'0' '25' '50'});
        set(gca,'Xtick',1:12,'Xticklabel',' ');
        set(gca,'Box','off')
        ylabel({'Intents per'; 'min.'},'FontSize',paper_font_size);
        plot1_pos = get(Cplot(1),'Position');
        boxplot_pos = get(gca,'Position');
        set(gca,'Position',[plot1_pos(1) boxplot_pos(2) plot1_pos(3) boxplot_pos(4)]);
        mlr_intents = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,9));
        if mlr_intents.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_intents.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
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

        axes(Cplot(2));hold on;
        h_latency = errorbar(1:size(bmi_performance,1), bmi_performance(:,12)./1000, bmi_performance(:,13)./1000, '-ok','MarkerFaceColor','k','LineWidth',1);
        ylim([-1 1]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[-1 -0.5 0 0.5 1],'YtickLabel',{'-1' '-0.5' '0' '0.5' '1'});
        set(gca,'Xtick',1:12,'Xticklabel',{' '});
        ylabel({'Intent detection';'latency (sec.)'},'FontSize',paper_font_size);
        %xlabel('Sessions','FontSize',10);
        plot1_pos = get(Cplot(1),'Position');
        latencyplot_pos = get(gca,'Position');
        set(gca,'Position',[plot1_pos(1) latencyplot_pos(2) plot1_pos(3) latencyplot_pos(4)]);
        mlr_latency = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,12));
        if mlr_latency.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_latency.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_latency.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        axes(Cplot(4));hold on;
        h_cov = errorbar(1:size(bmi_performance,1), bmi_performance(:,10), bmi_performance(:,11), '-sk','MarkerFaceColor','k','LineWidth',1);
        ylim([0 4]); 
        xlim([0.5 13]);
        set(gca,'Ytick',[1 2 3],'YtickLabel',{'1' '2' '3'});
        set(gca,'Xtick',1:12,'Xticklabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12'});
        ylabel({'Subject';'rating'},'FontSize',paper_font_size);
        xlabel('Sessions','FontSize',10);
        mlr_likert = LinearModel.fit(1:size(bmi_performance,1),bmi_performance(:,10));
        if mlr_likert.coefTest <= 0.05
            line_regress = [ones(2,1) [1; size(bmi_performance,1)]]*mlr_likert.Coefficients.Estimate;
            plot(gca,[0  size(bmi_performance,1)+0.25],line_regress,'--k','LineWidth',0.5); hold on;
            text(size(bmi_performance,1)+0.25,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_likert.Coefficients.Estimate(2))},'FontSize',paper_font_size);
        end
        set(gca,'Xgrid','on');
        set(gca,'Ygrid','on');
        hold off;
        
        annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
        
        
        
        
        
    end
    
    
    
end
