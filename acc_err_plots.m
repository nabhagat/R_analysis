% TNSRE Abstract Plots
clear
mycolors_square = {'-rs','-bs','-ks','-gs','-bs'};
mycolors_circle = {'-ro','-bo','-ko','-go','-bo'};
myfacecolors = ['r','b','k','g','b'];
posterFontSize = 10;
paper_font_size = 10;
%load('acc_per_session.mat');
%load('err_per_session.mat');
% acc_per_session = [acc_per_session(1:2,:); acc_per_session(5,:); acc_per_session(3:4,:)]
% err_per_session = [err_per_session(1:2,:); err_per_session(5,:); err_per_session(3:4,:)]

%acc_per_session = 100.*acc_per_session;

max_ses3 = 4;
max_ses4 = 8;
max_ses5 = 9;
Subject_names = {'LSGR','PLSH','ERWS','BNBO'};
Sess_nums = 4:5;
maxY_ranges = [90 30 65 45];
features_names = {'Slope','-ve Peak', 'Area', 'Mahalanobis'};

plot_intent_fpr_min = 0;
plot_num_attempts = 0;
plot_different_metrics = 0;
plot_intent_only = 1;
plot_CoV = 1;
plot_tpr_fpr_comparison_old = 0;
plot_tpr_fpr_comparison_new = 0;
plot_performance_day4_day5 = 0;
compare_closed_loop_features = 0;
plot_with_likert =  0;

%%
% figure();
% T_plot = tight_subplot(1,2,[0.15],[0.2 0.15],[0.15 0.15]);
% 
% for i = 1:5
%     axes(T_plot(1)); hold on;
%     if i == 2 % Subject ERWS
%         %plot([4 5],acc_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     elseif i == 3 % Subject JF
%         %plot([5],acc_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     else % Subjects LSGR, PLSH, BNBO
%         plot([3 4 5],acc_per_session(i,1:3),mycolors_square{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',8)
%     end
% end
% axes(T_plot(1)); hold on; grid on;
% axis([2.5 5.5 0 100])
% set(gca,'XTick',[3 4 5]);
% set(gca,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(gca,'YTick',[0 25 50 75 100]);
% set(gca,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',12);
% xlabel('Days','FontSize',14);
% ylabel('% TPR','FontSize',14);
% 
% 
% for i = 1:5
%     axes(T_plot(2)); hold on;
%     if i == 2 % Subject ERWS
%         %plot([4 5],err_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     elseif i == 3 % Subject JF
%         %plot([5],err_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     else % Subjects LSGR, PLSH, BNBO
%         plot([3 4 5],err_per_session(i,1:3),mycolors_circle{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',8)
%     end
% end
% axes(T_plot(2)); hold on; grid on;
% axis([2.5 5.5 0 3]);
% set(gca,'XTick',[3 4 5]);
% set(gca,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(gca,'YTick',[0 0.5 1 2 3]);
% set(gca,'YTickLabel',{'0' '0.5' '1' '2' '3'},'FontSize',12);
% xlabel('Days','FontSize',14);
% ylabel('Error/min','FontSize',14);
% leg1 = legend('BNBO','LSGR','PLSH','Orientation','Horizontal','Location','SouthOutside')
% % legtitle = get(leg1,'Title');
% % set(legtitle,'String','Subjects');

%% Multiple Y axes plots
%     acc_per_session([2 3],:) = [];
%     err_per_session([2 3],:) = [];
%     figure;
%     S_plot = tight_subplot(1,3,[0.1],[0.1 0.15],[0.1 0.1]);
% for i = 1:3
%     axes(S_plot(i)); hold on;
%     %if i == 2 % Subject ERWS
%         %plot([4 5],acc_per_session(2,2:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     %elseif i == 3 % Subject JF
%         %plot([5],acc_per_session(3,3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%     %else % Subjects LSGR, PLSH, BNBO
%         %plot([3 4 5],acc_per_session(i,1:3),mycolors{i},'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i),'MarkerSize',10)
%         [ax,axline1,axline2] = plotyy([3 4 5],acc_per_session(i,1:3), [3 4 5], err_per_session(i,1:3),'plot','plot');
%         set(axline1,'marker','s','color',myfacecolors(i),'LineWidth',1.5,'MarkerFaceColor',myfacecolors(i))
%         set(axline2,'marker','o','color',myfacecolors(i),'LineWidth',1.5,'LineStyle','--','MarkerFaceColor',myfacecolors(i))
%     %end
% end
% 
% title('Performance across multiple days','FontSize',12)
% xlim(ax,[2 6]);
% ylim(axline1,[0 100])
% ylim(axline2,[0 3])
% set(axline1,'XTick',[3 4 5]);
% set(axline1,'XTickLabel',{'3' '4' '5'},'FontSize',12);
% set(axline1,'YTick',[0 25 50 75 100]);
% set(axline1,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',12);
% set(axline2,'YTick',[0 0.5 1 2 3]);
% set(axline2,'YTickLabel',{'0' '0.5' '1' '2' '3'},'FontSize',12);

%% Plot TPR per block -- Intent/min
if plot_intent_fpr_min == 1
            %max_block_range = 21;
             maxY = 100;
            %%
            figure('Position',[0 100 12*116 6*116]);
            R_plot = tight_subplot(2,3,[0.025 0.1],[0.2 0.4],[0.1 0.05]);
            axes(R_plot(1));
            pos1 = get(gca,'Position');
            set(gca,'Position',[pos1(1) pos1(2)-0.1 pos1(3) pos1(4)+0.2]);
            axes(R_plot(2));
            pos1 = get(gca,'Position');
            set(gca,'Position',[pos1(1) pos1(2)-0.1 pos1(3) pos1(4)+0.2]);
            axes(R_plot(3));
            pos1 = get(gca,'Position');
            set(gca,'Position',[pos1(1) pos1(2)-0.1 pos1(3) pos1(4)+0.2]);

            axes(R_plot(4));
            pos1 = get(gca,'Position');
            set(gca,'Position',[pos1(1) pos1(2) pos1(3) pos1(4)-0.1])
            axes(R_plot(5))
            pos1 = get(gca,'Position')
            set(gca,'Position',[pos1(1) pos1(2) pos1(3) pos1(4)-0.1])
            axes(R_plot(6))
            delete(R_plot(6));
%%
            for subj_n = 1:1
                bmi_performance = [];
                patient_performance = [];
                block_likert =[];    
                
                for n = 1:length(Sess_nums)
                    Session_Intent_per_min = [];
                    ses_n = Sess_nums(n);
                    folder_path = ['F:\Nikunj_Data\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_duration_min = diff(double(marker_block(block_start_stop_index,1)))/500/60;
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);

                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
                        Intent_per_min = 60./time_to_trigger_success_valid_trials;

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        bmi_performance = [bmi_performance;...
                                            [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
                                             mean(Intent_per_min) std(Intent_per_min) block_duration_min]];
                        patient_performance = [patient_performance;...
                                               ses_n block_n mean(block_performance(:,7)) std(block_performance(:,7)) ... % col 7 - Number of move attempts
                                               mean(block_performance(:,17)) std(block_performance(:,17)) sum(block_performance(:,17)) length(block_performance(:,17)) ]; % col 17 - Likert scale 
                        block_likert = [block_likert; block_performance(:,17)];
                        Session_Intent_per_min = [Session_Intent_per_min; Intent_per_min];
                    end % ends block_n loop

                    %        1          2               3                 4                 5               6                     7                          8                             9                                      10  
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min)]]
    
                    plotids = find(bmi_performance(:,1) == ses_n);
                    switch ses_n
                        case 3 % Session 3                       
            %                 axes(R_plot(4*subj_n-3)); 
            %                 %hold on; grid on;
            %                 errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
            %                 hold on;
            %                 plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',6);
            %                 axis([-1 max_ses5 0 maxY]);
            %                 if subj_n == 5
            %                     set(gca,'XTickLabel',1:max_ses3,'FontSize',12);
            %                     %xlabel('Closed-loop Blocks','FontSize',14);
            %                     text(-1,-2.25,'*p < 0.05','FontSize',10);
            %                 else
            %                     set(gca,'XTickLabel','','FontSize',12);
            %                 end

            %                 if subj_n == 1
            %                     title('Day 3','FontSize', 12);
            %                     mtit('Closed-loop BMI Performance','FontSize',12);
            %                     leg2 = legend('Motor Intents Detected/min (Average +/- S.D)', 'False Positives/min','Location','North','Orientation','Vertical');
            %                     set(leg2,'FontSize',12,'box','off');
            %                     pos_leg2 = get(leg2,'Position');
            %                     set(leg2,'Position',[pos_leg2(1)+0.125 (pos_leg2(2) + 0.11) pos_leg2(3) pos_leg2(4)]);
            %                 end

            %                 set(gca,'YTick',[0 maxY/2 maxY]);
            %                 set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',12);
            %                 set(gca,'XTick',[1:max_ses3]);
            %                 h1 = ylabel(Subject_names{subj_n},'FontSize',12,'Rotation',0);
            %                 posy = get(h1,'Position');
            %                 set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
                        case 4 % Session 4
                            axes(R_plot(3*subj_n-2)); 
                            %hold on; grid on;
                            %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                            [ax,hline1,hline2] = plotyy(1:length(plotids), bmi_performance(plotids,9), 1:length(plotids), (60/15).*bmi_performance(plotids,4),'plot' );
                            set(hline1,'marker','s','color','b','LineWidth',1.5,'MarkerFaceColor','b','LineStyle','none','MarkerSize',10)
                            set(hline2,'marker','s','color','r','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','r','LineStyle','none','MarkerSize',10)
                            
                            axis(ax(1),[-1 max_ses5+5 0 maxY]);
                            axis(ax(2),[-1 max_ses5+5 0 5]);
                            hold(ax(1),'on'); 
                            herrbar = errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',10,'LineWidth',2);
                            
                            [herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min),mean(Session_Intent_per_min));
                            set(herrbar_ses_intents1,'FaceColor','b','EdgeColor','b','LineWidth',2,'XData',11);
                            set(herrbar_ses_intents2,'Color','k','LineWidth',2,'XData',11);                         
                            hold(ax(1),'off');
                          
                            
                            
                            %hold on;
                            %plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',10);
                            
                            
                            if subj_n == 1
            %                     set(gca,'XTickLabel',1:max_ses4,'FontSize',12);
            %                     hxlab = xlabel('Blocks of 20 trials','FontSize',12);
            %                     pos_hxlab = get(hxlab,'Position');
            %                     set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.5) pos_hxlab(3)]);
            %                     text(-5,-10.25,'*p < 0.05','FontSize',12);
                                 %mtit('Closed-loop BMI Performance','FontSize',12);
                                leg2 = legend([hline1 hline2],'Motor Intents per min (Avg +/ SD)','False Positives per min','Location','North','Orientation','Vertical');
                                set(leg2,'FontSize',posterFontSize,'box','off');
                                pos_leg2 = get(leg2,'Position');
                                set(leg2,'Position',[pos_leg2(1) (pos_leg2(2) + 0.2) pos_leg2(3) pos_leg2(4)]);
                                mtit('BMI Performance', 'fontsize',posterFontSize,'xoff',-.15,'yoff',0.025,'FontWeight','bold');
                                
                            else
                                set(gca,'XTickLabel','','FontSize',posterFontSize);
                            end

                            if subj_n == 1
                                title('Day 4 (Backdrive Mode)','FontSize', posterFontSize);
                            end

                            set(ax(1),'YTick',[0 maxY/2 maxY]);
                            set(ax(1),'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',posterFontSize-2,'YColor','b');
                            set(ax(1),'XTick',[1:max_ses4]); 
                            set(ax(1),'XTickLabel','');
                            h1 = ylabel(ax(1),{'Motor Intents per min'},'FontSize',posterFontSize,'Rotation',90);
                            %posy = get(h1,'Position');
                            %set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
                            %text(posy(1) - 3,posy(2)+8,'Subject','FontSize',posterFontSize);
                            
                            axes(ax(2));            % Get the second axes
                            hold(ax(2),'on');               % Hold on to second axes
                            [herrbar_ses_fpr1 herrbar_ses_fpr2] = barwitherr(std((60/15).*bmi_performance(plotids,4)),mean((60/15).*bmi_performance(plotids,4)));
                            set(herrbar_ses_fpr1,'FaceColor','r','EdgeColor','r','LineWidth',2,'XData',12);
                            set(herrbar_ses_fpr2,'Color','k','LineWidth',2,'XData',12);   
                            
                            set(ax(2),'YTick',[0 3 5]);
                            set(ax(2),'YTickLabel',{'0' num2str(3) num2str(5)},'FontSize',posterFontSize-2,'YColor','r');
                            set(ax(2),'XTick',[1:max_ses4]); 
                            set(ax(2),'XTickLabel','');
                            h2 = ylabel(ax(2),'False Positives per min','FontSize',posterFontSize,'Rotation',90);
                            set(ax(2),'ycolor','r')
                           hold(ax(2),'off'); 
                                                  
                            axes(R_plot(4))
                            hold on;
                            barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                            set(barh,'FaceColor',[1 1 1],'LineWidth',2);
                            [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                            set(herrbar_dur1,'FaceColor',[1 1 1],'EdgeColor','k','LineWidth',2,'XData',11.5,'BarWidth',2);
                            set(herrbar_dur2,'Color','k','LineWidth',2,'XData',11.5);
                            
                            axis([-1 max_ses5+5 0 max(bmi_performance(plotids,11))+2]);
                            set(gca,'XTick',[1:max_ses4 11.5]); 
                            set(gca,'XTickLabel',{(1:max_ses4) 'Avg +/- SD'},'FontSize',posterFontSize-2);
                            set(gca,'YTick', [0 round(median(bmi_performance(plotids,11)))])
                            set(gca,'YTickLabel',{'0' num2str(round(median(bmi_performance(plotids,11))))},'FontSize',posterFontSize-2);
                            
                            barylab = ylabel({'Block';'Duration';'(min)'},'FontSize',posterFontSize,'Rotation',0);
                            posbar = get(barylab,'Position');
                            set(barylab,'Position',[(posbar(1))-2 posbar(2)-6 posbar(3)]);
                            %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',posterFontSize);
                            %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',posterFontSize);
                            hxlab = xlabel({'Blocks of 20 trials'},'FontSize',posterFontSize);
                            pos_hxlab = get(hxlab,'Position');
                            set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) - 1) pos_hxlab(3)]);
 
                        case 5
                            axes(R_plot(3*subj_n-1)); 
                            %hold on; grid on;
                            %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                            [ax,hline1,hline2] = plotyy(1:length(plotids), bmi_performance(plotids,9), 1:length(plotids), (60/15).*bmi_performance(plotids,4),'plot' );
                            set(hline1,'marker','s','color','b','LineWidth',1.5,'MarkerFaceColor','b','LineStyle','none','MarkerSize',11)
                            set(hline2,'marker','s','color','r','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','r','LineStyle','none','MarkerSize',11)
                            
                            axis(ax(1),[-1 max_ses5+5 0 maxY]);
                            axis(ax(2),[-1 max_ses5+5 0 5]);
                            hold(ax(1),'on'); 
                            herrbar = errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',10,'LineWidth',2);
                            
                            [herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min),mean(Session_Intent_per_min));
                            set(herrbar_ses_intents1,'FaceColor','b','EdgeColor','b','LineWidth',2,'XData',11);
                            set(herrbar_ses_intents2,'Color','k','LineWidth',2,'XData',11);                         
                            hold(ax(1),'off');
                          
                            
                            
                            %hold on;
                            %plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',10);
                            
                            
                            if subj_n == 1
                        
                            else
                                set(gca,'XTickLabel','','FontSize',posterFontSize);
                            end

                            if subj_n == 1
                                title('Day 5 (Triggered Mode)','FontSize', posterFontSize);
                            end

                            set(ax(1),'YTick',[0 maxY/2 maxY]);
                            set(ax(1),'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',posterFontSize-2,'YColor','b');
                            set(ax(1),'XTick',[1:max_ses5]); 
                            set(ax(1),'XTickLabel','');
                            h1 = ylabel(ax(1),{'Motor Intents per min'},'FontSize',posterFontSize,'Rotation',90);
                            %posy = get(h1,'Position');
                            %set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
                            %text(posy(1) - 3,posy(2)+8,'Subject','FontSize',posterFontSize);
                            
                            axes(ax(2));            % Get the second axes
                            hold(ax(2),'on');               % Hold on to second axes
                            [herrbar_ses_fpr1 herrbar_ses_fpr2] = barwitherr(std((60/15).*bmi_performance(plotids,4)),mean((60/15).*bmi_performance(plotids,4)));
                            set(herrbar_ses_fpr1,'FaceColor','r','EdgeColor','r','LineWidth',2,'XData',12);
                            set(herrbar_ses_fpr2,'Color','k','LineWidth',2,'XData',12);   
                            
                            set(ax(2),'YTick',[0 3 5]);
                            set(ax(2),'YTickLabel',{'0' num2str(3) num2str(5)},'FontSize',posterFontSize-2,'YColor','r');
                            set(ax(2),'XTick',[1:max_ses5]); 
                            set(ax(2),'XTickLabel','');
                            h2 = ylabel(ax(2),'False Positives per min','FontSize',posterFontSize,'Rotation',90);
                            set(ax(2),'ycolor','r')
                           hold(ax(2),'off'); 
                                                  
                            axes(R_plot(5))
                            hold on;
                            barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                            set(barh,'FaceColor',[1 1 1],'LineWidth',2);
                            [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                            set(herrbar_dur1,'FaceColor',[1 1 1],'EdgeColor','k','LineWidth',2,'XData',11.5,'BarWidth',2);
                            set(herrbar_dur2,'Color','k','LineWidth',2,'XData',11.5);
                            
                            axis([-1 max_ses5+5 0 max(bmi_performance(plotids,11))+2]);
                            set(gca,'XTick',[1:max_ses5 11.5]); 
                            set(gca,'XTickLabel',{(1:max_ses5) 'Avg +/- SD'},'FontSize',posterFontSize-2);
                            set(gca,'YTick', [0 round(median(bmi_performance(plotids,11)))])
                            set(gca,'YTickLabel',{'0' num2str(round(median(bmi_performance(plotids,11))))},'FontSize',posterFontSize-2);
                            
%                             barylab = ylabel({'Block';'Duration';'(min)'},'FontSize',posterFontSize,'Rotation',0);
%                             posbar = get(barylab,'Position');
%                             set(barylab,'Position',[(posbar(1))-2 posbar(2)-6 posbar(3)]);
                            %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',posterFontSize);
                            %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',posterFontSize);
                            hxlab = xlabel({'Blocks of 20 trials'},'FontSize',posterFontSize);
                            pos_hxlab = get(hxlab,'Position');
                            set(hxlab,'Position',[pos_hxlab(1)-1.5 (pos_hxlab(2) - 1) pos_hxlab(3)]);
                            
                        case 6
                            axes(R_plot(3*subj_n-1)); 
                            %hold on; grid on;
                            %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                            errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',10);
                            hold on;
                            plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',10);
                            axis([-1 max_ses5+1 0 maxY]);
                            if subj_n == 1
            %                     set(gca,'XTickLabel',1:max_ses5,'FontSize',12);
                                %xlabel('Closed-loop Blocks','FontSize',14);
                                %text(1,-33,'*p < 0.05','FontSize',12);
                            else
                                set(gca,'XTickLabel','','FontSize',12);
                            end

                            if subj_n == 1
                                title('Day 5 (Triggered Mode)','FontSize', posterFontSize);
                            end

                            set(gca,'YTick',[0 maxY/2 maxY]);
                            set(gca,'YTickLabel','');
                            %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
                            set(gca,'XTick',[1:max_ses5]);
                            set(gca,'XTickLabel','');

                            axes(R_plot(5))
                            barh = bar(1:length(plotids), bmi_performance(plotids,11));
                            axis([-1 max_ses5+1 0 8]);
                            set(gca,'XTickLabel',1:max_ses5,'FontSize',posterFontSize-2);
                            set(gca,'YTick', [0 6])
                            set(gca,'YTickLabel','','FontSize',posterFontSize-2);
                            set(barh,'FaceColor',[1 1 1],'LineWidth',2);
                            set(gca,'XTickLabel',1:max_ses5,'FontSize',posterFontSize-2);
                            hxlab = xlabel('Blocks of 20 trials','FontSize',posterFontSize);
                            pos_hxlab = get(hxlab,'Position');
                            set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) - 1) pos_hxlab(3)]);

                        otherwise
                            error('Incorrect Session Number in data.');
                    end %end switch 

                %axis(axis);
                %hold on; arrow([1 test_regress(1)],[length(block_TPR)+1 test_regress(2)],'LineWidth',1);
                %plot([1 length(block_TPR)],test_set*mlr_TPR.Coefficients.Estimate,'-k','LineWidth',1.5)




            %     set(gca,'YTick',[0 50 100]);
            %     set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',12);
            %     tag_text = strcat(Subject_names(i),', Intercept = ',num2str(mlr_TPR.Coefficients.Estimate(1)),', Slope =  ',num2str(mlr_TPR.Coefficients.Estimate(2)));
            %     if mlr_TPR.coefTest <= 0.05
            %         tag_text = strcat(tag_text,'*');
            %     end
            %     text(10,15,tag_text,'FontSize',12,'FontWeight','normal','Rotation',0,'BackgroundColor',[1 1 1]);
            %     if i == 1
            %         leg1 = legend('Day 3', 'Day 4', 'Day 5','Orientation','Horizontal','Location','NorthOutside');
            %         ylabel('% True Positives','FontSize',14);
            %     end
                % % legtitle = get(leg1,'Title');
                % % set(legtitle,'String','Subjects');


                %Regression Analysis
%                 if strcmp( Subject_names{subj_n} ,'ERWS') 
%                     bmi_performance((bmi_performance (:,1) == 3),:) = [];
%                 elseif  strcmp( Subject_names{subj_n} ,'JF')
%                     bmi_performance((bmi_performance (:,1) == 3),:) = [];
%                     bmi_performance((bmi_performance (:,1) == 4),:) = [];
%                 end

                mlr_intents = LinearModel.fit(1:length(plotids),bmi_performance(plotids,9));
                if mlr_intents.coefTest <= 0.05
                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_intents.Coefficients.Estimate;
                    line_regress = line_regress + max(bmi_performance(plotids,9)');
                    axes(ax(1))
                    hold(ax(1),'on');
                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                    plot(ax(1),[-1  size(plotids,1)+1],line_regress,'--b','LineWidth',2); hold on;
                    text(3,line_regress(1),'*','FontSize',16)
                     hold off;
                    axes(ax(2))
                end

%                     mlr_fpr_min = LinearModel.fit(1:length(plotids), (60/15).*bmi_performance(plotids,4));
%                 if mlr_fpr_min.coefTest <= 0.05
%                     line_regress = [ones(2,1) [1; length(plotids)]]*mlr_fpr_min.Coefficients.Estimate;
%                     %axes(R_plot(4*subj_n-3)); 
%                     axis(axis);
%                     %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
%             %          ah=axes('position',[.2,.2,.6,.6],'visible','off'); % <- select your pos...
%                     plot([1 size(plotids,1)],line_regress,':r','LineWidth',0.2); hold on;
%                     text(7,[1 7]*mlr_fpr_min.Coefficients.Estimate + 0.75,'*','FontSize',16)
%                 end

                end % ends ses_n loop

                %%
                axes(R_plot(3*subj_n)); 
                %hold on; grid on;
                errY = 100.*[std(bmi_performance(:,5)) std(bmi_performance(:,7));
                                 std(bmi_performance(:,6)) std(bmi_performance(:,8))];
                Y = 100.*[mean(bmi_performance(:,5)) mean(bmi_performance(:,7));
                                 mean(bmi_performance(:,6)) mean(bmi_performance(:,8))];
                h = barwitherr(errY,Y);
                set(h(1),'FaceColor',[0.6 0.6 0.6]);
                set(h(2),'FaceColor',[0     0    1]);
                 %set(h(1),'LineWidth',4,'EdgeColor','g');
                %set(h(2),'FaceColor',1/255*[148 0 230]);
                %set(h(2),'LineWidth',4,'EdgeColor',1/255*[148 0 230]);
                axis([0.5 2.5 0 100]);
                set(gca,'YTick',[0 50 100]);
                set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',posterFontSize-2);
                if subj_n == 1
                   set(gca,'XTickLabel',{'% True', '% False'},'FontSize',posterFontSize);
                   text(0.8,-15.5,'Positives','FontSize',posterFontSize);
                   text(1.8,-15.5,'Positives','FontSize',posterFontSize);
                   %ylabel('Percentage','FontSize',12);
                else
                   set(gca,'XTickLabel','','FontSize',posterFontSize-2);
                end
                %pos = get(gca,'Position');
                %set(gca,'Position',[pos(1)+0.05 pos(2:4)])
                if subj_n == 1
                    tit = title('BMI Accuracy','FontSize',posterFontSize);
                    pos_tit = get(tit,'Position'); 
                    set(tit,'Position',[pos_tit(1)-0.1 pos_tit(2)+50 pos_tit(3)],'FontWeight','bold');
                    leg1 = legend(h,'without EMG validation', 'with EMG validation','Location','North','Orientation','Vertical');
                    set(leg1,'FontSize',posterFontSize-2,'box','off');
                    pos_leg1 = get(leg1,'Position');
                    set(leg1,'Position',[pos_leg1(1)-0.02 (pos_leg1(2) + 0.2) pos_leg1(3) pos_leg1(4)]);
                end


            end % ends subj_n loop
             %print -dtiff -r450 PLSH_block_accuracy_modified.tif
             %saveas(gca,'PLSH_block_accuracy_modified.fig')
end

%% Plot TPR per block -- Percentage
% % %max_block_range = 21;
% % max_ses3 = 4;
% % max_ses4 = 8;
% % max_ses5 = 9;
% % Subject_names = {'BNBO','PLSH','LSGR','ERWS','JF'};
% % Sess_nums = 3:5;
% % 
% % maxY = 100;
% % figure('Position',[0 -500 8*116 8*116]);
% % R_plot = tight_subplot(5,4,[0.025],[0.1 0.15],[0.15 0.1]);
% % 
% % for subj_n = 1:5
% %     bmi_performance = [];
% %     patient_performance = [];
% %     
% %     for n = 1:length(Sess_nums)
% %         ses_n = Sess_nums(n);
% %         folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
% %         fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
% %         if ~exist(fileid,'file')
% %             continue
% %         end
% %         cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',1,1); 
% %         unique_blocks = unique(cl_ses_data(:,1));
% %         for m = 1:length(unique_blocks)
% %             block_n = unique_blocks(m);
% %             block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
% %             
% %             ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
% %             ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
% %             block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR
% %             
% %             ind_catch_trials = find(block_performance(:,4) == 2);
% %             ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
% %             block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR
% %             
% %             time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
% %             Intent_per_min = 60./time_to_trigger_success_valid_trials;
% %             
% %             ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
% %             ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
% %             EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
% %             EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);
% %             
% %             ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,9) == 1)); % col 9 - EEG+EMG decisions
% %             ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,9) == 1));
% %             EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
% %             EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);
% %             
% %             bmi_performance = [bmi_performance;...
% %                                 [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
% %                                  mean(Intent_per_min) std(Intent_per_min)]];
% %             patient_performance = [patient_performance;...
% %                                    ses_n block_n mean(block_performance(:,7)) std(block_performance(:,7)) ... % col 7 - Number f move attempts
% %                                    mean(block_performance(:,14)) std(block_performance(:,14))]; % col 14 - Likert scale 
% %         end % ends block_n loop
% %         
% %         %    1      2        3         4        5       6         7           8             9                   10  
% %         % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min)]]
% %         
% %         plotids = find(bmi_performance(:,1) == ses_n);
% %         switch ses_n
% %             case 3 % Session 3                       
% %                 axes(R_plot(4*subj_n-3)); 
% %                 %hold on; grid on;
% %                 %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 hold on;
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,3),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',6);
% %                 axis([-1 max_ses5 0 maxY]);
% %                 if subj_n == 5
% %                     set(gca,'XTickLabel',1:max_ses3,'FontSize',12);
% %                     %xlabel('Closed-loop Blocks','FontSize',14);
% %                     %text(1,-20,'*p < 0.05','FontSize',12);
% %                 else
% %                     set(gca,'XTickLabel','','FontSize',12);
% %                 end
% %                 
% %                 if subj_n == 1
% %                     title('Day 3','FontSize', 12);
% %                     mtit('Closed-loop BMI Performance','FontSize',12);
% %                     leg2 = legend('Percentage True Positives', 'False Positives/min','Location','North','Orientation','Horizontal');
% %                     set(leg2,'FontSize',10,'box','off');
% %                     pos_leg2 = get(leg2,'Position');
% %                     set(leg2,'Position',[pos_leg2(1)+0.125 (pos_leg2(2) + 0.075) pos_leg2(3) pos_leg2(4)]);
% %                 end
% %                 
% %                 set(gca,'YTick',[0 maxY/2 maxY]);
% %                 set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',12);
% %                 set(gca,'XTick',[1:max_ses3]);
% %                 h1 = ylabel(Subject_names{subj_n},'FontSize',12,'Rotation',0);
% %                 posy = get(h1,'Position');
% %                 set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
% %             case 4
% %                 axes(R_plot(4*subj_n-2)); 
% %                 %hold on; grid on;
% %                 %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
% %                 %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 hold on;
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,3),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',6);
% %                 axis([-1 max_ses5 0 maxY]);
% %                 if subj_n == 5
% %                     set(gca,'XTickLabel',1:max_ses4,'FontSize',12);
% %                     xlabel('Blocks of 20 trials','FontSize',12);
% %                     %text(1,-33,'*p < 0.05','FontSize',12);
% %                 else
% %                     set(gca,'XTickLabel','','FontSize',12);
% %                 end
% %                 
% %                 if subj_n == 1
% %                     title('Day 4','FontSize', 12);
% %                 end
% %                 
% %                 set(gca,'YTick',[0 maxY/2 maxY]);
% %                 set(gca,'YTickLabel','');
% %                 %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
% %                 set(gca,'XTick',[1:max_ses4]);
% %             case 5
% %                 axes(R_plot(4*subj_n-1)); 
% %                 %hold on; grid on;
% %                 %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
% %                 %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 hold on;
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,3),'bs','MarkerFaceColor','b','MarkerSize',6);
% %                 plot(1:length(plotids), 100.*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',6);
% %                 axis([-1 max_ses5+1 0 maxY]);
% %                 if subj_n == 5
% %                     set(gca,'XTickLabel',1:max_ses5,'FontSize',12);
% %                     %xlabel('Closed-loop Blocks','FontSize',14);
% %                     %text(1,-33,'*p < 0.05','FontSize',12);
% %                 else
% %                     set(gca,'XTickLabel','','FontSize',12);
% %                 end
% %                 
% %                 if subj_n == 1
% %                     title('Day 5','FontSize', 12);
% %                 end
% %                 
% %                 set(gca,'YTick',[0 maxY/2 maxY]);
% %                 set(gca,'YTickLabel','');
% %                 %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
% %                 set(gca,'XTick',[1:max_ses5]);
% %             otherwise
% %                 error('Incorrect Session Number in data.');
% %         end %end switch 
% %            
% %     %axis(axis);
% %     %hold on; arrow([1 test_regress(1)],[length(block_TPR)+1 test_regress(2)],'LineWidth',1);
% %     %plot([1 length(block_TPR)],test_set*mlr_TPR.Coefficients.Estimate,'-k','LineWidth',1.5)
% %     
% %     
% %     
% %     
% % %     set(gca,'YTick',[0 50 100]);
% % %     set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',12);
% % %     tag_text = strcat(Subject_names(i),', Intercept = ',num2str(mlr_TPR.Coefficients.Estimate(1)),', Slope =  ',num2str(mlr_TPR.Coefficients.Estimate(2)));
% % %     if mlr_TPR.coefTest <= 0.05
% % %         tag_text = strcat(tag_text,'*');
% % %     end
% % %     text(10,15,tag_text,'FontSize',12,'FontWeight','normal','Rotation',0,'BackgroundColor',[1 1 1]);
% % %     if i == 1
% % %         leg1 = legend('Day 3', 'Day 4', 'Day 5','Orientation','Horizontal','Location','NorthOutside');
% % %         ylabel('% True Positives','FontSize',14);
% % %     end
% %     % % legtitle = get(leg1,'Title');
% %     % % set(legtitle,'String','Subjects');
% % 
% %     end % ends ses_n loop
% %     axes(R_plot(4*subj_n)); 
% %     %hold on; grid on;
% %     errY = 100.*[std(bmi_performance(:,5)) std(bmi_performance(:,7));
% %                      std(bmi_performance(:,6)) std(bmi_performance(:,8))];
% %     Y = 100.*[mean(bmi_performance(:,5)) mean(bmi_performance(:,7));
% %                      mean(bmi_performance(:,6)) mean(bmi_performance(:,8))];
% %     h = barwitherr(errY,Y);
% %     set(h(1),'FaceColor','g');
% %      %set(h(1),'LineWidth',4,'EdgeColor','g');
% %     set(h(2),'FaceColor',1/255*[148 0 230]);
% %     %set(h(2),'LineWidth',4,'EdgeColor',1/255*[148 0 230]);
% %     axis([0.5 2.5 0 100]);
% %     set(gca,'YTick',[0 50 100]);
% %     set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',12);
% %     if subj_n == 5
% %        set(gca,'XTickLabel',{'% TPR', '% FPR'},'FontSize',12);
% %        %ylabel('Percentage','FontSize',12);
% %     else
% %        set(gca,'XTickLabel','','FontSize',12);
% %     end
% %     pos = get(gca,'Position');
% %     set(gca,'Position',[pos(1)+0.05 pos(2:4)])
% %     if subj_n == 1
% %         tit = title('Detection Accuracy','FontSize',12);
% %         pos_tit = get(tit,'Position'); 
% %         set(tit,'Position',[pos_tit(1)-0.1 pos_tit(2)+50 pos_tit(3)]);
% %         leg1 = legend(h,'EEG Only', 'EEG+EMG','Location','North','Orientation','Horizontal');
% %         set(leg1,'FontSize',10,'box','off');
% %         pos_leg1 = get(leg1,'Position');
% %         set(leg1,'Position',[pos_leg1(1)-0.065 (pos_leg1(2) + 0.075) pos_leg1(3) pos_leg1(4)]);
% %     end
% %     %mlr_TPR = LinearModel.fit(1:length(block_TPR),block_TPR);
% %     %test_regress = [ones(2,1) [1; length(block_TPR)]]*mlr_TPR.Coefficients.Estimate;
% % end % ends subj_n loop
% % % print -dtiff -r450 All_subjects_block_accuracy_bw.tif
% % % saveas(gca,'All_subjects_block_accuracy_bw.fig')

%% Number of attempts per block
if plot_num_attempts == 1
            %max_block_range = 21;
            max_ses3 = 4;
            max_ses4 = 7;
            max_ses5 = 8;
            Subject_names = {'BNBO','PLSH','LSGR','ERWS','JF'};
            Sess_nums = 4:5;

            maxY = 4;
            figure('Position',[0 100 8*116 4*116]);
            S_plot = tight_subplot(1,2,[0.025],[0.2 0.3],[0.15 0.1]);

            for subj_n = 1:1
            %     bmi_performance = [];
            %     patient_performance = [];

                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    unique_blocks = unique(cl_ses_data(:,1));

            %%
            % %         for m = 1:length(unique_blocks)
            % %             block_n = unique_blocks(m);
            % %             block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
            % %             
            % %             ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
            % %             ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
            % %             block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR
            % %             
            % %             ind_catch_trials = find(block_performance(:,4) == 2);
            % %             ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
            % %             block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR
            % %             
            % %             time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
            % %             Intent_per_min = 60./time_to_trigger_success_valid_trials;
            % %             
            % %             ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
            % %             ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
            % %             EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
            % %             EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);
            % %             
            % %             ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,9) == 1)); % col 9 - EEG+EMG decisions
            % %             ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,9) == 1));
            % %             EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
            % %             EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);
            % %             
            % %             bmi_performance = [bmi_performance;...
            % %                                 [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
            % %                                  mean(Intent_per_min) std(Intent_per_min)]];
            % %             patient_performance = [patient_performance;...
            % %                                    ses_n block_n mean(block_performance(:,7)) std(block_performance(:,7)) ... % col 7 - Number of move attempts
            % %                                    mean(block_performance(:,14)) std(block_performance(:,14))]; % col 14 - Likert scale \                  
            % %         end % ends block_n loop

                    %    1      2        3         4        5       6         7           8             9                   10  
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min)]]
            %%        
                    plotids = find(patient_performance(:,1) == ses_n);
                    switch ses_n
                        case 3 % Session 3                       
                            axes(S_plot(3*subj_n-2)); 
                            %hold on; grid on;
                            %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                            hold on;
                            errorbar(1:length(plotids),patient_performance(plotids,3),patient_performance(plotids,4),'ks','MarkerFaceColor','k','MarkerSize',6);
                           % errorbar(1:length(plotids),patient_performance(plotids,5),patient_performance(plotids,6),'ms','MarkerFaceColor','m','MarkerSize',6);
                            axis([-1 max_ses5 0 maxY]);
                            if subj_n == 1
                                set(gca,'XTickLabel',1:max_ses3,'FontSize',posterFontSize-2);
                                %xlabel('Closed-loop Blocks','FontSize',14);
                                text(-1,-2,'*p < 0.05','FontSize',posterFontSize-2);
                            else
                                set(gca,'XTickLabel','','FontSize',posterFontSize-2);
                            end

                            if subj_n == 1
                                title('Day 3','FontSize', posterFontSize);
                                mtit('Number of Attempts per Trial (Average +/- S.D)','FontSize',posterFontSize);
            %                     leg2 = legend('Number of Attempts per trial (Average +/- S.D)', 'Likert Score (Average +/- S.D)','Location','North','Orientation','Horizontal');
            %                     set(leg2,'FontSize',posterFontSize-2,'box','off');
            %                     pos_leg2 = get(leg2,'Position');
            %                     set(leg2,'Position',[pos_leg2(1)+0.2 (pos_leg2(2) + 0.075) pos_leg2(3) pos_leg2(4)]);
                            end

                            set(gca,'YTick',[0 maxY/2 maxY]);
                            set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',posterFontSize-2);
                            set(gca,'XTick',[1:max_ses3]);
                            h1 = ylabel(Subject_names{subj_n},'FontSize',posterFontSize,'Rotation',0);
                            posy = get(h1,'Position');
                            set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
                        case 4
                            axes(S_plot(2*subj_n-1)); 
                            %hold on; grid on;
                            %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                            %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                            hold on;
                            errorbar(1:length(plotids),patient_performance(plotids,3),patient_performance(plotids,4),'ks','MarkerFaceColor','k','MarkerSize',10);
                            %errorbar(1:length(plotids),patient_performance(plotids,5),patient_performance(plotids,6),'ms','MarkerFaceColor','m','MarkerSize',6);
                            axis([-1 max_ses5 0 maxY]);
                            if subj_n == 1
                                mtit('Number of attempts per trial (Average +/- S.D)','FontSize',posterFontSize);
                                set(gca,'XTickLabel',1:max_ses4,'FontSize',posterFontSize-2);
                                hxlab = xlabel('Blocks of 20 trials','FontSize',posterFontSize);
                                pos_hxlab = get(hxlab,'Position');
                                set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.2) pos_hxlab(3)]);
                                %text(1,-15,'*p < 0.05','FontSize',posterFontSize-2);
                            else
                                set(gca,'XTickLabel','','FontSize',posterFontSize-2);
                            end

                            if subj_n == 1
                                title('Day 4 (Backdrive Mode)','FontSize', posterFontSize);
                            end

                            set(gca,'YTick',[0 maxY/2 maxY]);
                            set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)});
                            %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',posterFontSize);
                            set(gca,'XTick',[1:max_ses4]);
                            h1 = ylabel(Subject_names{subj_n},'FontSize',posterFontSize,'Rotation',0);
                            posy = get(h1,'Position');
                            set(h1,'Position',[(posy(1) - 1.5) (posy(2)-0.1) posy(3)]);
                            text(posy(1) - 2.6,posy(2)+0.5,'Subject','FontSize',posterFontSize);

                        case 5
                            axes(S_plot(2*subj_n)); 
                            %hold on; grid on;
                            %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                            %errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',6);
                            hold on;
                            errorbar(1:length(plotids),patient_performance(plotids,3),patient_performance(plotids,4),'ks','MarkerFaceColor','k','MarkerSize',10);
                            %errorbar(1:length(plotids),patient_performance(plotids,5),patient_performance(plotids,6),'ms','MarkerFaceColor','m','MarkerSize',6);
                            axis([-1 max_ses5+1 0 maxY]);
                            if subj_n == 1
                                set(gca,'XTickLabel',1:max_ses5,'FontSize',posterFontSize-2);
                                %xlabel('Closed-loop Blocks','FontSize',14);
                                %text(1,-33,'*p < 0.05','FontSize',posterFontSize-2);
                            else
                                set(gca,'XTickLabel','','FontSize',posterFontSize-2);
                            end

                            if subj_n == 1
                                title('Day 5 (Triggered Mode)','FontSize', posterFontSize);
                            end

                            set(gca,'YTick',[0 maxY/2 maxY]);
                            set(gca,'YTickLabel','');
                            %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',posterFontSize-2);
                            set(gca,'XTick',[1:max_ses5]);
                            hxlab = xlabel('Blocks of 20 trials','FontSize',posterFontSize);
                                pos_hxlab = get(hxlab,'Position');
                            set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.2) pos_hxlab(3)]);
                        otherwise
                            error('Incorrect Session Number in data.');
                    end %end switch 

                %axis(axis);
                %hold on; arrow([1 test_regress(1)],[length(block_TPR)+1 test_regress(2)],'LineWidth',1);
                %plot([1 length(block_TPR)],test_set*mlr_TPR.Coefficients.Estimate,'-k','LineWidth',1.5)




            %     set(gca,'YTick',[0 50 100]);
            %     set(gca,'YTickLabel',{'0' '50' '100'},'FontSize',posterFontSize-2);
            %     tag_text = strcat(Subject_names(i),', Intercept = ',num2str(mlr_TPR.Coefficients.Estimate(1)),', Slope =  ',num2str(mlr_TPR.Coefficients.Estimate(2)));
            %     if mlr_TPR.coefTest <= 0.05
            %         tag_text = strcat(tag_text,'*');
            %     end
            %     text(10,15,tag_text,'FontSize',posterFontSize-2,'FontWeight','normal','Rotation',0,'BackgroundColor',[1 1 1]);
            %     if i == 1
            %         leg1 = legend('Day 3', 'Day 4', 'Day 5','Orientation','Horizontal','Location','NorthOutside');
            %         ylabel('% True Positives','FontSize',14);
            %     end
                % % legtitle = get(leg1,'Title');
                % % set(legtitle,'String','Subjects');


                %Regression Analysis
                if strcmp( Subject_names{subj_n} ,'ERWS') 
                    patient_performance((patient_performance (:,1) == 3),:) = [];
                elseif  strcmp( Subject_names{subj_n} ,'JF')
                    patient_performance((patient_performance (:,1) == 3),:) = [];
                    patient_performance((patient_performance (:,1) == 4),:) = [];
                end

                mlr_attempts = LinearModel.fit(1:length(plotids),patient_performance(plotids,3));
                if mlr_attempts.coefTest <= 0.05
                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_attempts.Coefficients.Estimate;
                    %axes(R_plot(4*subj_n-3)); 
                    axis(axis);
                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                    plot([1  size(plotids,1)],line_regress,':b','LineWidth',0.2); hold on;
                     text(6,[1 7]*mlr_attempts.Coefficients.Estimate + 1.75,'*','FontSize',16)
                end



                end % ends ses_n loop
            %%    
            % %     if strcmp( Subject_names{subj_n} ,'ERWS') 
            % %         patient_performance((patient_performance (:,1) == 3),:) = [];
            % %     elseif  strcmp( Subject_names{subj_n} ,'JF')
            % %         patient_performance((patient_performance (:,1) == 3),:) = [];
            % %         patient_performance((patient_performance (:,1) == 4),:) = [];
            % %     end
            % %         
            % %     mlr_attempts = LinearModel.fit(1:size(patient_performance,1),patient_performance(:,3));
            % %     if mlr_attempts.coefTest <= 0.05
            % %         line_regress = [ones(2,1) [1; size(patient_performance,1)]]*mlr_attempts.Coefficients.Estimate;
            % %         axes(S_plot(3*subj_n-2));
            % %         axis(axis);
            % %         %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
            % %         plot([1 size(patient_performance,1)],line_regress,':k','LineWidth',0.2); hold on;
            % %          text(6,[1 7]*mlr_attempts.Coefficients.Estimate + 0.5,'*','FontSize',16)
            % %     end
            % % 
            % %         mlr_likert = LinearModel.fit(1:size(patient_performance,1),patient_performance(:,5));
            % %     if mlr_likert.coefTest <= 0.05
            % %         line_regress = [ones(2,1) [1; size(patient_performance,1)]]*mlr_likert.Coefficients.Estimate;
            % %         axes(S_plot(3*subj_n-2));
            % %         axis(axis);
            % %         %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
            % % %          ah=axes('position',[.2,.2,.6,.6],'visible','off'); % <- select your pos...
            % %         plot([1 size(patient_performance,1)],line_regress,':m','LineWidth',0.2); hold on;
            % %         text(7,[1 7]*mlr_likert.Coefficients.Estimate + 0.5,'*','FontSize',16)
            % %     end

            end % ends subj_n loop
            % print -dtiff -r450 All_subjects_block_accuracy_bw.tif
            % saveas(gca,'All_subjects_block_accuracy_bw.fig')
end

%% Plotting different metrics

if plot_different_metrics == 1
            %max_block_range = 21;
             
            %%
            figure('Position',[100 1200 5*116 6*116]);
            R_plot = tight_subplot(4,2,[0.1 0.05],[0.1 0.1],[0.1 0.05]);
            
%%
            for subj_n = 1:4
                bmi_performance = [];
                patient_performance = [];
                block_likert =[];    
                
                for n = 1:length(Sess_nums)
                    Session_Intent_per_min = [];
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_duration_min = diff(double(marker_block(block_start_stop_index,1)))/500/60;
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);

                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
                        Intent_per_min = 60./time_to_trigger_success_valid_trials;

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        bmi_performance = [bmi_performance;...
                                            [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
                                             mean(Intent_per_min) std(Intent_per_min) block_duration_min]];
                        patient_performance = [patient_performance;...
                                               ses_n block_n mean(block_performance(:,7)) std(block_performance(:,7)) ... % col 7 - Number of move attempts
                                               mean(block_performance(:,17)) std(block_performance(:,17)) sum(block_performance(:,17)) length(block_performance(:,17)) ]; % col 17 - Likert scale 
                        block_likert = [block_likert; block_performance(:,17)];
                        Session_Intent_per_min = [Session_Intent_per_min; Intent_per_min];
                    end % ends block_n loop

                    %        1          2               3                 4                 5               6                     7                          8                             9                                      10  
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min)]]

                    plotids = find(bmi_performance(:,1) == ses_n);
                    P_bci  = mean([bmi_performance(plotids,3) 1-bmi_performance(plotids,4)],2);
                    R_bci = 1 + P_bci.*log2(P_bci) + (1 - P_bci).*log2(1-P_bci);
                    metric = R_bci.*bmi_performance(plotids,9); 
                    maxY = 20;
                    switch ses_n
                        
                        case 4 % Session 4
                            axes(R_plot(2*subj_n-1)); 
                            hold on; grid on;
                            plot(1:length(plotids), metric,'rs','MarkerFaceColor','r','MarkerSize',8);                           
                            axis([0 max_ses4+1 0 maxY]);
                            set(gca,'Xtick',1:8);
                            set(gca,'XtickLabel',{'1' '2' '3' '4' '5' '6' '7' '8'});
                            set(gca,'Ytick',0:5:20);
                            set(gca,'YTickLabel',{'0' '5' '10' '15' '20'});
                            %set(gca,'Ytick',[0 0.5 1]);
                            %set(gca,'YtickLabel',{'0' '0.5' '1'});
                            ylabel(['S' num2str(subj_n)])
                            %herrbar = errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',8,'LineWidth',1);
                            %hold on;
                            %plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',10);                          
                            
                            if subj_n == 1
                                title('Day 4, Triggered');
                            end
                            
                            if subj_n == 2
                                title('Backdrive');
                            end
                            
                            if subj_n == 3
                                title('Triggered');
                            end
                            
                            if subj_n == 4
                                xlabel('Block of 20 trials');
                                title('Backdrive');
                            end
                            
                            
                                                  
                             
                        case 5
                            axes(R_plot(2*subj_n)); 
                            hold on; grid on;
                            plot(1:length(plotids), metric,'rs','MarkerFaceColor','r','MarkerSize',8);                           
                            axis([0 max_ses5+1 0 maxY]);
                            set(gca,'Xtick',1:9);
                             set(gca,'XtickLabel',{'1' '2' '3' '4' '5' '6' '7' '8' '9'});
                             set(gca,'Ytick',0:5:20);
                            %set(gca,'Ytick',[0 0.5 1]);
                            %set(gca,'YtickLabel',{'0' '20' '40' '60' '80' '100'});
                            %herrbar = errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',8,'LineWidth',1);

                            if subj_n == 1
                                title('Day 5, Triggered');
                            end
                            
                            if subj_n == 2
                                title('Triggered');
                            end
                            
                            if subj_n == 3
                                title('Triggered');
                            end
                            
                            if subj_n == 4
                                xlabel('Block of 20 trials');
                                title('Triggered');
                            end
                            
                        otherwise
                            error('Incorrect Session Number in data.');
                    end %end switch 
               

                end % ends ses_n loop             
            end % ends subj_n loop
            mtit('Bits per min','xoff',-0.0,'yoff',0.03);
             %print -dtiff -r450 PLSH_block_accuracy_modified.tif
             %saveas(gca,'PLSH_block_accuracy_modified.fig')
end

%% Plot Intent/min only
if plot_intent_only == 1
                     
            figure('Position',[100 1100 7.16*116 7*116]);     % [left bottom width height]
            height_inc = 0.05; 
            height_shift = 0.005;
            I_plot = tight_subplot(8,2,[0.05 0.02],[0.1 0.01],[0.1 0.01]);
            
            for p = 1:4:13
                axes(I_plot(p));
                pos_p = get(gca,'Position'); 
                set(gca,'Position',[pos_p(1) pos_p(2)-height_inc pos_p(3) pos_p(4)+height_inc]);
                
                axes(I_plot(p+1));
                pos_p = get(gca,'Position'); 
                set(gca,'Position',[pos_p(1) pos_p(2)-height_inc pos_p(3) pos_p(4)+height_inc]);   
                
                axes(I_plot(p+2));
                pos_p = get(gca,'Position'); 
                set(gca,'Position',[pos_p(1) pos_p(2)-height_shift pos_p(3) pos_p(4)-height_shift]);   
                
                axes(I_plot(p+3));
                pos_p = get(gca,'Position'); 
                set(gca,'Position',[pos_p(1) pos_p(2)-height_shift pos_p(3) pos_p(4)-height_shift]);   
            end
           
            for subj_n = 1:4
                bmi_performance = [];
                 
                for n = 1:length(Sess_nums)
                    Session_Intent_per_min = [];
                    ses_n = Sess_nums(n);
                    folder_path = ['F:\Nikunj_Data\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    % Print out mean and sd for the likert scale for this
                    % session and subject
                    display(sprintf('S%d, Day %d, likert score = %.2f +/- %.2f\n',subj_n,ses_n,mean(cl_ses_data(:,17)),std(cl_ses_data(:,17))));
                    if ses_n == 4
                        if subj_n == 1
                            likert_day4_all = [];
                        end
                        likert_day4_all = [likert_day4_all; cl_ses_data(:,17)];
                    else
                        if subj_n == 1
                            likert_day5_all = [];
                        end
                        likert_day5_all = [likert_day5_all; cl_ses_data(:,17)];
                    end
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_duration_min = diff(double(marker_block(block_start_stop_index,1)))/500/60;
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);

                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
                        Intent_per_min = 60./time_to_trigger_success_valid_trials;

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        bmi_performance = [bmi_performance;...
                                            [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
                                             mean(Intent_per_min) std(Intent_per_min) block_duration_min quantile(Intent_per_min,3)]];
                        
                        Session_Intent_per_min = [Session_Intent_per_min; [block_n.*ones(length(Intent_per_min),1) Intent_per_min]];
                    end % ends block_n loop

                    %        1          2               3                 4                 5               6                     7                          8                             9                                      10                                11              ...               
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min) block_duration ...  
                    %                    12                                              13                                           14
                    %       25-th percentile                median(Intent_per_min)        75-th percentile       ]

                    plotids = find(bmi_performance(:,1) == ses_n);
                    maxY = 100;
                    switch ses_n
                        case 4 % Session 4
                            axes_no = 4*subj_n-3;
                            axes(I_plot(axes_no)); 
                            hold on;

% Simple plot
%                             hold on; grid on;
%                             plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
% plotyy
%                             [ax,hline1,hline2] = plotyy(1:length(plotids), bmi_performance(plotids,9), 1:length(plotids), (60/15).*bmi_performance(plotids,4),'plot' );
%                             set(hline1,'marker','s','color','b','LineWidth',1.5,'MarkerFaceColor','b','LineStyle','none','MarkerSize',10)
%                             set(hline2,'marker','s','color','r','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','r','LineStyle','none','MarkerSize',10)
%                             axis(ax(1),[-1 max_ses5+5 0 maxY]);
%                             axis(ax(2),[-1 max_ses5+5 0 5]);
%                             hold(ax(1),'on'); 
%                             herrbar = errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',10,'LineWidth',2);                                                         
%                             hold(ax(1),'off');
% boxplots                          
                            hbox_axes = boxplot(Session_Intent_per_min(:,2),Session_Intent_per_min(:,1),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(hbox_axes(6,1:length(unique_blocks)),'Color','k');
                            set(hbox_axes(7,1:length(unique_blocks)),'MarkerSize',4);
                            h_axes = gca;
                            
                            %[herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min(:,2)),mean(Session_Intent_per_min(:,2)));
                            %set(herrbar_ses_intents1,'FaceColor',[1 1 1],'EdgeColor','b','LineWidth',1.5,'XData',11,'BarWidth',1);
                            %set(herrbar_ses_intents2,'Color','k','LineWidth',1.5,'XData',11);                         
                            h_overall = boxplot(h_axes, Session_Intent_per_min(:,2),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(h_overall(6),'Color','k');
                            set(h_overall(7),'MarkerSize',4);
                            set([hbox_axes h_overall],'LineWidth',1);
                            axis(h_axes,[0 max_ses4+4 0  maxY_ranges(subj_n)]);
                            
                            outlier_vals = unique(Session_Intent_per_min(Session_Intent_per_min(:,2) > maxY_ranges(subj_n),1));
                            if ~isempty(outlier_vals)
                                [~,vals_loc,~] = intersect(unique_blocks,outlier_vals);
                                if ((subj_n == 2) || (subj_n == 4))
                                    %plot(vals_loc,(maxY_ranges(subj_n)-2).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-2,'^k','MarkerFaceColor','k');
                                elseif subj_n == 1
                                    %plot(vals_loc,(maxY_ranges(subj_n)-6).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-6,'^k','MarkerFaceColor','k');
                                else
                                    %plot(vals_loc,(maxY_ranges(subj_n)-5).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-5,'^k','MarkerFaceColor','k');
                                end
                                
                            end
                                
                            
                            set(h_axes,'YGrid','on')
                            if subj_n == 1 || (subj_n == 3)
                                set(h_axes,'YTick',[0 25 maxY/2 maxY]);
                                set(h_axes,'YTickLabel',{'0' '25' num2str(maxY/2) num2str(maxY)},'FontSize',paper_font_size-1,'YColor','k');
                            else
                                set(h_axes,'YTick',[0 10 25 maxY/2  maxY]);
                                set(h_axes,'YTickLabel',{'0' '10' '25' num2str(maxY/2) num2str(maxY)},'FontSize',paper_font_size-1,'YColor','k');
                            end
                            
                            switch subj_n
                                case 1
                                    %title({'Day 4';'S1 (AT)'},'FontSize',paper_font_size-1);
                                    title({'Day 4'},'FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S1 (AT)','FontSize',paper_font_size-1);
                                    
                                case 2
                                    %title('S2 (\bfBD\rm)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S2 (\bfBD\rm)','FontSize',paper_font_size-1);
                                    
                                case 3
                                    %title('S3 (AT)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S3 (AT)','FontSize',paper_font_size-1);
                                case 4
                                    %title('S4 (\bfBD\rm)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S4 (\bfBD\rm)','FontSize',paper_font_size-1);
                            end
                            h1 = ylabel(h_axes,{'Intents per';'min'},'FontSize',paper_font_size-1,'Rotation',90);
                            posy = get(h1,'Position');                          
                                                                                                         
                            axes(I_plot(axes_no + 2))
                            
                            hold on;
                            if plot_CoV == 1
                                % CoV = sd/mean
                                reg_CoV = bmi_performance(plotids,10)./bmi_performance(plotids,9);
                                plot(1:length(plotids),reg_CoV,'sk','MarkerFaceColor','k');
                                %plot(1:length(plotids),reg_CoV,'-k');
                                plot(11,std(Session_Intent_per_min(:,2))/mean(Session_Intent_per_min(:,2)),'sk','MarkerFaceColor','k');
                                
                                % Quartile CoV = (q75-q25)/(q75+q25)
                                quartile_CoV = (bmi_performance(plotids,14) - bmi_performance(plotids,12))./(bmi_performance(plotids,14) + bmi_performance(plotids,12));
                                %plot(1:length(plotids),quartile_CoV,'ob');
                                %plot(1:length(plotids),quartile_CoV,'-b');
                                %quant_overall = quantile(Session_Intent_per_min(:,2),3);
                                %plot(11,(quant_overall(3)-quant_overall(1))/(quant_overall(3)+quant_overall(1)),'sb');
                                
                                
                                if (subj_n == 1)
                                    ses4_max_reg_Cov = 4.5;%max(reg_CoV);
                                elseif subj_n == 2
                                    ses4_max_reg_Cov = 3;%max(reg_CoV);
                                elseif subj_n == 3
                                    ses4_max_reg_Cov = 5.5;%max(reg_CoV);
                                elseif subj_n == 4
                                    ses4_max_reg_Cov = 1.5;%max(reg_CoV);
                                end
                                axis([0 max_ses4+4 0 ses4_max_reg_Cov+1]);
                                set(gca,'XTick',[1:length(unique_blocks) 11]); 
                                set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',paper_font_size-1);
                                set(gca,'YTick', [0 ses4_max_reg_Cov-0.5])
                                set(gca,'YTickLabel',{'0' num2str(ses4_max_reg_Cov-0.5)},'FontSize',paper_font_size-1);
                                set(gca,'XGrid','on','YGrid', 'on','Box','on')
                                
                                barylab = ylabel({'CoV'},'FontSize',paper_font_size-1,'Rotation',90);
                                posbar = get(barylab,'Position');
                                set(barylab,'Position',[posy(1) posbar(2) posbar(3)]);
                                
                                
                                mlr_ses4_CoV = LinearModel.fit(1:length(plotids),reg_CoV);
                                if mlr_ses4_CoV.coefTest <= 0.05
                                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_ses4_CoV.Coefficients.Estimate;
                                    %axes(h_axes)
                                    %hold(ax(1),'on');
                                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                                    %set(gca,'XTickLabel',{' '});
                                    plot(I_plot(axes_no + 2),[-1  size(plotids,1)+1],line_regress,'--k','LineWidth',0.5); hold on;
                                    text(size(plotids,1)+1,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_ses4_CoV.Coefficients.Estimate(2))},'FontSize',paper_font_size-1);
                                    %hold off;
                                    %axes(ax(2))
                                end
                
                            else
                                barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                                set(barh,'FaceColor',[1 1 1],'LineWidth',1,'BarWidth',0.5);
                                %dur_overall = boxplot(gca, bmi_performance(plotids,11),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','+','boxstyle','outline','colors',[0.6 0.6 0.6]); % symbol - Outliers take same color as box
                                [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                                set(herrbar_dur1,'EdgeColor','k','LineWidth',1,'XData',11,'BarWidth',0.5,'FaceColor',[1 1 1]);
                                set(herrbar_dur2,'Color','k','LineWidth',1,'XData',11);

                                axis([0 max_ses4+4 0 max(bmi_performance(plotids,11))+2]);
                                set(gca,'XTick',[1:length(unique_blocks) 11]); 
                                set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',paper_font_size-1);
                                ses4_median_dur = 5; % round(median(bmi_performance(plotids,11)));
                                set(gca,'YTick', [0 ses4_median_dur])
                                set(gca,'YTickLabel',{'0' num2str(ses4_median_dur)},'FontSize',paper_font_size-1);
                                set(gca,'XGrid','on','YGrid', 'on')

                                %if subj_n == 1
                                    barylab = ylabel({'Length';'(min)'},'FontSize',paper_font_size-1,'Rotation',90);
                                    posbar = get(barylab,'Position');
                                    set(barylab,'Position',[posy(1) posbar(2) posbar(3)]);
                                %end
                                %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',paper_font_size-1);
                                %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',paper_font_size-1);
                            end

                                if subj_n == 4
                                    hxlab = xlabel({'Blocks of 20 trials'},'FontSize',paper_font_size-1);
                                    pos_hxlab = get(hxlab,'Position');
                                    set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) ) pos_hxlab(3)]);
                                end

                                dur_pos = get(gca,'Position');
                                intent_pos = get(h_axes,'Position');
                                set(h_axes,'Position',[dur_pos(1) intent_pos(2) dur_pos(3) intent_pos(4)]);
                         
                        case 5
                           
                            axes_no = 4*subj_n-2;
                            axes(I_plot(axes_no)); 
                            hold on;

                            % boxplots                          
                            hbox_axes = boxplot(Session_Intent_per_min(:,2),Session_Intent_per_min(:,1),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(hbox_axes(6,1:length(unique_blocks)),'Color','k');
                            set(hbox_axes(7,1:length(unique_blocks)),'MarkerSize',4);
                            h_axes = gca;
                            h_overall = boxplot(h_axes, Session_Intent_per_min(:,2),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(h_overall(6),'Color','k');
                            set(h_overall(7),'MarkerSize',4);
                            set([hbox_axes h_overall],'LineWidth',1);
                            
                            axis(h_axes,[0 max_ses5+3 0  maxY_ranges(subj_n)]);
                            % Add triangles when the outliers is outside the
                            % axes and not shown in figure
                            outlier_vals = unique(Session_Intent_per_min(Session_Intent_per_min(:,2) > maxY_ranges(subj_n),1));
                            if ~isempty(outlier_vals)
                                [~,vals_loc,~] = intersect(unique_blocks,outlier_vals);
                                if ((subj_n == 2) || (subj_n == 4))
                                    %plot(vals_loc,(maxY_ranges(subj_n)-2).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-2,'^k','MarkerFaceColor','k');
                                elseif subj_n == 1
                                    %plot(vals_loc,(maxY_ranges(subj_n)-6).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-6,'^k','MarkerFaceColor','k');
                                else
                                    %plot(vals_loc,(maxY_ranges(subj_n)-5).*ones(length(vals_loc)),'^k','MarkerFaceColor','k');
                                    %plot(11,maxY_ranges(subj_n)-5,'^k','MarkerFaceColor','k');
                                    
                                end
                                
                            end
                            
                            %[herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min(:,2)),mean(Session_Intent_per_min(:,2)));
                            %set(herrbar_ses_intents1,'FaceColor',[1 1 1],'EdgeColor','b','LineWidth',1.5,'XData',11,'BarWidth',1);
                            %set(herrbar_ses_intents2,'Color','k','LineWidth',1.5,'XData',11);                         
                            
                            set(h_axes,'YGrid','on')
                            if (subj_n == 1) || (subj_n == 3)
                                set(h_axes,'YTick',[0 25 maxY/2 maxY]);
                                set(h_axes,'YTickLabel',{' '});
                            else
                                set(h_axes,'YTick',[0 10 25 maxY/2 maxY]);
                                set(h_axes,'YTickLabel',{' '});
                            end
                            
                            switch subj_n
                                case 1
                                    title({'Day 5'},'FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S1 (AT)','FontSize',paper_font_size-1);
                                    %h1 = ylabel(h_axes,{'Intents per';'min'},'FontSize',paper_font_size-1,'Rotation',90);
                                    %posy = get(h1,'Position');
                                case 2
                                    %title('S2 (AT)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S2 (AT)','FontSize',paper_font_size-1);
                                case 3
                                    %title('S3 (AT)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S3 (AT)','FontSize',paper_font_size-1);
                                case 4
                                    %title('S4 (AT)','FontSize',paper_font_size-1);
                                    text(0,maxY_ranges(subj_n)-5,'S4 (AT)','FontSize',paper_font_size-1);
                            end
                            
                            %h1 = ylabel(h_axes,{'Intents per min'},'FontSize',paper_font_size-1,'Rotation',90);
                            %posy = get(h1,'Position');
                            %set(h1,'Position',[(posy(1) - 0.05) posy(2:3)]);
                            %text(posy(1) - 3,posy(2)+8,'Subject','FontSize',paper_font_size-1);
                                                                             
                            axes(I_plot(axes_no + 2))
                            
                            hold on;
                            if plot_CoV == 1
                                % CoV = sd/mean
                                reg_CoV = bmi_performance(plotids,10)./bmi_performance(plotids,9);
                                plot(1:length(plotids),reg_CoV,'sk','MarkerFaceColor','k');
                                %plot(1:length(plotids),reg_CoV,'-k');
                                plot(11,std(Session_Intent_per_min(:,2))/mean(Session_Intent_per_min(:,2)),'sk','MarkerFaceColor','k');
                                
                                % Quartile CoV = (q75-q25)/(q75+q25)
                                quartile_CoV = (bmi_performance(plotids,14) - bmi_performance(plotids,12))./(bmi_performance(plotids,14) + bmi_performance(plotids,12));
                                %plot(1:length(plotids),quartile_CoV,'ob');
                                %plot(1:length(plotids),quartile_CoV,'-b');
                                %quant_overall = quantile(Session_Intent_per_min(:,2),3);
                                %overall_quartile_CoV = (quant_overall(3)-quant_overall(1))/(quant_overall(3)+quant_overall(1));
                                %plot(11,overall_quartile_CoV,'sb');
                                
                                if (subj_n == 1)
                                    ses5_max_reg_Cov = 4.5;%max(reg_CoV);
                                elseif subj_n == 2
                                    ses5_max_reg_Cov = 3;%max(reg_CoV);
                                elseif subj_n == 3
                                    ses5_max_reg_Cov = 5.5;%max(reg_CoV);
                                elseif subj_n == 4
                                    ses5_max_reg_Cov = 1.5;%max(reg_CoV);
                                end
                                axis([0 max_ses5+3 0 ses5_max_reg_Cov+1]);
                                set(gca,'XTick',[1:length(unique_blocks) 11]); 
                                set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',paper_font_size-1);
                                set(gca,'YTick', [0 ses5_max_reg_Cov-0.5]);
                                set(gca,'YTickLabel',{' '},'FontSize',paper_font_size-1);
                                set(gca,'XGrid','on','Ygrid','on','Box','on');
                                
                                %barylab = ylabel({'CoV';' '},'FontSize',paper_font_size-1,'Rotation',90);
                                %posbar = get(barylab,'Position');
                                %set(barylab,'Position',[posy(1) posbar(2) posbar(3)]);
                                
                                mlr_ses5_CoV = LinearModel.fit(1:length(plotids),reg_CoV);
                                if mlr_ses5_CoV.coefTest <= 0.05
                                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_ses5_CoV.Coefficients.Estimate;
                                    %axes(h_axes)
                                    %hold(ax(1),'on');
                                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                                    %set(gca,'XTickLabel',{' '});
                                    plot(I_plot(axes_no + 2),[-1  size(plotids,1)+1],line_regress,'--k','LineWidth',0.5); hold on;
                                    text(size(plotids,1)+1,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_ses5_CoV.Coefficients.Estimate(2))},'FontSize',paper_font_size-1);
                                    %hold off;
                                    %axes(ax(2))
                                end
                                
                            else
                                barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                                set(barh,'FaceColor',[1 1 1],'LineWidth',1,'BarWidth',0.5);
                                [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                                set(herrbar_dur1,'EdgeColor','k','LineWidth',1,'XData',11,'BarWidth',0.5,'FaceColor',[1 1 1]);
                                set(herrbar_dur2,'Color','k','LineWidth',1,'XData',11);

                                axis([0 max_ses5+3 0 max(bmi_performance(plotids,11))+2]);
                                set(gca,'XTick',[1:length(unique_blocks) 11]); 
                                set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',paper_font_size-1);
                                set(gca,'YTick', [0 ses4_median_dur]);
                                set(gca,'YTickLabel',{' '},'FontSize',paper_font_size-1);
                                set(gca,'XGrid','on','Ygrid','on');

                                %barylab = ylabel({'Duration';'(min)'},'FontSize',paper_font_size-1,'Rotation',90);
                                %posbar = get(barylab,'Position');
                                %set(barylab,'Position',[(posbar(1))-1 posbar(2)-4 posbar(3)]);
                                %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',paper_font_size-1);
                                %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',paper_font_size-1);
                                                                
                            end
                                if subj_n == 4
                                    hxlab = xlabel({'Blocks of 20 trials'},'FontSize',paper_font_size-1);
                                    pos_hxlab = get(hxlab,'Position');
                                    set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) ) pos_hxlab(3)]);
                                end
                           
                            
                        otherwise
                            error('Incorrect Session Number in data.');
                    end %end switch 

                %Regression Analysis
%                 if strcmp( Subject_names{subj_n} ,'ERWS') 
%                     bmi_performance((bmi_performance (:,1) == 3),:) = [];
%                 elseif  strcmp( Subject_names{subj_n} ,'JF')
%                     bmi_performance((bmi_performance (:,1) == 3),:) = [];
%                     bmi_performance((bmi_performance (:,1) == 4),:) = [];
%                 end

                mlr_intents = LinearModel.fit(1:length(plotids),bmi_performance(plotids,13));
                if mlr_intents.coefTest <= 0.05
                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_intents.Coefficients.Estimate;
                    %line_regress = line_regress - min(bmi_performance(plotids,12)');
                    axes(h_axes)
                    %hold(ax(1),'on');
                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                    set(gca,'XTickLabel',{' '});
                    plot(h_axes,[-1  size(plotids,1)+1],line_regress,'--k','LineWidth',0.5); hold on;
                    %text(size(plotids,1)+1,line_regress(2)+2,sprintf('Slope \n %.2f',mlr_intents.Coefficients.Estimate(2)),'FontSize',10)
                    text(size(plotids,1)+1,line_regress(2)+0.5,{sprintf(' %.2f*',mlr_intents.Coefficients.Estimate(2))},'FontSize',paper_font_size-1);
                    %hold off;
                    %axes(ax(2))
                end
                end % ends ses_n loop

                
            end % ends subj_n loop
             %print -dtiff -r450 PLSH_block_accuracy_modified.tif
             %saveas(gca,'PLSH_block_accuracy_modified.fig')

             % Expand axes to fill figure
            %fig = gcf;
            %style = hgexport('factorystyle');
            %style.Bounds = 'tight';
            %hgexport(fig,'-clipboard',style,'applystyle', true);
            %drawnow;
            
            annotation('textbox',[0 0 0.1 0.07],'String','*\itp\rm < 0.05','EdgeColor','none');
            response = input('Save the figure yourself Dude!! [y/n]: ','s');
%             if strcmp(response,'y')
%                  tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_intents_per_min.tif'];
%                  fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_intents_per_min.tif'];
%                 print('-dtiff', '-r300', tiff_filename); 
%                 saveas(gcf,fig_filename);
%             else
%                 disp('Save figure aborted');
%             end

display(sprintf('Overall likert score, Day 4 = %.3f +/- %.3f, Day 5 = %.3f +/- %.3f\n',...
    mean(likert_day4_all),std(likert_day4_all),mean(likert_day5_all),std(likert_day5_all)));
end

%% Plot TPR FPR comparison  - OLD format                   
if plot_tpr_fpr_comparison_old == 1
                     
            figure('Position',[100 1100 4*116 8*116]);     % [left bottom width height]
            T_plot = tight_subplot(4,2,[0.05 0.01],[0.1 0.05],[0.1 0.3]);
                      
            for subj_n = 1:4
                bmi_performance = [];
                 
                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
                        block_duration_min = diff(double(marker_block(block_start_stop_index,1)))/500/60;
                        
                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        time_to_trigger_success_valid_trials = block_performance(ind_success_valid_trials,6); %col 6 - Time to Trigger
                        Intent_per_min = 60./time_to_trigger_success_valid_trials;

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        bmi_performance = [bmi_performance;...
                                            [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR...
                                             mean(Intent_per_min) std(Intent_per_min) block_duration_min median(Intent_per_min)]];
                        
                        
                    end % ends block_n loop

                    %        1          2               3                 4                 5               6                     7                          8                             9                                      10                                11                          12   
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min) block_duration median(Intent_per_min)]

                    plotids = find(bmi_performance(:,1) == ses_n);
                    
                    switch ses_n
                        case 4 % Session 4
                            axes_no = 2*subj_n-1;
                            axes(T_plot(axes_no)); 
                            hold on;
                            hbox_axes = boxplot(100.*bmi_performance(:,[5 7 6 8]),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                     'positions',[0.5 1.25 3 3.75]); % symbol - Outliers take same color as box
                            set(hbox_axes(6,1:4),'Color','k');
                            set(hbox_axes(5,[1 3]),'LineStyle','-');
                            set(hbox_axes([1 2],1:4),'LineStyle','-');
                            set(hbox_axes(7,[1 3]),'MarkerFaceColor',[0.6 0.6 0.6])
                            set(hbox_axes(7,1:4),'MarkerSize',4);
                            h_colors = findobj(gca,'Tag','Box');
                            box_colors = [1 1 1;0.6 0.6 0.6; 1 1 1; 0.6 0.6 0.6];
                            for j = 1:length(h_colors)
                                patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),box_colors(j,:),'FaceAlpha',0.5);
                            end
                            h_axes = gca;                            
                            set(hbox_axes,'LineWidth',1);
                            axis(h_axes,[0 4.5 -10  125]);
                            set(h_axes,'YGrid','on')
                            set(h_axes,'YTick',[0 50 100]);
                            set(h_axes,'YTickLabel',{'0' '50' '100'},'FontSize',paper_font_size-1,'YColor','k');
                            set(h_axes,'Xtick',[0.875 3.275]);
                            set(h_axes,'XtickLabel',{' '})
                            
                            set(h_axes,'Box','on')
                            
                            switch subj_n
                                case 1
                                    title('Day 4, S1 (AT)','FontSize',posterFontSize);                                   
                                case 2
                                    title('S2 (\bfBD\rm)','FontSize',posterFontSize);
                                case 3
                                    title('S3 (AT)','FontSize',posterFontSize);
                                    ax_pos = get(gca,'Position');
                                    text(ax_pos(1)-1.2, 40,'Closed-loop Performance','FontSize',posterFontSize,'Rotation',90);
                                case 4
                                    title('S4 (\bfBD\rm)','FontSize',posterFontSize);
                                    set(h_axes,'XtickLabel',{'TPR (%)' 'FPR (%)'})
                                    
                            end                                                                                                        
                            
                            case 5
                            axes_no = 2*subj_n;
                            axes(T_plot(axes_no)); 
                            hold on;

                            hbox_axes = boxplot(100.*bmi_performance(:,[5 7 6 8]),'plotstyle','traditional','widths',0.5,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                     'positions',[0.5 1.25 3 3.75]); % symbol - Outliers take same color as box
                            set(hbox_axes(6,1:4),'Color','k');
                            set(hbox_axes(5,[1 3]),'LineStyle','-');
                            set(hbox_axes([1 2],1:4),'LineStyle','-');
                            set(hbox_axes(7,[1 3]),'MarkerFaceColor',[0.6 0.6 0.6]);
                            set(hbox_axes(7,1:4),'MarkerSize',4);
                            h_colors = findobj(gca,'Tag','Box');
                            box_colors = [1 1 1;0.6 0.6 0.6; 1 1 1; 0.6 0.6 0.6];
                            for j = 1:length(h_colors)
                                patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),box_colors(j,:),'FaceAlpha',0.5);
                            end                           
                            h_axes = gca;                            
                            set(hbox_axes,'LineWidth',1);
                            axis(h_axes,[0 4.5 -10  125]);
                            set(h_axes,'YGrid','on')
                            set(h_axes,'YTick',[0 50 100]);
                            set(h_axes,'YTickLabel',{' '},'FontSize',posterFontSize,'YColor','k');
                            set(h_axes,'Xtick',[0.875 3.275]);
                            set(h_axes,'XtickLabel',{' '})
                            
                            switch subj_n
                                case 1
                                    title('Day 5, S1 (AT)','FontSize',posterFontSize);
                                    %h1 = ylabel(h_axes,{'Intents per';'min'},'FontSize',posterFontSize,'Rotation',90);
                                    %posy = get(h1,'Position');
                                    legend('EEG + EMG','EEG only','location','NorthEastOutside');
                                case 2
                                    title('S2 (AT)','FontSize',posterFontSize);
                                case 3
                                    title('S3 (AT)','FontSize',posterFontSize);
                                case 4
                                    title('S4 (AT)','FontSize',posterFontSize);
                                    set(h_axes,'XtickLabel',{'TPR (%)' 'FPR (%)'})
                            end
                            
                        otherwise
                            error('Incorrect Session Number in data.');
                    end %end switch 
                end % ends ses_n loop               
            end % ends subj_n loop
             %print -dtiff -r450 PLSH_block_accuracy_modified.tif
             %saveas(gca,'PLSH_block_accuracy_modified.fig')

             % Expand axes to fill figure
%             fig = gcf;
%             style = hgexport('factorystyle');
%             style.Bounds = 'tight';
%             hgexport(fig,'-clipboard',style,'applystyle', true);
%             drawnow;
            
            response = input('Save figure to folder [y/n]: ','s');
            if strcmp(response,'y')
                 tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_tpr_fpr.tif'];
                 fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_tpr_fpr.tif'];
                print('-dtiff', '-r300', tiff_filename); 
                saveas(gcf,fig_filename);
            else
                disp('Save figure aborted');
            end

end
                    
%% Plot TPR FPR comparison  - NEW format                   
if plot_tpr_fpr_comparison_new == 1
              
            tpr_fpr_performance = [];
            for subj_n = 1:4                 
                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
                                               
                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        tpr_fpr_performance = [tpr_fpr_performance;...
                                            [subj_n ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR]];                       
                    end % ends block_n loop
                end % ends ses_n loop               
            end % ends subj_n loop

                    %        1          2                 3                 4                 5               6                     7                          8                             9
                    % [subj_n  ses_n       block_n block_TPR block_FPR EEG_TPR      EEG_FPR        EEG_EMG_TPR     EEG_EMG_FPR
                                         
                    figure('Position',[100 1100 5*116 4.5*116]);     % [left bottom width height]
                    T_plot = tight_subplot(2,2,[0.01 0.1],[0.1 0.1],[0.1 0.01]);
                    
                    for ses = 4:5
                        Ses_tpr_fpr =  tpr_fpr_performance(find(tpr_fpr_performance(:,2) == ses),:);
                        axes(T_plot(ses-3)); 
                        hold on;
                        hbox1_axes = boxplot(100.*Ses_tpr_fpr(:,6), Ses_tpr_fpr(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[1.1 2.1 3.1 4.1]); % symbol - Outliers take same color as box
                        set(hbox1_axes,'LineWidth',1);
                        set(gca,'XtickLabel',{' '});
                        set(hbox1_axes(7,1:4),'MarkerFaceColor',[0.6 0.6 0.6])
                        set(hbox1_axes(7,1:4),'MarkerSize',4);
                        set(hbox1_axes(6,1:4),'Color','k');
                        set(hbox1_axes(5,1:4),'Color','k');
                        set(hbox1_axes([1 2],1:4),'LineStyle','-');
                        
                        h_colors = findobj(gca,'Tag','Box');
                        for j = 1:length(h_colors)
                            patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),[0.6 0.6 0.6]);
                        end
                        hbox2_axes = boxplot(100.*Ses_tpr_fpr(:,8), Ses_tpr_fpr(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[0.85 1.85 2.85 3.85]); % symbol - Outliers take same color as box
                        set(hbox2_axes,'LineWidth',1);                   
                        set(hbox2_axes(7,1:4),'MarkerSize',4);
                        set(gca,'XtickLabel',{' '});
                        set(hbox2_axes(6,1:4),'Color','k');
                        set(hbox2_axes(5,1:4),'Color','k');
                        set(hbox2_axes([1 2],1:4),'LineStyle','-');
                        
                        h_axes = gca;                            
                        axis(h_axes,[0.5 4.5 -10  115]);
                        set(h_axes,'YGrid','on')
                        set(h_axes,'YTick',[0 25 50 75 100]);
                        set(h_axes,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',posterFontSize,'YColor','k');
                        set(h_axes,'Xtick',[(0.85 + 1.1)/2 (1.85 + 2.1)/2 (2.85 + 3.1)/2 (3.85 + 4.1)/2]);
                        set(h_axes,'XtickLabel',{' '});
                        title(h_axes,['Day ' num2str(ses)]);
                        ylabel(h_axes,'TPR (%)');
                        set(h_axes,'Box','on')
                        
                        % Significance tests
                    	p_values = [];
                        for subj_n = 1:4
                            % both-tailed Wilcoxon Rank sum Test, i.e. median(EEG + EMG) >< median(EEG only)
                            [pwilcoxon,h,stats] = ranksum(Ses_tpr_fpr((Ses_tpr_fpr(:,1) == subj_n),8),Ses_tpr_fpr((Ses_tpr_fpr(:,1) == subj_n),6),'alpha',0.05,'tail','both');
                            if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                                p_values = [p_values 0.05];
                            elseif (pwilcoxon <= 0.01)
                                p_values = [p_values 0.01];
                            else
                                p_values = [p_values NaN];
                            end
                        end
                        sigstar({[0.85 1.1],[1.85 2.1], [2.85 3.1], [3.85 4.1]},p_values);
                            
                        % ----------------------------------- Plot FPR
                        axes(T_plot(ses-1)); 
                        hold on;
                        hbox1_axes = boxplot(100.*Ses_tpr_fpr(:,7), Ses_tpr_fpr(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[1.1 2.1 3.1 4.1]); % symbol - Outliers take same color as box
                        set(hbox1_axes,'LineWidth',1);                   
                        set(gca,'XtickLabel',{' '});
                        set(hbox1_axes(7,1:4),'MarkerFaceColor',[0.6 0.6 0.6])
                        set(hbox1_axes(7,1:4),'MarkerSize',4);
                        set(hbox1_axes(6,1:4),'Color','k');
                        set(hbox1_axes(5,1:4),'Color','k');
                        set(hbox1_axes([1 2],1:4),'LineStyle','-');
                        
                        h_colors = findobj(gca,'Tag','Box');
                        for j = 1:length(h_colors)
                            h_patch(j) = patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),[0.6 0.6 0.6]);
                        end
                        hbox2_axes = boxplot(100.*Ses_tpr_fpr(:,9), Ses_tpr_fpr(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[0.85 1.85 2.85 3.85]); % symbol - Outliers take same color as box
                        set(hbox2_axes,'LineWidth',1);                   
                        set(hbox2_axes(7,1:4),'MarkerSize',4);
                        set(gca,'XtickLabel',{' '});
                        set(hbox2_axes(6,1:4),'Color','k');
                        set(hbox2_axes(5,1:4),'Color','k');
                        set(hbox2_axes([1 2],1:4),'LineStyle','-');
                        
                        h_axes = gca;                            
                        axis(h_axes,[0.5 4.5 -10  115]);
                        set(h_axes,'YGrid','on')
                        set(h_axes,'YTick',[0 25 50 75 100]);
                        set(h_axes,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',posterFontSize,'YColor','k');
                        set(h_axes,'Xtick',[(0.85 + 1.1)/2 (1.85 + 2.1)/2 (2.85 + 3.1)/2 (3.85 + 4.1)/2]);
                        if ses == 4
                            set(h_axes,'XtickLabel',{'S1', 'S2 (BD)', 'S3', 'S4 (BD)'});
                        else
                            set(h_axes,'XtickLabel',{'S1', 'S2', 'S3', 'S4'});
                        end
                        xlabel(h_axes,'Subjects');
                        ylabel(h_axes,'FPR (%)');
                        set(h_axes,'Box','on');
                        
                        h_colors_w = findobj(gca,'Tag','Box');
                        for j = 1:1 %length(h_colors_w)
                            h_patch_w = patch(get(h_colors_w(j),'XData'), get(h_colors_w(j),'YData'),[1 1 1]);
                        end
                        
                        % Significance tests
                    	p_values = [];
                        for subj_n = 1:4
                            % both-tailed Wilcoxon Rank sum Test, i.e. median(EEG + EMG) > < median(EEG only)
                            [pwilcoxon,h,stats] = ranksum(Ses_tpr_fpr((Ses_tpr_fpr(:,1) == subj_n),9),Ses_tpr_fpr((Ses_tpr_fpr(:,1) == subj_n),7),'alpha',0.05,'tail','both');
                            if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                                p_values = [p_values 0.05];
                            elseif (pwilcoxon <= 0.01)
                                p_values = [p_values 0.01];
                            else
                                p_values = [p_values NaN];
                            end
                        end
                        sigstar({[0.85 1.1],[1.85 2.1], [2.85 3.1], [3.85 4.1]},p_values);                        
                    end
             
                         axes_pos = get(gca,'Position');    % [left bottom width height]
                         box_leg = legend([h_patch_w h_patch(4)],'EEG+EMG control', 'EEG control only', 'location','NorthOutside','Orientation','horizontal');
                         box_leg_pos = get(box_leg,'position');                        
                         set(box_leg,'position',[0.225, 0.95, box_leg_pos(3:4)]);
             
             %annotation('textbox',[0,0,0.1,0.1],'String','*p < 0.05; **p < 0.01','LineWidth',0);             
                
             %print -dtiff -r450 PLSH_block_accuracy_modified.tif
             %saveas(gca,'PLSH_block_accuracy_modified.fig')

             % Expand axes to fill figure
%             fig = gcf;
%             style = hgexport('factorystyle');
%             style.Bounds = 'tight';
%             hgexport(fig,'-clipboard',style,'applystyle', true);
%             drawnow;
            
            response = input('Save figure to folder [y/n]: ','s');
            if strcmp(response,'y')
                 tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_tpr_fpr_new.tif'];
                 fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_tpr_fpr_new.fig'];
                print('-dtiff', '-r300', tiff_filename); 
                saveas(gcf,fig_filename);
            else
                disp('Save figure aborted');
            end

end
                    
%% Plot performance for day 4 and day 5

if plot_performance_day4_day5 == 1
              
            tpr_fpr_performance = [];
            for subj_n = 1:4                 
                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
                    
                    unique_blocks = unique(cl_ses_data(:,1));
                    for m = 1:length(unique_blocks)
                        block_n = unique_blocks(m);
                        load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
                        block_start_stop_index = find(marker_block(:,2) == 50);
                        block_performance = cl_ses_data(cl_ses_data(:,1) == block_n,:);
                                               
                        ind_valid_trials = find(block_performance(:,4) == 1);  % col 4 - Valid(1) or Catch(2)
                        ind_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); % col 5 - Intent detected
                        block_TPR = length(ind_success_valid_trials)/length(ind_valid_trials);      % TPR

                        ind_catch_trials = find(block_performance(:,4) == 2);
                        ind_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        block_FPR = length(ind_failed_catch_trials)/length(ind_catch_trials); %FPR

                        ind_eeg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,8) == 1)); % col 8 - EEG decisions
                        ind_eeg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,8) == 1));
                        EEG_TPR = length(ind_eeg_success_valid_trials)/length(ind_valid_trials);
                        EEG_FPR = length(ind_eeg_failed_catch_trials)/length(ind_catch_trials);

                        % Correction: Use col 5 - Intent detected instead of col 9 - EEG+EMG decisions
                        ind_eeg_emg_success_valid_trials = find((block_performance(:,4) == 1) & (block_performance(:,5) == 1)); 
                        ind_eeg_emg_failed_catch_trials = find((block_performance(:,4) == 2) & (block_performance(:,5) == 1));
                        EEG_EMG_TPR = length(ind_eeg_emg_success_valid_trials)/length(ind_valid_trials);
                        EEG_EMG_FPR = length(ind_eeg_emg_failed_catch_trials)/length(ind_catch_trials);

                        tpr_fpr_performance = [tpr_fpr_performance;...
                                            [subj_n ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR]];                       
                    end % ends block_n loop
                end % ends ses_n loop               
            end % ends subj_n loop

                    %        1          2                 3                 4                 5               6                     7                          8                             9
                    % [subj_n  ses_n       block_n block_TPR block_FPR EEG_TPR      EEG_FPR        EEG_EMG_TPR     EEG_EMG_FPR
                                         
                    figure('Position',[100 1100 3.5*116 3.5*116]);     % [left bottom width height]
                    T_plot = tight_subplot(1,2,[0.01 0.05],[0.01 0.25],[0.05 0.01]);
                   
                    for perf = 8:9
                        Ses4_perf =  tpr_fpr_performance((tpr_fpr_performance(:,2) == 4),[1, perf]);        % EEG_EMG_TPR or FPR
                        Ses4_perf = [Ses4_perf; [5*ones(size(Ses4_perf,1),1) Ses4_perf(:,2)]];
                        Ses5_perf =  tpr_fpr_performance((tpr_fpr_performance(:,2) == 5),[1, perf]);        % EEG_EMG_TPR or FPR
                        Ses5_perf = [Ses5_perf; [5*ones(size(Ses5_perf,1),1) Ses5_perf(:,2)]];
                        
                        
                        axes(T_plot(perf-7)); 
                        hold on;
                        hbox1_axes = boxplot(100.*Ses5_perf(:,2), Ses5_perf(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[1.15 2.15 3.15 4.15 5.15]); % symbol - Outliers take same color as box
                        set(hbox1_axes,'LineWidth',1);
                        set(gca,'XtickLabel',{' '});
                        set(hbox1_axes(7,1:5),'MarkerFaceColor',[0.6 0.6 0.6])
                        set(hbox1_axes(7,1:5),'MarkerSize',4);
                        set(hbox1_axes(6,1:5),'Color','k');
                        set(hbox1_axes(5,1:5),'Color','k');
                        set(hbox1_axes([1 2],1:5),'LineStyle','-');
                                              
                        
                        h_colors = findobj(gca,'Tag','Box');
                        for j = 1:length(h_colors)
                            h_patch(j) = patch(get(h_colors(j),'XData'), get(h_colors(j),'YData'),[0.6 0.6 0.6]);
                        end
                        hbox2_axes = boxplot(100.*Ses4_perf(:,2), Ses4_perf(:,1),'plotstyle','traditional','widths',0.2,'labelorientation','horizontal','symbol','ko','colors','k',...
                                                                                                                                 'positions',[0.85 1.85 2.85 3.85 4.85]); % symbol - Outliers take same color as box
                        set(hbox2_axes,'LineWidth',1);                   
                        set(hbox2_axes(7,1:5),'MarkerSize',4);
                        set(gca,'XtickLabel',{' '});
                        set(hbox2_axes(6,1:5),'Color','k');
                        set(hbox2_axes(5,1:5),'Color','k');
                        set(hbox2_axes([1 2],1:5),'LineStyle','-');
                        
                        h_colors_w = findobj(gca,'Tag','Box');
                        for j = 1:1 %length(h_colors_w)
                            h_patch_w = patch(get(h_colors_w(j),'XData'), get(h_colors_w(j),'YData'),[1 1 1]);
                        end
                        
                        h_axes = gca;                            
                        axis(h_axes,[0.5 5.5 -10  115]);
                        set(h_axes,'YGrid','on')
                        set(h_axes,'YTick',[0 25 50 75 100]);
                        set(h_axes,'Xtick',[(0.85 + 1.15)/2 (1.85 + 2.15)/2 (2.85 + 3.15)/2 (3.85 + 4.15)/2 (4.85 + 5.15)/2]);
                                                
                        if perf == 8
                            title(h_axes,'True Positives (%)');
                            set(h_axes,'XtickLabel',{'S1', 'S2', 'S3', 'S4','All','FontSize',9});
                            set(h_axes,'YTickLabel',{'0' '25' '50' '75' '100'},'FontSize',9,'YColor','k');
                        else
                            title(h_axes,'False Positives (%)');
                            set(h_axes,'XtickLabel',{'S1', 'S2', 'S3', 'S4','All'},'FontSize',9);
                            set(h_axes,'YTickLabel',{' '},'FontSize',9,'YColor','k');
                        end
                        %ylabel(h_axes,'Performance ');
                        set(h_axes,'Box','on')
                        hxlab = xlabel(gca,{' ';'Subjects'});
                        %pos_hxlab = get(hxlab,'Position');
                        %set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 3) pos_hxlab(3)]);
                        
                        % Significance tests
                    	p_values = [];
                        for subj_n = 1:5
                            % both-tailed Wilcoxon Rank sum Test, i.e. median(day 4) >< median(day 5)
                            [pwilcoxon,h,stats] = ranksum(Ses4_perf((Ses4_perf(:,1) == subj_n),2),Ses5_perf((Ses5_perf(:,1) == subj_n),2),'alpha',0.05,'tail','both');
                            if (pwilcoxon <= 0.05) && (pwilcoxon > 0.01) 
                                p_values = [p_values 0.05];
                            elseif (pwilcoxon <= 0.01)
                                p_values = [p_values 0.01];
                            else
                                p_values = [p_values NaN];
                            end
                        end
                        sigstar({[0.85 1.15],[1.85 2.15], [2.85 3.15], [3.85 4.15],[4.85 5.15]},p_values);
                            
                    end
                         axes_pos = get(gca,'Position');    % [left bottom width height]
                         box_leg = legend([h_patch_w h_patch(4)],'Day 4', 'Day 5', 'location','NorthOutside','Orientation','horizontal');
                         box_leg_pos = get(box_leg,'position');       
                         box_leg_title = get(box_leg,'title');
                         set(box_leg_title,'String','Closed-loop EEG control with EMG gating');
                         set(box_leg,'FontSize',9,'box','on');
                         set(box_leg,'position',[0.275, 0.85, box_leg_pos(3:4)]);
             
             
            response = input('Save figure to folder [y/n]: ','s');
            if strcmp(response,'y')
            %     tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_day4_day5.tif'];
            %    fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_day4_day5.fig'];
            %    print('-dtiff', '-r600', tiff_filename); 
            %    saveas(gcf,fig_filename);
            else
                disp('Save figure aborted');
            end

end    

%% Plotting features vs Intents/min

if compare_closed_loop_features == 1
                figure('Position',[1000 1400 6*116 4*116]);
                T_plot = tight_subplot(2,4,[0.15 0.05],[0.1 0.1],[0.1 0.01]);
                subj_n = 3;
                for n = 1:length(Sess_nums)
                    ses_n = Sess_nums(n);
                    folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
                    fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
                    if ~exist(fileid,'file')
                        continue
                    end
                    cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1);               
                                           
                    ind_success_valid_trials = find((cl_ses_data(:,4) == 1) & (cl_ses_data(:,5) == 1)); % col 4 - Valid(1) or Catch(2), col 5 - Intent detected,                
                    Ses_Intent_per_min{n} = 60./cl_ses_data(ind_success_valid_trials,6);                        % col 6 - Time to trigger
                    Ses_features{n} = cl_ses_data(ind_success_valid_trials,10:13);       % [slope, -ve peak, area, mahalanobis]
                    likert_score{n} = cl_ses_data(ind_success_valid_trials,17);
                end
                                           
                    grouping_vector = [ones(size(Ses_Intent_per_min{1},1),1); 2*ones(size(Ses_Intent_per_min{2},1),1)];
                    combined_likert_score = [likert_score{1};likert_score{2}];
                    combined_group = [(11:15)';(21:25)'; grouping_vector*10 + combined_likert_score];
                    %combined_likert_score(combined_likert_score == 2) = 1;
                    %combined_likert_score(combined_likert_score == 4) = 5;
                    unique(combined_likert_score)
                    length_ses4 = size(Ses_Intent_per_min{1},1);
                    length_ses5 = size(Ses_Intent_per_min{2},1);
                    Combined_intent_per_min = [Ses_Intent_per_min{1};Ses_Intent_per_min{2}];
                    Combined_features_vectors = [Ses_features{1}(:,:); Ses_features{2}(:,:)];
                    
                    [c1,ia1,ic1] = unique(likert_score{1});
                    likert1_freq = [];
                    for i = 1:numel(c1)
                        likert1_freq = [likert1_freq length(find(likert_score{1} == c1(i)))];
                    end
                    
                    [c2,ia2,ic2] = unique(likert_score{2});
                    likert2_freq = [];
                    for i = 1:numel(c2)
                        likert2_freq = [likert2_freq length(find(likert_score{2} == c2(i)))];
                    end
                    
                    display(['Likert Scores: Day 4 - ' num2str(c1') '; Day 5 -  ' num2str(c2')]);
                    display(['Freq: Day 4 - ' num2str(likert1_freq) '; Day 5 -' num2str(likert2_freq)]);
                    sum(likert1_freq)
                    sum(likert2_freq)
                               
                    %% plot 1
                    axes(T_plot(1)); hold on;
                    feature_to_plot = 1;
                    if plot_with_likert == 1
                        hscatter_1 = gscatter([100*ones(10,1);Combined_features_vectors(:,feature_to_plot)],[100*ones(10,1);Combined_intent_per_min],...
                                                               combined_group,'rymkbrymkb','oooooxxxxx',[],'off','Slope','Intents per min');
                    else
                         hscatter_1 = gscatter(Combined_features_vectors(:,feature_to_plot),Combined_intent_per_min,...
                                                               grouping_vector,'rb','ox',[],'off','Slope','Intents per min');
                    end
                    set(hscatter_1(1:size(hscatter_1,1)),'MarkerSize',6);%set(hscatter_1(2),'MarkerSize',3);
                    
                    hparent = get(hscatter_1(1),'parent');
                    xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    ylim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                    if subj_n == 3
                        ylim([0 100]);
                    end
                    xlim_val = xlim;
                    ylim_val = ylim;
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{num2str(ylim_val(1)) num2str(mean(ylim_val)) num2str(ylim_val(2))});  
                    set(gca,'FontSize',paper_font_size-1);                   
                    if plot_with_likert == 1
                        legendflex([hscatter_1(4), hscatter_1(9)],{'Day 4','Day 5'},'ncol',2, 'ref',T_plot(1),'anchor',[1 7],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5]);
                        legendflex([hscatter_1(1:5)],{'1','2','3','4','5'},'ncol',5,'nrow',1','ref',T_plot(4),'anchor',[3 5],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5],'title','Likert scores');
                    else
                        legendflex([hscatter_1(1), hscatter_1(2)],{'Day 4','Day 5'},'ncol',2, 'ref',T_plot(1),'anchor',[1 7],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5]);
                    end
                   
                   %% plot 2
                    axes(T_plot(2)); hold on;
                    feature_to_plot = 4;
                    if plot_with_likert == 1
                        hscatter_2 = gscatter([100*ones(10,1);Combined_features_vectors(:,feature_to_plot)],[100*ones(10,1);Combined_intent_per_min],...
                                                                combined_group,'rymkbrymkb','oooooxxxxx',[],'off','Mahalanobis',' ');
                    else
                        hscatter_2 = gscatter(Combined_features_vectors(:,feature_to_plot),Combined_intent_per_min,...
                                                               grouping_vector,[[1 0 0];[0 0 1]],'ox',[],'off','Mahalanobis', ' ');
                    end
                    set(hscatter_2(1:size(hscatter_2,1)),'MarkerSize',6);%set(hscatter_1(2),'MarkerSize',3);
                    hparent = get(hscatter_2(1),'parent');
                    xlim([floor(0), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);        
                    ylim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                    xlim_val = xlim;
                    ylim_val = ylim;
                    if subj_n == 3
                        ylim([0 100]);
                    end
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{' '});  
                   set(gca,'FontSize',paper_font_size-1,'XDir','normal');
                    
                   %% plot 3
                    axes(T_plot(3)); hold on;
                    feature_to_plot = 2;
                    if plot_with_likert == 1
                        hscatter_3 = gscatter([100*ones(10,1);Combined_features_vectors(:,feature_to_plot)],[100*ones(10,1);Combined_intent_per_min],...
                                                                combined_group,'rymkbrymkb','oooooxxxxx',[],'off','-ve Peak',' ');
                    else
                        hscatter_3 = gscatter(Combined_features_vectors(:,feature_to_plot),Combined_intent_per_min,...
                                                                grouping_vector,[[1 0 0];[0 0 1]],'ox',[],'off','-ve Peak', ' ');
                    end

                    set(hscatter_3(1:size(hscatter_3,1)),'MarkerSize',6);%set(hscatter_1(2),'MarkerSize',3);
                    hparent = get(hscatter_3(1),'parent');
                    xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    ylim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                    xlim_val = xlim;
                    ylim_val = ylim;
                    if subj_n == 3
                        ylim([0 100]);
                    end
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{' '});  
                   set(gca,'FontSize',paper_font_size-1);    
                    
                   %% plot 4
                   axes(T_plot(4)); hold on;
                    feature_to_plot = 3;
                    if plot_with_likert == 1
                        hscatter_4 = gscatter([100*ones(10,1);Combined_features_vectors(:,feature_to_plot)],[100*ones(10,1);Combined_intent_per_min],...
                                                                combined_group,'rymkbrymkb','oooooxxxxx',[],'off','Area',' ');
                    else
                        hscatter_4 = gscatter(Combined_features_vectors(:,feature_to_plot),Combined_intent_per_min,...
                                                                grouping_vector,[[1 0 0];[0 0 1]],'ox',[],'off','Area', ' ');                                         
                    end
                    set(hscatter_4(1:size(hscatter_4,1)),'MarkerSize',4);%set(hscatter_1(2),'MarkerSize',3);
                    hparent = get(hscatter_4(1),'parent');
                    xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    ylim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                    xlim_val = xlim;
                    ylim_val = ylim;
                    if subj_n == 3
                        ylim([0 100]);
                    end
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{' '});  
                   set(gca,'FontSize',paper_font_size-1);  
                    
                    %% plot 5
                    axes(T_plot(5));
                    feature_to_plot = 1;
                    [f_s4,x_s4] = hist(Combined_features_vectors(1:length_ses4,feature_to_plot),20);
                    [f_s5,x_s5] = hist(Combined_features_vectors(length_ses4+1:length_ses4+length_ses5,feature_to_plot),20);
                    h_patch(1) = jbfill(x_s4,f_s4./trapz(x_s4,f_s4), zeros(1,20),'r','r',1,0.5);
                    h_patch(2) = jbfill(x_s5,f_s5./trapz(x_s5,f_s5), zeros(1,20),'b','b',1,0.5);
                   xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    xlim_val = xlim;
                    ylim([0 1]);
                    set(gca,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[0 0.5 1],'YtickLabel',{'0' '0.5' '1'});  
                    xlabel('Slope','FontSize',paper_font_size-1); ylabel('PDF','FontSize',paper_font_size - 1);

                    [legend_h,object_h,plot_h,text_str] = ...
                        legendflex([h_patch(1), h_patch(2)],{'Day 4','Day 5'},'ncol',2, 'ref',T_plot(5),...
                                            'anchor',[1 7],'buffer',[0 0],'box','off','xscale',0.3,'padding',[0 5 5]);
                        set(object_h(3),'FaceAlpha',0.5);
                        set(object_h(4),'FaceAlpha',0.5);
    
                    %% plot 6
                    axes(T_plot(6));
                    feature_to_plot = 4;
                    [f_s4,x_s4] = hist(Combined_features_vectors(1:length_ses4,feature_to_plot),20);
                    [f_s5,x_s5] = hist(Combined_features_vectors(length_ses4+1:length_ses4+length_ses5,feature_to_plot),20);
                    jbfill(x_s4,f_s4./trapz(x_s4,f_s4), zeros(1,20),'r','r',1,0.5);
                    jbfill(x_s5,f_s5./trapz(x_s5,f_s5), zeros(1,20),'b','b',1,0.5);
                   xlim([floor(0), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    xlim_val = xlim;
                    ylim([0 1]);
                    set(gca,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[0 0.5 1],'YtickLabel',{' '});  
                    xlabel('Mahalanobis','FontSize',paper_font_size-1); %ylabel('PDF','FontSize',paper_font_size - 1);
                    set(gca,'XDir','normal');
                    
                    %% plot 7
                    axes(T_plot(7));
                    feature_to_plot = 2;
                    [f_s4,x_s4] = hist(Combined_features_vectors(1:length_ses4,feature_to_plot),20);
                    [f_s5,x_s5] = hist(Combined_features_vectors(length_ses4+1:length_ses4+length_ses5,feature_to_plot),20);
                    jbfill(x_s4,f_s4./trapz(x_s4,f_s4), zeros(1,20),'r','r',1,0.5);
                    jbfill(x_s5,f_s5./trapz(x_s5,f_s5), zeros(1,20),'b','b',1,0.5);
                   xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    xlim_val = xlim;
                    ylim([0 1]);
                    set(gca,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[0 0.5 1],'YtickLabel',{' '});  
                    xlabel('-ve Peak','FontSize',paper_font_size-1); %ylabel('PDF','FontSize',paper_font_size - 1);
                    
                    %% plot 8
                    axes(T_plot(8));
                    feature_to_plot = 3;
                    [f_s4,x_s4] = hist(Combined_features_vectors(1:length_ses4,feature_to_plot),20);
                    [f_s5,x_s5] = hist(Combined_features_vectors(length_ses4+1:length_ses4+length_ses5,feature_to_plot),20);
                    jbfill(x_s4,f_s4./trapz(x_s4,f_s4), zeros(1,20),'r','r',1,0.5);
                    jbfill(x_s5,f_s5./trapz(x_s5,f_s5), zeros(1,20),'b','b',1,0.5);
                   xlim([floor(mean(Combined_features_vectors(:,feature_to_plot))-2*std(Combined_features_vectors(:,feature_to_plot))), ...
                                ceil(mean(Combined_features_vectors(:,feature_to_plot))+2*std(Combined_features_vectors(:,feature_to_plot)))]);
                    xlim_val = xlim;
                    ylim([0 1]);
                    set(gca,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[0 0.5 1],'YtickLabel',{' '});  
                    xlabel('Area','FontSize',paper_font_size-1); %ylabel('PDF','FontSize',paper_font_size - 1);
                    
                    
                    mtit(['Subject S' num2str(subj_n)],'fontsize',paper_font_size-1,'xoff',0.0,'yoff',-0.06);
                    fig = gcf;
                    style = hgexport('factorystyle');
                    style.Bounds = 'tight';
                    hgexport(fig,'-clipboard',style,'applystyle', true);
                    drawnow;

                    response = input('Save figure to folder [y/n]: ','s');
                    if strcmp(response,'y')
                        if plot_with_likert == 1
                            tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_names{subj_n} '_compare_modes_likert.tif'];
                        else
                            tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_names{subj_n} '_compare_modes.tif'];
                        end
                         print('-dtiff', '-r600', tiff_filename); 
                    else
                        disp('Save figure aborted');
                    end
                    
                    %% plot 9 - Feature space combining day 4 and 5
                     if plot_with_likert == 1                    
                        figure('Position',[1000 1400 6*116 4*116]);
                     else
                        figure('Position',[1000 1400 3.5*116 2*116]);
                     end
                    U_plot = tight_subplot(1,2,[0.1 0.15],[0.2 0.2],[0.15 0.01]);
                    
                    axes(U_plot(1)); hold on;
                    features_to_plot = [1 4];
                    total_intent_min = [100*ones(10,1);Combined_intent_per_min];
                    if plot_with_likert == 1  
                        fscatter_1 = gscatter([100*ones(10,1);Combined_features_vectors(:,features_to_plot(1))],[100*ones(10,1);Combined_features_vectors(:,features_to_plot(2))],...
                                                                combined_group,'rymkbrymkb','oooooxxxxx',[],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});    
                    else
                        fscatter_1 = gscatter(Combined_features_vectors(:,features_to_plot(1)),Combined_features_vectors(:,features_to_plot(2)),...
                                                                grouping_vector,[[1 0 0];[0 0 1]],'ox',[],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});     
                    end                                       
                    set(fscatter_1(1:length(fscatter_1)),'MarkerSize',5); 
                    hparent = get(fscatter_1(1),'parent');
                    xlim([floor(mean(Combined_features_vectors(:,features_to_plot(1)))-2*std(Combined_features_vectors(:,features_to_plot(1)))), ...
                                ceil(mean(Combined_features_vectors(:,features_to_plot(1)))+2*std(Combined_features_vectors(:,features_to_plot(1))))]);
                    ylim([floor(0),...
                                ceil(mean(Combined_features_vectors(:,features_to_plot(2)))+2*std(Combined_features_vectors(:,features_to_plot(2))))]);
                    xlim_val = xlim;
                    ylim_val = ylim;
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{num2str(ylim_val(1)) num2str(mean(ylim_val)) num2str(ylim_val(2))});  
                    set(gca,'FontSize',paper_font_size-1);                                     
                    
                    if plot_with_likert == 1
                        gu = unique(combined_group);
                        for k = 1:numel(gu)
                              set(fscatter_1(k), 'ZData', total_intent_min( combined_group == gu(k) ));
                        end
                        view(3);                            
                        zlim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                        zlabel('Intents per min','FontSize',paper_font_size-1);
                         legendflex([fscatter_1(4), fscatter_1(9)],{'Day 4','Day 5'},'ncol',2, 'ref',U_plot(1),'anchor',[1 7],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5]);
                         legendflex([fscatter_1(1:5)],{'1','2','3','4','5'},'ncol',5,'nrow',1','ref',U_plot(2),'anchor',[3 5],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5],'title','Likert scores');
                    else
                        legendflex([fscatter_1(1), fscatter_1(2)],{'Day 4','Day 5'},'ncol',2, 'ref',U_plot(1),'anchor',[1 7],'buffer',[0 5],'box','on','xscale',0.3,'padding',[0 5 5]);   
                    end
                    %% plot 10                   
                    axes(U_plot(2)); hold on;
                    features_to_plot = [2 3];
                    if plot_with_likert == 1
                         fscatter_2 = gscatter([100*ones(10,1);Combined_features_vectors(:,features_to_plot(1))],[100*ones(10,1);Combined_features_vectors(:,features_to_plot(2))],...
                                                            combined_group,'rymkbrymkb','oooooxxxxx',[],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});                    
                    else
                        fscatter_2 = gscatter(Combined_features_vectors(:,features_to_plot(1)),Combined_features_vectors(:,features_to_plot(2)),...
                                                                grouping_vector,[[1 0 0];[0 0 1]],'ox',[],'off',features_names{features_to_plot(1)},features_names{features_to_plot(2)});     
                    end         
                   
                    set(fscatter_2(1:length(fscatter_2)),'MarkerSize',5); %set(fscatter_2(2),'MarkerSize',4);
                    
                    hparent = get(fscatter_2(1),'parent');
                    xlim([floor(mean(Combined_features_vectors(:,features_to_plot(1)))-2*std(Combined_features_vectors(:,features_to_plot(1)))), ...
                                ceil(mean(Combined_features_vectors(:,features_to_plot(1)))+2*std(Combined_features_vectors(:,features_to_plot(1))))]);
                    ylim([floor(mean(Combined_features_vectors(:,features_to_plot(2)))-2*std(Combined_features_vectors(:,features_to_plot(2)))), ...
                                ceil(mean(Combined_features_vectors(:,features_to_plot(2)))+2*std(Combined_features_vectors(:,features_to_plot(2))))]);
                    xlim_val = xlim;
                    ylim_val = ylim;
                    set(hparent,'XTick',[xlim_val(1) mean(xlim_val) xlim_val(2)],'XTickLabel',{num2str(xlim_val(1)) num2str(mean(xlim_val)) num2str(xlim_val(2))},...
                        'Ytick',[ylim_val(1) mean(ylim_val) ylim_val(2)],'YtickLabel',{num2str(ylim_val(1)) num2str(mean(ylim_val)) num2str(ylim_val(2))});  
                    set(gca,'FontSize',paper_font_size-1);                 
                    if plot_with_likert == 1
                         gu = unique(combined_group);
                        for k = 1:numel(gu)
                              set(fscatter_2(k), 'ZData', total_intent_min( combined_group == gu(k) ));
                        end
                        view(3);                            
                        zlim([floor(0), ceil(mean(Combined_intent_per_min)+2*std(Combined_intent_per_min))]);
                        zlabel('Intents per min','FontSize',paper_font_size-1);
                    end
                    
                    mtit(['Subject S' num2str(subj_n)],'fontsize',paper_font_size-1,'xoff',0.0,'yoff',-0.06);
                    fig = gcf;
                    style = hgexport('factorystyle');
                    style.Bounds = 'tight';
                    hgexport(fig,'-clipboard',style,'applystyle', true);
                    drawnow;

                    response = input('Save figure to folder [y/n]: ','s');
                    if strcmp(response,'y')
                        if plot_with_likert == 1
                            tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_names{subj_n} '_compare_feature_space_likert.tif'];
                        else
                            tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\' Subject_names{subj_n} '_compare_feature_space.tif'];
                        end
                         print('-dtiff', '-r600', tiff_filename); 
                    else
                        disp('Save figure aborted');
                    end
               
                    
end

