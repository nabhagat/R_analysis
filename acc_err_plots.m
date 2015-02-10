% TNSRE Abstract Plots
clear
mycolors_square = {'-rs','-bs','-ks','-gs','-bs'};
mycolors_circle = {'-ro','-bo','-ko','-go','-bo'};
myfacecolors = ['r','b','k','g','b'];
posterFontSize = 10;
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
maxY_ranges = [90 30 55 40];

plot_intent_fpr_min = 0;
plot_num_attempts = 0;
plot_different_metrics = 0;
plot_intent_only = 1;
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

%% Plotting features vs Intents/min
% % 
% % for subj_n = 1:1               
% %                 for n = 1:length(Sess_nums)
% %                     ses_n = Sess_nums(n);
% %                     folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
% %                     fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
% %                     if ~exist(fileid,'file')
% %                         continue
% %                     end
% %                     cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
% %                     Ses_success_trials = find((cl_ses_data(:,4) == 1) & (cl_ses_data(:,5) == 1)); % col 5 - Intent detected,                
% %                     Ses_Intent_per_min = 60./cl_ses_data(Ses_success_trials,6);
% %                     Ses_features = cl_ses_data(Ses_success_trials,10:13);
% %     
% %                     [sorted_Ses_Intent, sort_order] = sort(Ses_Intent_per_min);
% %                     sorted_Ses_features = Ses_features(sort_order,:);
% %                    %sorted_Ses_Intent = Ses_Intent_per_min;
% %                    %sorted_Ses_features = Ses_features;
% %                     
% %                     figure('Position',[0 50 12*116 4*116]);
% %                     T_plot = tight_subplot(2,4,[0.15 0.1],[0.1],[0.1]);
% %                     
% %                     axes(T_plot(1));
% %                     plot(sorted_Ses_features(:,1), sorted_Ses_Intent,'or','MarkerSize',6,'LineWidth',2);
% %                     ylabel({'Motor Intents per min'},'FontSize',posterFontSize);
% %                     %xlabel('Slope','FontSize',posterFontSize);
% %                     ylim([0 40]);
% %                     xlim([-15 5]);
% %                     
% %                     axes(T_plot(2));
% %                     plot(sorted_Ses_features(:,2), sorted_Ses_Intent,'og','MarkerSize',6,'LineWidth',2);
% %                     %ylabel('Motor Intents per min','FontSize',posterFontSize);
% %                     %xlabel('-ve Peak','FontSize',posterFontSize);
% %                     set(gca,'YTickLabel',[]);
% %                     ylim([0 40]);
% %                     xlim([-15 5]);
% %                     
% %                     axes(T_plot(3));
% %                     plot(sorted_Ses_features(:,3), sorted_Ses_Intent,'ob','MarkerSize',6,'LineWidth',2);
% %                     %ylabel('Motor Intents per min','FontSize',posterFontSize);
% %                     %xlabel('AUC','FontSize',posterFontSize);
% %                     set(gca,'YTickLabel',[]);
% %                     ylim([0 40]);
% %                     xlim([-15 5]);
% %                     
% %                     axes(T_plot(4));
% %                     plot(sorted_Ses_features(:,4), sorted_Ses_Intent,'ok','MarkerSize',6,'LineWidth',2);
% %                     %ylabel('Motor Intents per min','FontSize',posterFontSize);
% %                     %xlabel('Mahalanobis','FontSize',posterFontSize);
% %                     set(gca,'YTickLabel',[]);
% %                     ylim([0 40]);
% %                     xlim([0 10]);
% %                     
% %                     axes(T_plot(5));
% %                     hist(sorted_Ses_features(:,1),10);
% %                     h1 = findobj(gca,'Type','patch');
% %                     title('Slope','FontSize',14);
% %                     set(h1,'FaceColor','w','EdgeColor','r','LineWidth',2);
% %                     xlim([-15 5]);
% %                     
% % 
% %                     axes(T_plot(6));
% %                     hist(sorted_Ses_features(:,2),10);
% %                     h1 = findobj(gca,'Type','patch');
% %                     title('-ve Peak','FontSize',14);
% %                     set(h1,'FaceColor','w','EdgeColor','g','LineWidth',2);
% %                     xlim([-15 5]);
% %                     
% %                     axes(T_plot(7));
% %                     hist(sorted_Ses_features(:,3),10);
% %                     h1 = findobj(gca,'Type','patch');
% %                     title('AUC','FontSize',14);
% %                     set(h1,'FaceColor','w','EdgeColor','b','LineWidth',2);
% %                     xlim([-15 5]);
% %                     
% %                     axes(T_plot(8));
% %                     hist(sorted_Ses_features(:,4),10);
% %                     h1 = findobj(gca,'Type','patch');
% %                     title('Mahalanobis','FontSize',14);
% %                     set(h1,'FaceColor','w','EdgeColor','k','LineWidth',2);
% %                     xlim([0 10]);
% %                     
% %                     mtit([Subject_names{subj_n} ', Day ' num2str(ses_n)],'fontsize',14,'yoff',0.025);
% % %%                    
% % %                     axes(T_plot(5));
% % %                     plot3(sorted_Ses_features(:,1), sorted_Ses_features(:,2), sorted_Ses_Intent,'-r','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     zlabel({'Motor Intents per min'},'FontSize',posterFontSize);
% % %                     xlabel('Slope','FontSize',posterFontSize);
% % %                     ylabel('-ve Peak','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% % %                     
% % %                     axes(T_plot(6));
% % %                     plot3(sorted_Ses_features(:,1), sorted_Ses_features(:,3), sorted_Ses_Intent,'-r','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     %zlabel({'Motor Intents per min';'sorted'},'FontSize',posterFontSize);
% % %                     xlabel('Slope','FontSize',posterFontSize);
% % %                     ylabel('AUC','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% % %                     
% % %                     axes(T_plot(7));
% % %                     plot3(sorted_Ses_features(:,1), sorted_Ses_features(:,4), sorted_Ses_Intent,'-r','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     %zlabel({'Motor Intents per min';'sorted'},'FontSize',posterFontSize);
% % %                     xlabel('Slope','FontSize',posterFontSize);
% % %                     ylabel('Mahalanobis','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% % %                     
% % %                     axes(T_plot(9));
% % %                     plot3(sorted_Ses_features(:,4), sorted_Ses_features(:,2), sorted_Ses_Intent,'-k','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     zlabel({'Motor Intents per min'},'FontSize',posterFontSize);
% % %                     xlabel('Mahalanobis','FontSize',posterFontSize);
% % %                     ylabel('-ve Peak','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% % %                     
% % %                     axes(T_plot(10));
% % %                     plot3(sorted_Ses_features(:,4), sorted_Ses_features(:,3), sorted_Ses_Intent,'-k','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     %zlabel({'Motor Intents per min';'sorted'},'FontSize',posterFontSize);
% % %                     xlabel('Mahalanobis','FontSize',posterFontSize);
% % %                     ylabel('AUC','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% % %                     
% % %                     axes(T_plot(11));
% % %                     plot3(sorted_Ses_features(:,3), sorted_Ses_features(:,2), sorted_Ses_Intent,'-b','MarkerSize',6, 'MarkerFaceColor', 'w');
% % %                     %zlabel({'Motor Intents per min';'sorted'},'FontSize',posterFontSize);
% % %                     xlabel('AUC','FontSize',posterFontSize);
% % %                     ylabel('-ve Peak','FontSize',posterFontSize);
% % %                     zlim([0 40]);
% % %                     grid on;
% %                     
% %                     %%  Histograms for feature vectors
% % %                     figure;
% % %                     subplot(2,2,1); hold on
% % %                     hist(sorted_Ses_features(:,1),10);
% % %                     h1 = findobj(gca,'Type','patch');
% % %                     title('Slope','FontSize',14);
% % %                     set(h1,'FaceColor','w','EdgeColor','r');
% % % 
% % %                     subplot(2,2,2); hold on
% % %                     hist(sorted_Ses_features(:,2),10);
% % %                     h2 = findobj(gca,'Type','patch');
% % %                     title('Negative Peak','FontSize',14);
% % %                     set(h1,'FaceColor','g');
% % %                     
% % %                     subplot(2,2,3); hold on
% % %                     hist(sorted_Ses_features(:,3),10);
% % %                     h3 = findobj(gca,'Type','patch');
% % %                     title('Area Under the Curve','FontSize',14);
% % %                     set(h1,'FaceColor','b');
% % %                     
% % %                     subplot(2,2,4); hold on
% % %                     hist(sorted_Ses_features(:,4),10);
% % %                     h4 = findobj(gca,'Type','patch');
% % %                     title('Mahalanobis Distance','FontSize',14);
% % %                     set(h1,'FaceColor','k');
% %                     
% %                     
% %                     
% %                 end
% % end

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
                     
            figure('Position',[100 1100 7*116 8*116]);     % [left bottom width height]
            height_inc = 0.05; 
            height_shift = 0.005;
            I_plot = tight_subplot(8,2,[0.05 0.02],[0.1 0.05],[0.1 0.02]);
            
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
                                             mean(Intent_per_min) std(Intent_per_min) block_duration_min median(Intent_per_min)]];
                        
                        Session_Intent_per_min = [Session_Intent_per_min; [block_n.*ones(length(Intent_per_min),1) Intent_per_min]];
                    end % ends block_n loop

                    %        1          2               3                 4                 5               6                     7                          8                             9                                      10                                11                          12   
                    % [ses_n block_n block_TPR block_FPR EEG_TPR EEG_FPR EEG_EMG_TPR EEG_EMG_FPR mean(Intent_per_min) std(Intent_per_min) block_duration median(Intent_per_min)]

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
                            set(hbox_axes(6,1:length(unique_blocks)),'Color','r');
                            h_axes = gca;
                            
                            %[herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min(:,2)),mean(Session_Intent_per_min(:,2)));
                            %set(herrbar_ses_intents1,'FaceColor',[1 1 1],'EdgeColor','b','LineWidth',1.5,'XData',11,'BarWidth',1);
                            %set(herrbar_ses_intents2,'Color','k','LineWidth',1.5,'XData',11);                         
                            h_overall = boxplot(h_axes, Session_Intent_per_min(:,2),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(h_overall(6),'Color','r');
                            set([hbox_axes h_overall],'LineWidth',1);
                            axis(h_axes,[0 max_ses4+4 0  maxY_ranges(subj_n)]);
                            set(h_axes,'YGrid','on')
                            if subj_n == 1
                                set(h_axes,'YTick',[0 25 maxY/2 75 maxY]);
                                set(h_axes,'YTickLabel',{'0' '25' num2str(maxY/2) '75' num2str(maxY)},'FontSize',posterFontSize,'YColor','k');
                            else
                                set(h_axes,'YTick',[0 10 25 maxY/2 75 maxY]);
                                set(h_axes,'YTickLabel',{'0' '10' '25' num2str(maxY/2) '75' num2str(maxY)},'FontSize',posterFontSize,'YColor','k');
                            end
                            
                            switch subj_n
                                case 1
                                    title('Day 4, S1 (AT)','FontSize',posterFontSize);
                                    
                                case 2
                                    title('S2 (\bfBD\rm)','FontSize',posterFontSize);
                                case 3
                                    title('S3 (AT)','FontSize',posterFontSize);
                                case 4
                                    title('S4 (\bfBD\rm)','FontSize',posterFontSize);
                            end
                            h1 = ylabel(h_axes,{'Intents per';'min'},'FontSize',posterFontSize,'Rotation',90);
                            posy = get(h1,'Position');                          
                                                                                                         
                            axes(I_plot(axes_no + 2))
                            
                            hold on;
                            barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                            set(barh,'FaceColor',[1 1 1],'LineWidth',1,'BarWidth',0.5);
                            %dur_overall = boxplot(gca, bmi_performance(plotids,11),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','+','boxstyle','outline','colors',[0.6 0.6 0.6]); % symbol - Outliers take same color as box
                            [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                            set(herrbar_dur1,'EdgeColor','k','LineWidth',1,'XData',11,'BarWidth',0.5,'FaceColor',[1 1 1]);
                            set(herrbar_dur2,'Color','k','LineWidth',1,'XData',11);
                            
                            axis([0 max_ses4+4 0 max(bmi_performance(plotids,11))+2]);
                            set(gca,'XTick',[1:length(unique_blocks) 11]); 
                            set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',posterFontSize);
                            ses4_median_dur = 5; % round(median(bmi_performance(plotids,11)));
                            set(gca,'YTick', [0 ses4_median_dur])
                            set(gca,'YTickLabel',{'0' num2str(ses4_median_dur)},'FontSize',posterFontSize);
                            set(gca,'XGrid','on','YGrid', 'on')
                            
                            %if subj_n == 1
                                barylab = ylabel({'Length';'(min)'},'FontSize',posterFontSize,'Rotation',90);
                                posbar = get(barylab,'Position');
                                set(barylab,'Position',[posy(1) posbar(2) posbar(3)]);
                            %end
                            %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',posterFontSize);
                            %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',posterFontSize);
                            
                            if subj_n == 4
                                hxlab = xlabel({'Blocks of 20 trials'},'FontSize',posterFontSize);
                                pos_hxlab = get(hxlab,'Position');
                                set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) - 1) pos_hxlab(3)]);
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
                            set(hbox_axes(6,1:length(unique_blocks)),'Color','r');
                            h_axes = gca;
                            h_overall = boxplot(h_axes, Session_Intent_per_min(:,2),'positions', [11],'plotstyle','traditional','widths',0.5,'symbol','o','colors','k'); % symbol - Outliers take same color as box
                            set(h_overall(6),'Color','r');
                            set([hbox_axes h_overall],'LineWidth',1);
                            axis(h_axes,[0 max_ses5+3 0  maxY_ranges(subj_n)]);
                            %[herrbar_ses_intents1 herrbar_ses_intents2] = barwitherr(std(Session_Intent_per_min(:,2)),mean(Session_Intent_per_min(:,2)));
                            %set(herrbar_ses_intents1,'FaceColor',[1 1 1],'EdgeColor','b','LineWidth',1.5,'XData',11,'BarWidth',1);
                            %set(herrbar_ses_intents2,'Color','k','LineWidth',1.5,'XData',11);                         
                            
                            set(h_axes,'YGrid','on')
                            if subj_n == 1
                                set(h_axes,'YTick',[0 25 maxY/2 75 maxY]);
                                set(h_axes,'YTickLabel',{' '});
                            else
                                set(h_axes,'YTick',[0 10 25 maxY/2 75 maxY]);
                                set(h_axes,'YTickLabel',{' '});
                            end
                            
                            switch subj_n
                                case 1
                                    title('Day 5, S1 (AT)','FontSize',posterFontSize);
                                    %h1 = ylabel(h_axes,{'Intents per';'min'},'FontSize',posterFontSize,'Rotation',90);
                                    %posy = get(h1,'Position');
                                case 2
                                    title('S2 (AT)','FontSize',posterFontSize);
                                case 3
                                    title('S3 (AT)','FontSize',posterFontSize);
                                case 4
                                    title('S4 (AT)','FontSize',posterFontSize);
                            end


                            
                            %h1 = ylabel(h_axes,{'Intents per min'},'FontSize',posterFontSize,'Rotation',90);
                            %posy = get(h1,'Position');
                            %set(h1,'Position',[(posy(1) - 0.05) posy(2:3)]);
                            %text(posy(1) - 3,posy(2)+8,'Subject','FontSize',posterFontSize);
                                                                             
                            axes(I_plot(axes_no + 2))
                            
                            hold on;
                            barh = bar([1:length(plotids)], bmi_performance(plotids,11));
                            set(barh,'FaceColor',[1 1 1],'LineWidth',1,'BarWidth',0.5);
                            [herrbar_dur1 herrbar_dur2] = barwitherr(std(bmi_performance(plotids,11)),mean(bmi_performance(plotids,11)));
                            set(herrbar_dur1,'EdgeColor','k','LineWidth',1,'XData',11,'BarWidth',0.5,'FaceColor',[1 1 1]);
                            set(herrbar_dur2,'Color','k','LineWidth',1,'XData',11);
                            
                            axis([0 max_ses5+3 0 max(bmi_performance(plotids,11))+2]);
                            set(gca,'XTick',[1:length(unique_blocks) 11]); 
                            set(gca,'XTickLabel',{(1:length(unique_blocks)) 'Overall'},'FontSize',posterFontSize);
                            set(gca,'YTick', [0 ses4_median_dur]);
                            set(gca,'YTickLabel',{' '},'FontSize',posterFontSize);
                            set(gca,'XGrid','on','Ygrid','on');
                            
                            %barylab = ylabel({'Duration';'(min)'},'FontSize',posterFontSize,'Rotation',90);
                            %posbar = get(barylab,'Position');
                            %set(barylab,'Position',[(posbar(1))-1 posbar(2)-4 posbar(3)]);
                            %text(posbar(1) -3.2,posbar(2)-2,'Duration','FontSize',posterFontSize);
                            %text(posbar(1) -2.7,posbar(2)-6,'(min)','FontSize',posterFontSize);
                            
                            if subj_n == 4
                                hxlab = xlabel({'Blocks of 20 trials'},'FontSize',posterFontSize);
                                pos_hxlab = get(hxlab,'Position');
                                set(hxlab,'Position',[pos_hxlab(1)-2 (pos_hxlab(2) - 1) pos_hxlab(3)]);
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

                mlr_intents = LinearModel.fit(1:length(plotids),bmi_performance(plotids,12));
                if mlr_intents.coefTest <= 0.05
                    line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_intents.Coefficients.Estimate;
                    %line_regress = line_regress - min(bmi_performance(plotids,12)');
                    axes(h_axes)
                    %hold(ax(1),'on');
                    %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
                    plot(h_axes,[-1  size(plotids,1)+1],line_regress,'--b','LineWidth',0.5); hold on;
                    text(size(plotids,1)+1,line_regress(2)+2,sprintf('Slope \n %.2f',mlr_intents.Coefficients.Estimate(2)),'FontSize',10)
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

            response = input('Save figure to folder [y/n]: ','s');
            if strcmp(response,'y')
                 tiff_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_intents_per_min.tif'];
                 fig_filename = ['C:\NRI_BMI_Mahi_Project_files\Figures_for_paper\all_subjects_intents_per_min.tif'];
                print('-dtiff', '-r300', tiff_filename); 
                saveas(gcf,fig_filename);
            else
                disp('Save figure aborted');
            end

end
                    
                    
                    
                    
