% TNSRE Abstract Plots
mycolors_square = {'-rs','-bs','-ks','-gs','-bs'};
mycolors_circle = {'-ro','-bo','-ko','-go','-bo'};
myfacecolors = ['r','b','k','g','b'];
posterFontSize = 18;
%load('acc_per_session.mat');
%load('err_per_session.mat');
% acc_per_session = [acc_per_session(1:2,:); acc_per_session(5,:); acc_per_session(3:4,:)]
% err_per_session = [err_per_session(1:2,:); err_per_session(5,:); err_per_session(3:4,:)]

%acc_per_session = 100.*acc_per_session;

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
%max_block_range = 21;
max_ses3 = 4;
max_ses4 = 7;
max_ses5 = 8;
Subject_names = {'BNBO','PLSH','LSGR','ERWS','JF'};
Sess_nums = 4:5;

maxY = 40;
figure('Position',[0 100 11*116 6*116]);
R_plot = tight_subplot(2,3,[0.025],[0.2 0.4],[0.15 0.1]);
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

for subj_n = 1:1
    bmi_performance = [];
    patient_performance = [];
    block_likert =[];    
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
        end % ends block_n loop
        
        %    1      2        3         4        5       6         7           8             9                   10  
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
            case 4
                axes(R_plot(3*subj_n-2)); 
                %hold on; grid on;
                %plot(1:length(plotids), block_TPR(plotids),'-bs','LineWidth',1.5,'LineStyle','-','MarkerFaceColor','b','MarkerSize',8);
                errorbar(1:length(plotids),bmi_performance(plotids,9),bmi_performance(plotids,10),'bs','MarkerFaceColor','b','MarkerSize',10);
                hold on;
                plot(1:length(plotids), (60/15).*bmi_performance(plotids,4),'rs','MarkerFaceColor','r','MarkerSize',10);
                axis([-1 max_ses5 0 maxY]);
                
                if subj_n == 1
%                     set(gca,'XTickLabel',1:max_ses4,'FontSize',12);
%                     hxlab = xlabel('Blocks of 20 trials','FontSize',12);
%                     pos_hxlab = get(hxlab,'Position');
%                     set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.5) pos_hxlab(3)]);
%                     text(-5,-10.25,'*p < 0.05','FontSize',12);
                     %mtit('Closed-loop BMI Performance','FontSize',12);
                    leg2 = legend('Motor Intents Detected/min (Average +/- S.D)', 'False Positives/min','Location','North','Orientation','Vertical');
                    set(leg2,'FontSize',posterFontSize,'box','off');
                    pos_leg2 = get(leg2,'Position');
                    set(leg2,'Position',[pos_leg2(1)+0.1 (pos_leg2(2) + 0.25) pos_leg2(3) pos_leg2(4)]);
                else
                    set(gca,'XTickLabel','','FontSize',posterFontSize);
                end
                
                if subj_n == 1
                    title('Day 4 (Backdrive Mode)','FontSize', posterFontSize);
                end
                
                set(gca,'YTick',[0 maxY/2 maxY]);
                set(gca,'YTickLabel','');
                %set(gca,'YTickLabel',{'0' '10' '20' '30'},'FontSize',12);
                set(gca,'XTick',[1:max_ses4]); 
                set(gca,'XTickLabel','');
                 set(gca,'YTick',[0 maxY/2 maxY]);
                set(gca,'YTickLabel',{'0' num2str(maxY/2) num2str(maxY)},'FontSize',posterFontSize-2);
                
                h1 = ylabel(Subject_names{subj_n},'FontSize',posterFontSize,'Rotation',0);
                posy = get(h1,'Position');
                set(h1,'Position',[(posy(1) - 2) posy(2:3)]);
                text(posy(1) - 3,posy(2)+8,'Subject','FontSize',posterFontSize);
                
                axes(R_plot(4))
                barh = bar(1:length(plotids), bmi_performance(plotids,11));
                axis([-1 max_ses5 0 8]);
                set(gca,'XTickLabel',1:max_ses4,'FontSize',posterFontSize-2);
                set(gca,'YTick', [0 6])
                set(gca,'YTickLabel',{'0' '6'},'FontSize',posterFontSize-2);
                set(barh,'FaceColor',[1 1 1],'LineWidth',2);
                barylab = ylabel('Block','FontSize',posterFontSize,'Rotation',0);
                posbar = get(barylab,'Position');
                set(barylab,'Position',[(posbar(1))-2 posbar(2:3)]);
                text(posbar(1) -3.2,posbar(2)-2,'Duration (min)','FontSize',posterFontSize);
                hxlab = xlabel('Blocks of 20 trials','FontSize',posterFontSize);
                pos_hxlab = get(hxlab,'Position');
                set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.5) pos_hxlab(3)]);
                %text(-5,-10.25,'*p < 0.05','FontSize',12);

                
            case 5
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
                set(hxlab,'Position',[pos_hxlab(1) (pos_hxlab(2) - 0.5) pos_hxlab(3)]);
                
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
    if strcmp( Subject_names{subj_n} ,'ERWS') 
        bmi_performance((bmi_performance (:,1) == 3),:) = [];
    elseif  strcmp( Subject_names{subj_n} ,'JF')
        bmi_performance((bmi_performance (:,1) == 3),:) = [];
        bmi_performance((bmi_performance (:,1) == 4),:) = [];
    end
        
    mlr_intents = LinearModel.fit(1:length(plotids),bmi_performance(plotids,9));
    if mlr_intents.coefTest <= 0.05
        line_regress = [ones(2,1) [1; size(plotids,1)]]*mlr_intents.Coefficients.Estimate;
        %axes(R_plot(4*subj_n-3)); 
        axis(axis);
        %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
        plot([1  size(plotids,1)],line_regress,':b','LineWidth',0.2); hold on;
         text(6,[1 7]*mlr_intents.Coefficients.Estimate + 1.75,'*','FontSize',16)
    end

        mlr_fpr_min = LinearModel.fit(1:length(plotids), (60/15).*bmi_performance(plotids,4));
    if mlr_fpr_min.coefTest <= 0.05
        line_regress = [ones(2,1) [1; length(plotids)]]*mlr_fpr_min.Coefficients.Estimate;
        %axes(R_plot(4*subj_n-3)); 
        axis(axis);
        %hold on; arrow([1 line_regress(1)],[size(patient_performance,1)+1 line_regress(2)],'LineWidth',1);
%          ah=axes('position',[.2,.2,.6,.6],'visible','off'); % <- select your pos...
        plot([1 size(plotids,1)],line_regress,':r','LineWidth',0.2); hold on;
        text(7,[1 7]*mlr_fpr_min.Coefficients.Estimate + 0.75,'*','FontSize',16)
    end
    
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
    pos = get(gca,'Position');
    set(gca,'Position',[pos(1)+0.05 pos(2:4)])
    if subj_n == 1
        tit = title('BMI Accuracy','FontSize',posterFontSize);
        pos_tit = get(tit,'Position'); 
        set(tit,'Position',[pos_tit(1)-0.1 pos_tit(2)+30 pos_tit(3)]);
        leg1 = legend(h,'without EMG validation', 'with EMG validation','Location','North','Orientation','Vertical');
        set(leg1,'FontSize',posterFontSize-2,'box','off');
        pos_leg1 = get(leg1,'Position');
        set(leg1,'Position',[pos_leg1(1)-0.02 (pos_leg1(2) + 0.15) pos_leg1(3) pos_leg1(4)]);
    end
    
    
end % ends subj_n loop
% print -dtiff -r450 All_subjects_block_accuracy_bw.tif
% saveas(gca,'All_subjects_block_accuracy_bw.fig')

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