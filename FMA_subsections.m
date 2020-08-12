% Program to analyze changes in FM-UE by subsections

% NRI Project
% Created by Nikunj Bhagat on 11-26-2018
clc;
clear all;
blue = [0, 153, 255]./255; 
light_blue = [102 204 255]./255; % #66ccff
orange = [228, 108, 10]./255;
light_orange = [255, 179, 102]./255; % #ffb366
blue_color = [0 0.64  1];%[0, 114, 178]./255;
orange_color = [1 0.3 0]; %[213, 94, 0]./255;
purple_color = [0.3   0   0.6];
green_color = [0   0.5    0];

paper_font_size = 11;

%% Fugl-Meyer scores
% Categories order = [Flexor synergies, extensor synergies, combined synergies, movement out-of-synergy,...
%                     wrist, hand, speed/co-ordination., reflexes] 
FMA_cats = zeros(10,5,8); 
FMA_cats(:,:,1) = [11	10	10	11	11;
9	10	9	10	7;
10	9	10	10	10;
8	10	9	nan	11;
8	9	10	nan	9;
8	7	9	9	10;
9	9	10	9	nan;
9	9	9	10	10;
5	5	9	nan	10;
9	10	10	10	11];

FMA_cats(:,:,2) = [5	5	6	6	6;
2	4	4	3	2;
5	5	5	6	6;
3	2	3	nan	4;
4	4	4	nan	2;
5	4	4	6	6;
5	4	5	5	nan;
4	4	4	5	4;
3	3	5	nan	5;
4	4	4	4	5];

FMA_cats(:,:,3) = [4	4	5	5	5;
3	3	3	2	2;
3	5	5	4	6;
0	0	0	nan	2;
3	5	6	nan	5;
4	4	5	4	4;
4	4	5	6	nan;
4	4	5	6	5;
2	2	4	nan	3;
4	4	4	4	4];

FMA_cats(:,:,4) = [5	4	5	5	5;
0	0	1	1	1;
3	3	3	3	3;
0	0	0	nan	2;
3	3	3	nan	3;
2	2	2	2	2;
3	3	3	3	nan;
5	3	5	4	4;
2	2	0	nan	3;
1	1	2	4	2];

FMA_cats(:,:,5) = [7	7	9	9	9;
2	2	2	3	2;
9	8	8	8	10;
1	2	2	nan	0;
5	7	9	nan	7;
8	8	9	8	7;
4	4	5	8	nan;
8	8	8	9	6;
0	0	0	nan	0;
4	4	4	5	6];

FMA_cats(:,:,6) = [13	13	14	14	14;
2	2	1	1	2;
13	13	11	13	13;
1	2	2	nan	1;
9	10	11	nan	11;
13	13	13	14	13;
5	7	7	8	nan;
14	14	14	14	13;
3	3	1	nan	3;
5	5	6	8	8];

FMA_cats(:,:,7) = [4	4	3	4	4;
2	1	3	2	1;
2	1	1	2	2;
1	1	4	nan	3;
5	1	4	nan	2;
3	3	3	2	3;
5	3	4	5	nan;
3	3	4	3	4;
3	1	4	nan	5;
4	5	2	4	3];

FMA_cats(:,:,8) = [4	4	4	4	4;
4	4	4	4	4;
4	4	4	4	4;
4	4	4	nan	2;
4	4	4	nan	4;
4	4	4	4	4;
4	4	4	4	nan;
4	4	4	4	2;
4	4	4	nan	2;
4	4	4	4	4];

FMA_scores = sum(FMA_cats(:,:,:),3);

FMA_cats_max = [12 6 6 6 10 14 6 6];

%% ARAT, JTHFT, Grip and Pinch strength scores

ARAT_scores =  [43	55	53	55;
                4	4	6	5;
                45	54	42	49;
                4	4	nan	4;
                39	44	nan	40;
                30	36	38	42;
                25	32	35	nan;
                42	52	51	53;
                9	10	nan	9;
                12	14	18	16];
            
JTHFT_scores = [2.0800    2.3500    2.3800    1.9700;
                     0    0.0300         0         0;
                1.8500    1.8900    1.8900    1.9400;
                     0         0       NaN         0;
                1.4900    1.2600       NaN    1.4500;
                1.7300    1.9200    1.7100    2.0000;
                0.5200    0.4900    0.4800       NaN;
                2.2900    1.3100    1.8500    1.6400;
                     0         0       NaN    0.0700;
                0.5000    0.7200    0.9600    0.8900];

Grip_scores = [ 31.67	33.00	31.33	33.00;
                4.33	2.50	4.17	5.67;
                14.67	12.17	16.67	18.33;
                3.17	5.00	nan     2.67;
                15.00	10.67	nan     16.00;
                9.33	10.00	9.83	9.33;
                11.33	10.90	10.00	nan;
                14.16	27.33	15.67	12.00;
                2.60	4.00	nan     3.30;
                5.00	10.83	5.67	10.00];
            
Pinch_scores = [7.00	6.20	6.80	7.33;
                3.83	4.38	4.17	4.00;
                4.09	4.55	5.67	4.00;
                2.00	2.00	nan 	3.00;
                6.00	5.33	nan 	5.50;
                4.50	4.50	4.52	5.45;
                7.67	7.43	7.60	nan;
                5.67	10.33	5.00	4.83;
                0.00	2.00	nan 	2.50;
                4.00	3.33	4.50	4.33];

NIHSS_scores = [3	0	1	0;
                3	4	5	5;
                3	4	4	1;
                3	1	nan	1;
                4	3	nan	5;
                2	2	2	2;
                7	4	3	nan;
                4	4	3	4;
                3	1	nan	1;
                2	2	2	2];            
%% ARAT scores by categories
% Categories order = [grasp(0-18), grip(0-12), pinch(0-18), gross movement(0-9)] 
ARAT_cats = zeros(10,4,4); 
ARAT_cats(:,:,1) = [14	18	18	18;
                    0	0	1	1;
                    16	16	14	14;
                    0	1	nan	0;
                    12	18	nan	13;
                    8	12	12	15;
                    7	9	10	nan;
                    15	17	18	18;
                    1	1	nan	1;
                    3	4	6	3];
                
ARAT_cats(:,:,2) = [8	10	8	10;
                    0	0	0	0;
                    8	11	8	9;
                    0	0	nan	0;
                    8	8	nan	8;
                    7	8	8	8;
                    7	6	8	nan;
                    8	10	8	8;
                    3	4	nan	4;
                    4	5	5	5];
                
ARAT_cats(:,:,3) = [12	18	18	18;
                    0	0	0	0;
                    12	18	13	17;
                    0	0	nan	0;
                    12	10	nan	12;
                    9	11	12	12;
                    5	10	10	nan;
                    12	16	18	18;
                    0	0	nan	0;
                    2	1	3	3];
                
ARAT_cats(:,:,4) = [9	9	9	9;
                    4	4	5	4;
                    9	9	7	9;
                    4	3	nan	4;
                    7	8	nan	7;
                    6	5	6	7;
                    6	7	7	nan;
                    7	9	7	9;
                    5	5	nan	4;
                    3	4	4	5];


ARAT_cats_max =[18 12 18 9];
%% Plot average change from baseline for Fugl Meyer, ARAT, JTHFT, NIHSS, Grip & Pinch strength
deltaARAT_scores = ARAT_scores - repmat(ARAT_scores(:,1),[1 4]);
deltaFMA_scores = FMA_scores - repmat(FMA_scores(:,2),[1 5]);
deltaJTHFT_scores = JTHFT_scores - repmat(JTHFT_scores(:,1),[1 4]);
deltaNIHSS_scores = NIHSS_scores - repmat(NIHSS_scores(:,1),[1 4]);
deltaGrip_scores = Grip_scores - repmat(Grip_scores(:,1),[1 4]);
deltaPinch_scores = Pinch_scores - repmat(Pinch_scores(:,1),[1 4]);

tempchange = deltaFMA_scores(:,3:5);
nanmean(tempchange(:))
nanstd(tempchange(:))

tempchange = deltaARAT_scores(:,2:4);
nanmean(tempchange(:))
nanstd(tempchange(:))

transparency = 0.4;
figure('Position',[300 5 6*116 4*116]); 
Splot = tight_subplot(2,2,[0.125 0.05],[0.1 0.05],[0.1 0.05]);

axes(Splot(1)); hold on;
jbfill(0:0.5:4, 10*ones(1,9),-3.75*ones(1,9),[0.8, 0.8, 0.8],'none',1,transparency); 
xrange = [0, 4.5, 6.5, 13];
hold on;
h_fma = errorbar(xrange, nanmean(deltaFMA_scores(:,2:5)), nanstd(deltaFMA_scores(:,2:5)),'-o','Color',blue_color,'MarkerFaceColor',blue_color,'LineWidth',0.5,'MarkerSize',6);
ylim([-4 12]); 
xlim([-1 15]);
set(gca,'Ytick',[-2 0 2 4 6 8 10],'YTickLabel',-2:2:10,'FontSize',paper_font_size-1);
set(gca,'Xtick',xrange,'Xticklabel',{'Baseline', '1wk-', '2wk-', '2mo-Post'},'FontSize',paper_font_size-1);
% fix_xticklabels(gca,0.01);
%legend([h_arat h_fma],{'Action Research Arm Test','Fugl-Meyer Upper Ext.'},'Orientatio n','Vertical','Location','NorthEastOutside')
line([-1 15],[0 0],'LineStyle','--','Color','k')
title('FMA-UE','FontSize',paper_font_size,'FontWeight','normal');
ylabel('Change from Baseline','FontSize',paper_font_size);
set(gca,'Xgrid','on');
set(gca,'Ygrid','on');
% annotation('textbox',[0.9 0 0.1 0.07],'String','*','EdgeColor','none','FontSize',24); 
sigstar({[0,4.5],[0,6.5],[0,13]},[0.05,0.01,0.05]); % Does not work with errorbars
text(0.5,9.25,'Therapy','FontSize',paper_font_size - 1);

axes(Splot(2)); hold on;
jbfill(0:0.5:4, 10*ones(1,9),-3.75*ones(1,9),[0.8, 0.8, 0.8],'none',1,transparency); 
xrange = [0, 4.5, 6.5, 13];
hold on;
h_arat = errorbar(xrange, nanmean(deltaARAT_scores), nanstd(deltaARAT_scores),'-o','Color',orange_color,'MarkerFaceColor',orange_color,'LineWidth',0.5,'MarkerSize',6);
ylim([-4 14]); 
xlim([-1 15]);
set(gca,'Ytick',[-2 0 2 4 6 8 10 12],'YTickLabel',-2:2:12,'FontSize',paper_font_size-1);
set(gca,'Xtick',xrange,'Xticklabel',{'Baseline', '1wk-', '2wk-', '2mo-Post'},'FontSize',paper_font_size-1);
line([-1 15],[0 0],'LineStyle','--','Color','k')
title('ARAT','FontSize',paper_font_size,'FontWeight','normal');
% ylabel('Change from Baseline','FontSize',10);
grid('on');
sigstar({[0,4.5],[0,6.5],[0,13]},[0.05,0.05,0.05]); % Does not work with errorbars
text(0.5,9.25,'Therapy','FontSize',paper_font_size - 1);

axes(Splot(3)); hold on;
jbfill(0:0.5:4, 2*ones(1,9),-3*ones(1,9),[0.8, 0.8, 0.8],'none',1,transparency); 
xrange = [0, 4.5, 6.5, 13];
hold on;
h_jthft = errorbar(xrange-0.1, nanmean(deltaJTHFT_scores), nanstd(deltaJTHFT_scores),'-o','Color',blue_color,'MarkerFaceColor',blue_color,'LineWidth',0.5,'MarkerSize',6);
% h_nihss = errorbar(xrange+0.1, nanmean(deltaNIHSS_scores), nanstd(deltaNIHSS_scores),'-s','Color',orange_color,'MarkerFaceColor',orange_color,'LineWidth',0.5,'MarkerSize',6);
ylim([-3 3]); 
xlim([-1 15]);
set(gca,'Ytick',[-4 -2 0 2 4],'YTickLabel',-4:2:4,'FontSize',paper_font_size-1);
set(gca,'Xtick',xrange,'Xticklabel',{'Baseline', '1wk-', '2wk-', '2mo-Post'},'FontSize',paper_font_size-1);
line([-1 15],[0 0],'LineStyle','--','Color','k')
title('JTHFT','FontSize',paper_font_size,'FontWeight','normal');
ylabel('Change from Baseline','FontSize',paper_font_size);
set(gca,'Xgrid','on');
set(gca,'Ygrid','on');
% legend([h_jthft h_nihss],{'JTHFT','NIHSS'},'Orientation','Vertical','Location','NorthEast')
% annotation('textbox',[0.9 0 0.1 0.07],'String','*','EdgeColor','none','FontSize',24); 
text(0.5,1.75,'Therapy','FontSize',paper_font_size - 1);


axes(Splot(4)); hold on;
jbfill(0:0.5:4,6*ones(1,9),-3.75*ones(1,9),[0.8, 0.8, 0.8],'none',1,transparency); 
xrange = [0, 4.5, 6.5, 13];
hold on;
h_grip = errorbar(xrange-0.1, nanmean(deltaGrip_scores), nanstd(deltaGrip_scores),'-o','Color',blue_color,'MarkerFaceColor',blue_color,'LineWidth',0.5,'MarkerSize',6);
h_pinch = errorbar(xrange+0.1, nanmean(deltaPinch_scores), nanstd(deltaPinch_scores),'-s','Color',orange_color,'MarkerFaceColor',orange_color,'LineWidth',0.5,'MarkerSize',6);
ylim([-4 8]); 
xlim([-1 15]);
set(gca,'Ytick',-4:2:8,'YTickLabel',-4:2:8,'FontSize',paper_font_size-1);
set(gca,'Xtick',xrange,'Xticklabel',{'Baseline', '1wk-', '2wk-', '2mo-Post'},'FontSize',paper_font_size-1);
line([-1 15],[0 0],'LineStyle','--','Color','k');
title('Grip & Pinch strengths','FontSize',paper_font_size,'FontWeight','normal');
% ylabel('Change from Baseline','FontSize',paper_font_size);
set(gca,'Xgrid','on');
set(gca,'Ygrid','on');
legend([h_grip h_pinch],{'Grip','Pinch'},'Orientation','Vertical','Location','NorthEast')
text(0.5,5.5,'Therapy','FontSize',paper_font_size - 1);
annotation('textbox',[0.775 0 0.25 0.05],'String','**\itp\rm < 0.001; *\itp\rm < 0.05','EdgeColor','none','FontSize',paper_font_size-2);    

%% Spider plot - using individual FMA & ARAT scores 

Label_cats = {{'Flexor';'synergy';'12'};...
              {'Extensor';'synergy';'6'};...
              {'Comb.';'synergies';'6'};...
              {'Out of';'synergy';'6'};...
              {'10';'Wrist'};...
              {'14';'Hand'};...
              {'6';'Speed/';'Co-od.'};...
              {'6';'Reflexes'}};          
Label_cats_pts_only = {'12'; '6'; '6'; '6'; '10'; '14'; '6'; '6'};
ARAT_Label_cats = {{'Grasp';'18'};...
              {'Grip';'12'};...
              {'Pinch';'18'};...
              {'Gross';'movement';'9'}}; 
ARAT_Label_cats_pts_only = {'18'; '12'; '18'; '9'};

Subjects_above_MCID = [1, 3, 5, 6, 7, 8, 9, 10];


for ind = 1:length(Subjects_above_MCID)
    figure('Position',[100 100 4*116 3*116], 'NumberTitle', 'off', 'Name', ['P' num2str(Subjects_above_MCID(ind))]); 
    Wplot = tight_subplot(1,2,0.15,0.1,0.1);

    FMA_scores = squeeze(FMA_cats(Subjects_above_MCID(ind),2:5,:));
    FMA_scores = [FMA_scores; FMA_cats_max];
%     FMA_scores(4,:) = FMA_scores(4,:) + 0.1
%     FMA_scores(5,:) = FMA_scores(5,:) + 0.1
    
    axes(Wplot(1)); hold on;
    spider_plot(FMA_scores,Label_cats_pts_only,FMA_cats_max, 5,1,0.5,0,'Marker', '.','MarkerSize',8, 'LineStyle', '-','LineWidth', 1);
    
    title(['P' num2str(Subjects_above_MCID(ind)), '              '],'FontSize',paper_font_size-2);  
    
    Subject_ARAT_scores = squeeze(ARAT_cats(Subjects_above_MCID(ind),:,:));
    Subject_ARAT_scores = [Subject_ARAT_scores; ARAT_cats_max];
%     Subject_ARAT_scores(4,:) = Subject_ARAT_scores(4,:) + 0.5;
%     Subject_ARAT_scores(5,:) = Subject_ARAT_scores(5,:) + 0.5;
    
    axes(Wplot(2)); hold on;
    spider_plot(Subject_ARAT_scores,ARAT_Label_cats_pts_only,ARAT_cats_max, ...
        5,1,0.5,0,'Marker', '.','MarkerSize',8, 'LineStyle', '-','LineWidth', 1);
%     title(['P' num2str(Subjects_above_MCID(ind))],'FontSize',paper_font_size-2);    

end

%% Overall FMA & ARAT by category - each column represents a category, each row a time point

% S9 = squeeze(FMA_cats(9,2:5,:));
% S9 = [S9; FMA_cats_max];
% 
% S10 = squeeze(FMA_cats(10,2:5,:));
% S10 = [S10; FMA_cats_max];
figure('Position',[500 200 6.5*116 3.5*116]);
Aplot = tight_subplot(1,2,0.1,[0.05 0.1], 0.05);

Avg_FMA_cats = squeeze(nanmean(FMA_cats(:,2:5,:),1));
std_FMA_cats = squeeze(nanstd(FMA_cats(:,2:5,:),1));
Avg_std_FMA_cats_upper = [Avg_FMA_cats+std_FMA_cats; FMA_cats_max];
Avg_std_FMA_cats_lower = [Avg_FMA_cats-std_FMA_cats; FMA_cats_max];
Avg_FMA_cats = [Avg_FMA_cats; FMA_cats_max];

axes(Aplot(1)); hold on;
hsp1 = spider_plot(Avg_FMA_cats,Label_cats,FMA_cats_max, 5,1,0.5,0,...
    'Marker', '.','MarkerSize',12,'LineStyle', '-','LineWidth', 1);
spider_plot(Avg_std_FMA_cats_upper,Label_cats,FMA_cats_max, 5,1,0.5,1,...
    'Marker', '.','MarkerSize',6,'LineStyle', 'none','LineWidth', 1);
spider_plot(Avg_std_FMA_cats_lower,Label_cats,FMA_cats_max, 5,1,0.5,1,...
    'Marker', '.','MarkerSize',6,'LineStyle', 'none','LineWidth', 1);
% title('FMA-UE','FontSize',paper_font_size);
%legend(hsp(1:4),{'Baseline', 'Post-1wk.', 'Post-2wk.', 'Post-2mo.'},'FontSize',10)

Avg_ARAT_cats = squeeze(nanmean(ARAT_cats,1));
std_ARAT_cats = squeeze(nanstd(ARAT_cats,1));
Avg_std_ARAT_cats_upper = [Avg_ARAT_cats+std_ARAT_cats; ARAT_cats_max];
Avg_std_ARAT_cats_lower = [Avg_ARAT_cats-std_ARAT_cats; ARAT_cats_max];
Avg_ARAT_cats = [Avg_ARAT_cats; ARAT_cats_max];

axes(Aplot(2)); hold on;
hsp = spider_plot(Avg_ARAT_cats,ARAT_Label_cats,ARAT_cats_max,5,1,0.5,0,...
    'Marker', '.','MarkerSize',12,'LineStyle', '-','LineWidth', 1);
spider_plot(Avg_std_ARAT_cats_upper,ARAT_Label_cats,ARAT_cats_max,5,1,0.5,1,...
    'Marker', '.','MarkerSize',6,'LineStyle', 'none','LineWidth', 1);
spider_plot(Avg_std_ARAT_cats_lower,ARAT_Label_cats,ARAT_cats_max,5,1,0.5,1,...
    'Marker', '.','MarkerSize',6,'LineStyle', 'none','LineWidth', 1);
% title('ARAT','FontSize',paper_font_size);
legend(hsp(1:4),{'Baseline', '1wk-Post', '2wk-Post', '2mo-Post'},'FontSize',paper_font_size, 'Location','southeastoutside');

%% Adjust scores - use last value carry over for missing values
% % %FMA_cats is already generated below % categories or cats
% % %FMA_mod_cats(7,4,:) = FMA_cats(7,4,:); % use last value carry over for missing values
% % FMA_cats_change_from_baseline2 = zeros(size(FMA_cats));
% % for type = 1:8
% %     FMA_cats_change_from_baseline2(:,:,type) = FMA_cats(:,:,type) - repmat(FMA_cats(:,2,type),1,size(FMA_cats,2));
% % end
% % %Total_FMA = sum(FMA_cats,3);
% % %Total_FMA_change_from_baseline2 = Total_FMA - repmat(Total_FMA(:,2),1,5); % must be equal to sum(FMA_mod_cats_change_from_baseline2,3)
% % FMA_cats_max = [12 6 6 6 10 14 6 6];
%% Spider plot example
% Point properties
num_of_points = 6;
row_of_points = 4;

% Random data
P = rand(row_of_points, num_of_points);

% Scale points by a factor
P(:, 2) = P(:, 2) * 2;
P(:, 3) = P(:, 3) * 3;
P(:, 4) = P(:, 4) * 4;
P(:, 5) = P(:, 5) * 5;

% Make random values negative
P(1:3, 3) = P(1:3, 3) * -1;
P(:, 5) = P(:, 5) * -1;

% Create generic labels
P_labels = cell(num_of_points, 1);

for ii = 1:num_of_points
    P_labels{ii} = sprintf('Label %i', ii);
end

% Figure properties
figure('units', 'normalized', 'outerposition', [0 0.05 1 0.95]);

% Axes properties
axes_interval = 2;
axes_precision = 1;

spider_plot(P(1,:), P_labels, axes_interval, axes_precision,'Marker', 'o',...
    'LineStyle', '-',...
    'LineWidth', 2,...
    'MarkerSize', 5);

