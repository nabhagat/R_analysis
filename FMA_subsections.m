% Program to analyze changes in FM-UE by subsections

% NRI Project
% Created by Nikunj Bhagat on 11-26-2018
clc;
clear all;
blue = [0, 153, 255]./255; 
light_blue = [102 204 255]./255; % #66ccff
orange = [228, 108, 10]./255;
light_orange = [255, 179, 102]./255; % #ffb366

%% Adjust scores - use last value carry over for missing values
%FMA_cats is already generated below % categories or cats
%FMA_mod_cats(7,4,:) = FMA_cats(7,4,:); % use last value carry over for missing values
FMA_cats_change_from_baseline2 = zeros(size(FMA_cats));
for type = 1:8
    FMA_cats_change_from_baseline2(:,:,type) = FMA_cats(:,:,type) - repmat(FMA_cats(:,2,type),1,size(FMA_cats,2));
end
%Total_FMA = sum(FMA_cats,3);
%Total_FMA_change_from_baseline2 = Total_FMA - repmat(Total_FMA(:,2),1,5); % must be equal to sum(FMA_mod_cats_change_from_baseline2,3)
FMA_cats_max = [12 6 6 6 10 14 6 6];
%% Spider plot - using it on the actual FMA scores and not the change from baseline
Label_cats = {{'Flexor';'synergy';'12'};...
              {'Extensor';'synergy';'6'};...
              {'Comb.';'synergies';'6'};...
              {'Out of';'synergy';'6'};...
              {'10';'Wrist'};...
              {'14';'Hand'};...
              {'6';'Speed/';'Co-od.'};...
              {'6';'Reflexes'}};          
Label_cats_pts_only = {'12'; '6'; '6'; '6'; '10'; '14'; '6'; '6'};
Subjects_FMA_above_MCID = [1, 3, 5, 7, 8, 9, 10];

figure('Position',[0 0 8*116 5*116]); 
Wplot = tight_subplot(2,4,0.05,0.05,0.05);

for ind = 1:length(Subjects_FMA_above_MCID)
    FMA_scores = squeeze(FMA_cats(Subjects_FMA_above_MCID(ind),2:5,:));
    FMA_scores = [FMA_scores; FMA_cats_max];
    
    axes(Wplot(ind)); hold on;
    spider_plot(FMA_scores,Label_cats_pts_only,10,1,0.5,'Marker', '.','MarkerSize',10,...
        'LineStyle', '-','LineWidth', 1.5);
    title(['S' num2str(Subjects_FMA_above_MCID(ind))],'FontSize',12);    
end


%% Overall FMA by category

% S9 = squeeze(FMA_cats(9,2:5,:));
% S9 = [S9; FMA_cats_max];
% 
% S10 = squeeze(FMA_cats(10,2:5,:));
% S10 = [S10; FMA_cats_max];

Avg_FMA_cats = squeeze(nanmean(FMA_cats(:,2:5,:),1));
std_FMA_cats = squeeze(nanstd(FMA_cats(:,2:5,:),1));
Avg_std_FMA_cats = [Avg_FMA_cats-std_FMA_cats; FMA_cats_max];
Avg_FMA_cats = [Avg_FMA_cats; FMA_cats_max];
% figure('Position',[500 200 4*116 4*116]);
hsp = spider_plot(Avg_FMA_cats,Label_cats,10,1,0.5,0,...
    'Marker', '.','MarkerSize',12,...
    'LineStyle', '-','LineWidth', 1.5);
% title('S10','FontSize',12);
legend(hsp(1:4),{'Baseline', 'Post-1wk.', 'Post-2wk.', 'Post-2mo.'},'FontSize',10)
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
%% Plot average change from baseline for Fugl Meyer and ARAT
ARAT_scores = [43	55	53	55;
4	4	6	5;
45	54	42	49;
4	4	nan	4;
39	44	nan	40;
30	36	38	42;
25	32	35	nan;
42	52	51	53;
9	10	nan	9;
12	14	18	16];

FMA_scores = sum(FMA_cats,3);

deltaARAT_scores = ARAT_scores - repmat(ARAT_scores(:,1),[1 4]);
deltaFMA_scores = FMA_scores - repmat(FMA_scores(:,2),[1 5]);

figure('Position',[1600 200 6*116 3*116]); 
Splot = tight_subplot(1,2,0.05,[0.15 0.1],[0.1 0.05]);

axes(Splot(1)); hold on;
xrange = 0:3;
h_fma = errorbar(xrange, nanmean(deltaFMA_scores(:,2:5)), nanstd(deltaFMA_scores(:,2:5)),'-sk','MarkerFaceColor','k','LineWidth',1,'MarkerSize',8);
ylim([-4 12]); 
xlim([-0.2 3.2]);
set(gca,'Ytick',[-2 0 2 4 6 8 10],'YTickLabel',-2:2:10,'FontSize',10);
set(gca,'Xtick',0:3,'Xticklabel',{'Baseline', 'Post-1wk.', 'Post-2wk.', 'Post-2mo.'},'FontSize',10);
%legend([h_arat h_fma],{'Action Research Arm Test','Fugl-Meyer Upper Ext.'},'Orientation','Vertical','Location','NorthEastOutside')
line([-0.5 4],[0 0],'LineStyle','--','Color','k')
title('Fugl-Meyer Upper Ext. (FM-UE)','FontSize',12,'FontWeight','normal');
% ylabel('Change from Baseline','FontSize',10);
set(gca,'Xgrid','on');
set(gca,'Ygrid','on');
%annotation('textbox',[0.9 0 0.1 0.07],'String','*','EdgeColor','none','FontSize',24); 
% sigstar({[0,1],[0,2],[0,3]},[0.05,0.001,0.05]); % Does not work with errorbars

axes(Splot(2)); hold on;
xrange = 0:3;
h_arat = errorbar(xrange, nanmean(deltaARAT_scores), nanstd(deltaARAT_scores),'-sk','MarkerFaceColor','w','LineWidth',1,'MarkerSize',10);
ylim([-4 14]); 
xlim([-0.2 3.2]);
set(gca,'Ytick',[-2 0 2 4 6 8 10 12],'YTickLabel',-2:2:12,'FontSize',10);
set(gca,'Xtick',0:3,'Xticklabel',{'Baseline', 'Post-1wk.', 'Post-2wk.', 'Post-2mo.'},'FontSize',10);
%legend([h_arat h_fma],{'Action Research Arm Test','Fugl-Meyer Upper Ext.'},'Orientation','Vertical','Location','NorthEastOutside')
line([-0.5 4],[0 0],'LineStyle','--','Color','k')
title('Action Research Arm Test (ARAT)','FontSize',12,'FontWeight','normal');
% ylabel('Change from Baseline','FontSize',10);
grid('on');
annotation('textbox',[0.75 0 0.25 0.07],'String','**\itp\rm < 0.001; *\itp\rm < 0.05','EdgeColor','none','FontSize',10);    
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

