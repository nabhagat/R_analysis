clc
clear

figure('Position',[0 100 8*116 4*116]);
S_plot = tight_subplot(1,2,[0.1],[0.2 0.3],[0.15 0.1]);

%Data for Plotting
sesh   = [1 2 3 4 5];                                                       %X axis sesion number
MAPR10 = [65.5954	59.5782	72.0848	76.1909	71.3078];                       %MAPR at 10% threshold
MAPRSD = [8.692004655	6.425474665	5.501923072	6.23937952	9.46023414];    %MAPR Standard deviation for the session
msmf   = [0.8484	0.9031	0.8663	0.8148	0.8571];                        %Smoothness Correlation Coefficient
msmfsd = [0.117011377	0.099685239	0.06019785	0.104703521	0.122212956];   %Standard Deviation of for smoothness... 

%MAPR Plot
axes(S_plot(1));
errorbar(sesh,MAPR10,MAPRSD,'ks','MarkerFaceColor','k','MarkerSize',10); 
hold on
scatter(sesh,MAPR10,'ks','filled');
set(gca,'xlim',[0 6],...
    'xtick',[1 2 3 4 5],...
    'ylim',[0 100],...
    'ytick',[0 50 100],...
    'fontsize',18-2);
mtitle = sprintf('Mean Arrest Period Ratio'); % \n (Average +/- S.D)');
title(mtitle,'fontsize',18);
set(get(gca,'YLabel'),'Rotation',0)
ylabel('BNBO','fontsize',18)
set(get(gca,'ylabel'),'units','normalize','Position',[-.2 .5]);
xlabel('Session','fontsize',18)
hold off

%MSM Factor plot
axes(S_plot(2));
errorbar(sesh,msmf,msmfsd,'ks','MarkerFaceColor','k','MarkerSize',10);
hold on
scatter(sesh,msmf,'ks','filled');
set(gca,'xlim',[0 6],...
    'xtick',[1 2 3 4 5],...
    'ylim',[0 1],...
    'ytick',[0 0.5 1],...
    'fontsize',18-2);
ptitle = sprintf('Smoothness Correlation Coefficient'); % \n (Average +/- S.D)');
title(ptitle,'fontsize',18)
xlabel('Session','fontsize',18)
hold off

