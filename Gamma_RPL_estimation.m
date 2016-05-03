% Program to evaluate Relative Power Level (RPL) and correlate with BMI performance across all subjects and sessions of clinical study - NRI Project
% Author: Nikunj A. Bhagat, Graduate Student, University of Houston, 
% Contact: nbhagat08[at]gmail.com
% Date: February 18, 2016
%------------------------------------------------------------------------------------------------------------------------
% Revisions
%                    
%                    
%------------------------------------------------------------------------------------------------------------------------
clear all; 
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab; % start EEGLAB from Matlab 
 
EEG_channel_nos = 1:64;
EMG_channel_nos = [17 22 41 42 45 46 51 55];
EEG_channel_nos(EMG_channel_nos) = [ ];

% Subject Details 
Subject_name = 'S9009'; % change1
Sess_num = '2';  % For calibration and classifier model             
closeloop_Sess_num = '15';     
Cond_num = 3;  % 1 - Active/User-driven; 2 - Passive; 3 - Triggered/User-triggered; 4 - Observation 
Block_num = 150;

%folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
closeloop_folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(closeloop_Sess_num) '\']; % change3

% Flags to control the processing 
readbv_files = 1;
blocks_nos_to_import = 8;
segment_eeg_baseline = 1;

%% Import raw BrainVision files (.eeg, .vhdr, .vmrk)  to EEGLAB dataset
if readbv_files == 1
    %total_no_of_trials = 0;
    EEG_dataset_to_merge = [];
    EMG_dataset_to_merge = [];
    
    for block_index = 1:length(blocks_nos_to_import)

%     Only for subject S9009
%         if block_index == 1 || block_index == 2
%             Cond_num = 1;
%         else
%             Cond_num = 3;
%         end
%     **********Only for subject S9009

        fprintf('\nImporting block # %d of %d blocks...\n',blocks_nos_to_import(block_index),length(blocks_nos_to_import));
        
        %total_no_of_trials = total_no_of_trials + 20;
       if blocks_nos_to_import(block_index) > 9
            EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' closeloop_Sess_num '_closeloop_block00' num2str(blocks_nos_to_import(block_index)) '.vhdr'], [], 1:64);
       else
            EEG = pop_loadbv(closeloop_folder_path, [Subject_name '_ses' closeloop_Sess_num '_closeloop_block000' num2str(blocks_nos_to_import(block_index)) '.vhdr'], [], 1:64);
       end

        EEG.setname=[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_eeg_raw'];
        EEG = eeg_checkset( EEG );
        
        % Swap channel data to obtain correct electrode representation - 9/2/2015
       if (strcmp(Subject_name,'S9007') && strcmp(Sess_num,'1'))
           temp_EEG = EEG; 
%            incorrect_nos = [57:64];
%            correct_nos = [42 41 51 17 45 46 55 22];
%            correct_labels = {'FT7','FT9','TP7','TP9','FT8','FT10','TP8','TP10'};
%            correct_labels_for_incorrect_nos = {'P1','P2','P6','PO7','PO3','POz','PO4','PO8'};
%            for i = 1:length(incorrect_nos)
%                EEG.data(correct_nos(i),:) = temp_EEG_data(incorrect_nos(i),:);
%                EEG.data(incorrect_nos(i),:) = temp_EEG_data(correct_nos(i),:);
%                EEG.chanlocs(incorrect_nos(i)).labels = correct_labels{i};
%            end
           EEG.data(1:16,:) = temp_EEG.data(1:16,:);
           for j = 1:16
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j).labels;
           end
           EEG.data(17,:) = temp_EEG.data(60,:);
           EEG.chanlocs(17).labels = 'TP9';
           EEG.data(18:21,:) = temp_EEG.data(17:20,:);
           for j = 18:21
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-1).labels;
           end
           EEG.data(22,:) = temp_EEG.data(64,:);
           EEG.chanlocs(22).labels = 'TP10';
           EEG.data(23:40,:) = temp_EEG.data(21:38,:);
           for j = 23:40
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-2).labels;
           end
           EEG.data(41,:) = temp_EEG.data(58,:);
           EEG.chanlocs(41).labels = 'FT9';
           EEG.data(42,:) = temp_EEG.data(57,:);
           EEG.chanlocs(42).labels = 'FT7';
           EEG.data(43:44,:) = temp_EEG.data(39:40,:);
           for j = 43:44
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-4).labels;
           end
           EEG.data(45,:) = temp_EEG.data(61,:);
           EEG.chanlocs(45).labels = 'FT8';
           EEG.data(46,:) = temp_EEG.data(62,:);
           EEG.chanlocs(46).labels = 'FT10';
           EEG.data(47:50,:) = temp_EEG.data(41:44,:);
           for j = 47:50
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-6).labels;
           end
           EEG.data(51,:) = temp_EEG.data(59,:);
           EEG.chanlocs(51).labels = 'TP7';
           EEG.data(52:54,:) = temp_EEG.data(45:47,:);
           for j = 52:54
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-7).labels;
           end
           EEG.data(55,:) = temp_EEG.data(63,:);
           EEG.chanlocs(55).labels = 'TP8';
           EEG.data(56:64,:) = temp_EEG.data(48:56,:);
           for j = 56:64
                EEG.chanlocs(j).labels = temp_EEG.chanlocs(j-8).labels;
           end
           
       end           
       
        EEG=pop_chanedit(EEG, 'lookup','C:\NRI_BMI_Mahi_Project_files\EEGLAB_13_1_1b\eeglab13_1_1b\plugins\dipfit2.2\standard_BESA\standard-10-5-cap385.elp');
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_eeg_raw.set'],...
            'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
        % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        %    eeglab redraw;
        EEG_dataset_to_merge = [EEG_dataset_to_merge CURRENTSET];

        EEG = pop_select( EEG,'channel',EMG_channel_nos);
        EEG.setname=[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_emg_raw'];
        EEG = eeg_checkset( EEG );
        EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import(block_index)) '_emg_raw.set'],...
            'filepath',closeloop_folder_path);
        EEG = eeg_checkset( EEG );
        % Update EEGLAB window
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        %    eeglab redraw;        
        EMG_dataset_to_merge = [EMG_dataset_to_merge CURRENTSET];
    end
        
% %     if ~isempty(EEG_dataset_to_merge)
% %         fprintf('All block have been imported. Merging %d blocks...\n', length(EEG_dataset_to_merge));
% %         if length(EEG_dataset_to_merge) == 1
% %             % Retrive old data set - No need to merge 
% %             [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EEG_dataset_to_merge,'study',0); 
% %         else
% %             EEG = pop_mergeset( ALLEEG,EEG_dataset_to_merge, 0);
% %         end
% %         EEG.setname=[Subject_name '_ses' Sess_num '_cond' num2str(Cond_num) '_block' num2str(total_no_of_trials) '_eeg_raw'];
% %         EEG = eeg_checkset( EEG );
% %         EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(total_no_of_trials) '_eeg_raw.set'],'filepath',folder_path);
% %         EEG = eeg_checkset( EEG );
% %     else
% %         errordlg('Error: EEG blocks could not be merged');
% %     end
% %     % Update EEGLAB window
% %         [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
% %     %    eeglab redraw;
% % 
% %     if ~isempty(EMG_dataset_to_merge)
% %         fprintf('All block have been imported. Merging %d blocks...\n', length(EMG_dataset_to_merge));
% %         if length(EMG_dataset_to_merge) == 1
% %             % Retrive old data set - No need to merge 
% %             [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG,CURRENTSET,'retrieve',EMG_dataset_to_merge,'study',0); 
% %         else
% %             EEG = pop_mergeset( ALLEEG,EMG_dataset_to_merge, 0);
% %         end
% %         EEG.setname=[Subject_name '_ses' Sess_num '_cond' num2str(Cond_num) '_block' num2str(total_no_of_trials) '_emg_raw'];
% %         EEG = eeg_checkset( EEG );
% %         EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(total_no_of_trials) '_emg_raw.set'],'filepath',folder_path);
% %         EEG = eeg_checkset( EEG );
% %     else
% %         errordlg('Error: EMG blocks could not be merged');
% %     end
        
    % Update EEGLAB window
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    %eeglab redraw;
end

%% Segment raw EEG to determine baseline (30s)
if segment_eeg_baseline == 1
    EEG = pop_loadset( [Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import) '_eeg_raw.set'], closeloop_folder_path); 
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;
    
    % Define Paramteres
    raw_eeg_Fs = EEG.srate;
    
    
    block_start_trigger_latency = [];
    for j=1:length(EEG.event)-1
        if (strcmp(EEG.event(j).type,'S 42'))
            block_start_trigger_latency = [block_start_trigger_latency; EEG.event(j).latency/raw_eeg_Fs];
            break;
        end
    end
    if isempty(block_start_trigger_latency)
        errordlg('Trial start triggere "S 42" not found');
    end
    
    % Plot pre-trial start EEG data and 'manually' select interval
    eegplot(EEG.data(EEG_channel_nos,:), 'srate', EEG.srate, 'spacing', 200, 'eloc_file', EEG.chanlocs, 'limits', [EEG.xmin*1000 block_start_trigger_latency],...
        'winlength', 30, 'title', 'EEG channel activities using eegplot()'); 
    baseline_start = input('Enter start time for 30 sec baseline segment in seconds: ');
    baseline_stop = baseline_start + 20.00;
    EEG = pop_select(EEG,'time',[baseline_start baseline_stop]);
    EEG.setname=[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import) '_eeg_baseline'];
    EEG = eeg_checkset( EEG );
    EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import) '_eeg_baseline.set'],...
        'filepath',closeloop_folder_path);
    EEG = eeg_checkset( EEG );
    
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
    eeglab redraw;
end

%% Filter baseline EEG segment and compute RPL

EEG = pop_loadset( [Subject_name '_ses' closeloop_Sess_num  '_closeloop_block' num2str(blocks_nos_to_import) '_eeg_baseline.set'], closeloop_folder_path); 
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
eeglab redraw;

%% 
% Define Paramteres
raw_eeg_Fs = EEG.srate;
raw_eeg = EEG.data;
[eeg_nbchns,eeg_pnts] = size(EEG.data);

% 60 Hz - notch filter
wo = 60/(raw_eeg_Fs/2);
quality_factor = 35; 
[b_notch,a_notch] = iirnotch(wo,wo/quality_factor); % iirnotch(wo,bw); wo = 60/(Fs/2);  bw = wo/35;
%fvtool(b_notch,a_notch,'Analysis','freq','Fs',raw_eeg_Fs)
notch_filtered_eeg = zeros(size(raw_eeg));
for i = 1:eeg_nbchns
    notch_filtered_eeg(i,:) = filtfilt(b_notch,a_notch,double(raw_eeg(i,:))); % filtering with zero-phase delay
end
%EEG.data = notch_filtered_eeg;

% Band pass filter between [0.1 - 100 Hz]
[b_bpf,a_bpf] = butter(4,([0.1 100]./(raw_eeg_Fs/2)),'bandpass');
BPFred_eeg = zeros(size(raw_eeg));
%fvtool(b_bpf,a_bpf,'Analysis','freq','Fs',raw_eeg_Fs)
for i = 1:eeg_nbchns
    BPFred_eeg(i,:) = filtfilt(b_bpf,a_bpf,notch_filtered_eeg(i,:)); % filtering with zero-phase delay
end

% EEG.data = BPFred_eeg;
% [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
% eeglab redraw;
% figure;
% [spectra,freqs,speccomp,contrib,specstd] = spectopo(EEG.data,0,raw_eeg_Fs,'freq',[10 30 70],...
%                                                                                 'chanlocs',EEG.chanlocs, 'nfft',512,'freqrange',[0.1 100], 'plotchans',EEG_channel_nos,...
%                                                                                 'winsize', 512, 'overlap',256);
% Equivalent to  [pxx,freqs] = pwelch(EEG.data(1,:),512,256,512,raw_eeg_Fs); figure; plot(freqs, 10*log10(pxx)) - % detrend makes no difference since band pass filtered                                                                           

% Compute power spectrum density using pwelch, default : window length is length(data)/8, overlap = 50% - preferred for reducing spectral leakage
    NFFT = 512; % Freq_Res = Fs/NFFT - This is true !!
    PSD_eeg = zeros(eeg_nbchns,((NFFT/2)+1));
    PSD_f = zeros(1,((NFFT/2)+1));      % range = (-fs/2:0:fs/2). But PSD is real and symmetric. Hence only +ve frequencies plotted
    for psd_len = 1:eeg_nbchns
        [PSD_eeg(psd_len,:),PSD_f] = pwelch(BPFred_eeg(psd_len,:),[],[],NFFT,raw_eeg_Fs);
    end
    figure; hold on; grid on;
    plot(PSD_f,10*log10(PSD_eeg(EEG_channel_nos,:)))
    xlabel('Frequency (Hz)');
    ylabel('PSD (dB)');
    title('EEG Baseline PSD');
    hold off

% Compute power in different bands 
delta_range = [0.1 4];
theta_range = [4 8];
alpha_range = [8 13];
beta_range = [13 30];
gamma_range1 = [30 70];
gamma_range2 = [55 85];
tot_range1 = [0.1 70];
%tot_range2 = [0.1 85];

%delta_power = trapz(PSD_f(PSD_f>=delta_range(1) & PSD_f<delta_range(2)),10*log10(PSD_eeg(:,PSD_f>=delta_range(1) & PSD_f<delta_range(2))'))';
delta_power = trapz(PSD_f(PSD_f>=delta_range(1) & PSD_f<delta_range(2)),(PSD_eeg(:,PSD_f>=delta_range(1) & PSD_f<delta_range(2))'))';
%theta_power = trapz(PSD_f(PSD_f>=theta_range(1) & PSD_f<theta_range(2)),10*log10(PSD_eeg(:,PSD_f>=theta_range(1) & PSD_f<theta_range(2))'))';
theta_power = trapz(PSD_f(PSD_f>=theta_range(1) & PSD_f<theta_range(2)),(PSD_eeg(:,PSD_f>=theta_range(1) & PSD_f<theta_range(2))'))';
%alpha_power = trapz(PSD_f(PSD_f>=alpha_range(1) & PSD_f<alpha_range(2)),10*log10(PSD_eeg(:,PSD_f>=alpha_range(1) & PSD_f<alpha_range(2))'))';
alpha_power = trapz(PSD_f(PSD_f>=alpha_range(1) & PSD_f<alpha_range(2)),(PSD_eeg(:,PSD_f>=alpha_range(1) & PSD_f<alpha_range(2))'))';
%beta_power = trapz(PSD_f(PSD_f>=beta_range(1) & PSD_f<beta_range(2)),10*log10(PSD_eeg(:,PSD_f>=beta_range(1) & PSD_f<beta_range(2))'))';
beta_power = trapz(PSD_f(PSD_f>=beta_range(1) & PSD_f<beta_range(2)),(PSD_eeg(:,PSD_f>=beta_range(1) & PSD_f<beta_range(2))'))';
%gamma_power1 = trapz(PSD_f(PSD_f>=gamma1_range(1) & PSD_f<gamma1_range(2)),10*log10(PSD_eeg(:,PSD_f>=gamma1_range(1) & PSD_f<gamma1_range(2))'))';
gamma_power1 = trapz(PSD_f(PSD_f>=gamma_range1(1) & PSD_f<gamma_range1(2)),(PSD_eeg(:,PSD_f>=gamma_range1(1) & PSD_f<gamma_range1(2))'))';
%gamma_power2 = trapz(PSD_f(PSD_f>=gamma_range2(1) & PSD_f<gamma_range2(2)),10*log10(PSD_eeg(:,PSD_f>=gamma_range2(1) & PSD_f<gamma_range2(2))'))';
gamma_power2 = trapz(PSD_f(PSD_f>=gamma_range2(1) & PSD_f<gamma_range2(2)),(PSD_eeg(:,PSD_f>=gamma_range2(1) & PSD_f<gamma_range2(2))'))';
%tot_power1 = trapz(PSD_f(PSD_f>=tot_range1(1) & PSD_f<tot_range1(2)),10*log10(PSD_eeg(:,PSD_f>=tot_range1(1) & PSD_f<tot_range1(2))'))';
tot_power1 = trapz(PSD_f(PSD_f>=tot_range1(1) & PSD_f<tot_range1(2)),(PSD_eeg(:,PSD_f>=tot_range1(1) & PSD_f<tot_range1(2))'))';
%tot_power2 = trapz(PSD_f(PSD_f>=tot_range2(1) & PSD_f<tot_range2(2)),10*log10(PSD_eeg(:,PSD_f>=tot_range2(1) & PSD_f<tot_range2(2))'))';

total_band_power = tot_power1(EEG_channel_nos); %delta_power(EEG_channel_nos) + theta_power(EEG_channel_nos) + alpha_power(EEG_channel_nos) + beta_power(EEG_channel_nos) + gamma_power1(EEG_channel_nos);
delta_norm = delta_power(EEG_channel_nos)./total_band_power;
delta_RPL = delta_norm./sum(delta_norm);
theta_norm = theta_power(EEG_channel_nos)./total_band_power;
theta_RPL = theta_norm./sum(theta_norm);
alpha_norm = alpha_power(EEG_channel_nos)./total_band_power;
alpha_RPL = alpha_norm./sum(alpha_norm);
beta_norm = beta_power(EEG_channel_nos)./total_band_power;
beta_RPL = beta_norm./sum(beta_norm);
gamma_norm1 = gamma_power1(EEG_channel_nos)./total_band_power;  
gamma_RPL1 = gamma_norm1./sum(gamma_norm1);

% gamma_ratio2 = gamma_power2(EEG_channel_nos)./(delta_power(EEG_channel_nos) + theta_power(EEG_channel_nos) + alpha_power(EEG_channel_nos) + beta_power(EEG_channel_nos)...
%     + gamma_power2(EEG_channel_nos));  
% gamma_RPL2 = gamma_ratio2./sum(gamma_ratio2);

figure; plot(EEG_channel_nos,delta_RPL,'-ok'); 
hold on; plot(EEG_channel_nos,theta_RPL,'-og')
hold on; plot(EEG_channel_nos,alpha_RPL,'-or')
hold on; plot(EEG_channel_nos,beta_RPL,'-om')
hold on; plot(EEG_channel_nos,gamma_RPL1,'-ob')

%% Plot topographic images of EEG bands

% Channels_used = [4 5 6 9 10 13 14 15 19 20 24 25 26 32 ...
%                  38 39 43 44 48 49 52 53 54 57 58];
% emarker_chans = [];
% for ch_sel_cnt = 1:length(Channels_sel)
%     emarker_chans(ch_sel_cnt) = find(Channels_used == Channels_sel(ch_sel_cnt));
% end

freq_intervals = {'\delta', '\theta', '\alpha', '\beta', '\gamma'};
figure('Position',[700 100 7*116 3*116]); 
Scalp_plot = tight_subplot(1,length(freq_intervals),[0.02 0.02],[0.05 0.1],[0.05 0.05]);
ScalpData = [delta_RPL, theta_RPL, alpha_RPL, beta_RPL, gamma_RPL1];

for tpt = 1:length(freq_intervals)
    axes(Scalp_plot(tpt));
   topoplot(ScalpData(:,tpt), EEG.chanlocs(EEG_channel_nos),'maplimits', [0 0.03],'style','both',...    
        'electrodes','on','plotchans',1:56,'plotrad', 0.55,...
        'gridscale', 300, 'drawaxis', 'off', 'whitebk', 'off',...
        'conv','off');
        %'emarker2',{emarker_chans,'.','k',16,2});
    %topoplot(ScalpData(:,tpt),EEG.chanlocs(EEG_channel_nos),'plotrad', 0.55);
    title(freq_intervals(tpt),'FontSize',18);
    %axis([-0.55 0.55 -1.82 1.82]);
end
%colorbar('Position',[0.96 0.35 0.015 0.35]);
cbar_axes = colorbar('location','SouthOutside','XTick',[0 0.03],'XTickLabel',{'0','0.03'});
%set(cbar_axes,'Position',[0.75 0.28 0.2 0.05]);
xlabel(cbar_axes,'Bandwise RPL','FontSize',12);
%export_fig MS_ses1_cond3_Scalp_maps '-png' '-transparent';



                                                                            
