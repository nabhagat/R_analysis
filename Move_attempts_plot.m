%% Plot number of attempts plot
clear
Subject_name = 'BNBO';
Session_nos = [3:5];
convert_datasets = 0;
%Block_nos = [2 3 4 5 6 7 8 9];                      % JF

% Block_nos = [5 6 7 8 0 0 0 0 0;                 %PLSH
%                       1 2 3 4 5 7 8 9 0;                
%                       3 4 5 6 7 8 9 10 11];

% Block_nos = [4 5 6 0 0 0 0 0 0;
%                       1 2 3 5 6 7 8 9 0;                % LSGR
%                       2 3 4 5 7 8 9 10 11];

% Block_nos = [3 4 5 6 7 8 9 10;                % ERWS
%                       4 5 6 7 8 0 0 0];

Block_nos = [5 6 9 0 0 0 0 0;              % BNBO
                      2 3 4 5 6 7 8 0;
                      2 3 4 5 6 7 8 9];

%% Find Move attempts
% failed_attempts_per_block = [];
% total_attempts_per_block = [];
% 
%  for ses_id = 1:length(Session_nos)
%      for block_id = 1: size(Block_nos,2)
%          Sess_num = Session_nos(ses_id);
%          Block_num = Block_nos(ses_id,block_id);
%          if Block_num == 0
%              continue
%          else
%              folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; 
%              block_stats = dlmread([folder_path Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_block_statistics.csv'],',',1,1); 
%              num_failed_attempts = block_stats(logical(block_stats(:,4)) ,3) - block_stats(logical(block_stats(:,4)),4);
%              failed_attempts_per_block = [failed_attempts_per_block; [sum(num_failed_attempts) Sess_num Block_num]];
%              total_attempts_per_block = [total_attempts_per_block; [sum(block_stats(:,3)) Sess_num Block_num]]; 
%              failed_attempts_per_block((failed_attempts_per_block(:,1) < 0),1) = 0;
%          end
%          
%      end
%  end
% 
%  save(['C:\NRI_BMI_Mahi_Project_files\R_analysis\' Subject_name '_failed_attempts_per_block.mat'], 'failed_attempts_per_block');
%  

%% load Brain Vision closedloop files and extract EEG, EMG signals
% eeglab;
% if convert_datasets == 1
%      for ses_id = 1:length(Session_nos)
%          for block_id = 1: size(Block_nos,2)
%              Sess_num = Session_nos(ses_id);
%              Block_num = Block_nos(ses_id,block_id);
%               if Block_num == 0
%                  continue
%              else
%                 folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; 
%                 EEG = pop_loadbv(folder_path, [Subject_name '_ses' num2str(Sess_num) '_closeloop000' num2str(Block_num) '.vhdr'], [], [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64]);
%                 EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_closeloop_block' num2str(Block_num) '_eeg_raw'];
%                 EEG = eeg_checkset( EEG );
%                 EEG=pop_chanedit(EEG, 'lookup','C:\\Program Files\\MATLAB\\R2013a\\toolbox\\eeglab\\plugins\\dipfit2.2\\standard_BESA\\standard-10-5-cap385.elp');
%                 EEG = eeg_checkset( EEG );
%                 EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_closeloop_block' num2str(Block_num) '_eeg_raw.set'],'filepath',folder_path);
%                 EEG = eeg_checkset( EEG );
%                 % Update EEGLAB window
%                     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%                 %    eeglab redraw;
% 
%                 EEG = pop_select( EEG,'channel',{'TP9' 'TP10' 'FT9' 'FT10'});
%                 EEG.setname=[Subject_name '_ses' num2str(Sess_num) '_closeloop_block' num2str(Block_num) '_emg_raw'];;
%                 EEG = eeg_checkset( EEG );
%                 %EEG = pop_saveset( EEG, 'filename',['BNBO_ses2_cond1_block' num2str(Block_num) '_emg_raw.set'],'filepath','C:\\NRI_BMI_Mahi_Project_files\\All_Subjects\\Subject_BNBO\\BNBO_Session2\\');
%                 EEG = pop_saveset( EEG, 'filename',[Subject_name '_ses' num2str(Sess_num) '_closeloop_block' num2str(Block_num) '_emg_raw.set'],'filepath',folder_path);
%                 EEG = eeg_checkset( EEG );
%                 % Update EEGLAB window
%                     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%                 %    eeglab redraw;
% 
%               end
%          end
%      end
% end
% eeglab redraw;
 
 %% Find EEG/EMG TPR/FPR
EEG_TPR_per_block = [];
EEG_FPR_per_block = [];
EEG_EMG_TPR_per_block = [];
EEG_EMG_FPR_per_block = [];


 for ses_id = 1:length(Session_nos)
     for block_id = 1: size(Block_nos,2)
         Sess_num = Session_nos(ses_id);
         Block_num = Block_nos(ses_id,block_id);
         if Block_num == 0
             continue
         else
             folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; 
             new_block_stats = dlmread([folder_path Subject_name '_ses' num2str(Sess_num) '_block' num2str(Block_num) '_new_block_stats.csv'],',',1,1); 
             valid_or_catch_flag = new_block_stats(:,3);
             EEG_decisions = new_block_stats(:,4);
             EEG_EMG_decisions = new_block_stats(:,5);
             
             EEG_TPR_per_block = [EEG_TPR_per_block; sum(EEG_decisions(valid_or_catch_flag == 1))/length(find(valid_or_catch_flag== 1))];
             EEG_FPR_per_block = [EEG_FPR_per_block; sum(EEG_decisions(valid_or_catch_flag == 2))/length(find(valid_or_catch_flag== 2))];
             EEG_EMG_TPR_per_block = [EEG_EMG_TPR_per_block; sum(EEG_EMG_decisions(valid_or_catch_flag == 1))/length(find(valid_or_catch_flag== 1))];
             EEG_EMG_FPR_per_block = [EEG_EMG_FPR_per_block; sum(EEG_EMG_decisions(valid_or_catch_flag == 2))/length(find(valid_or_catch_flag== 2))];
             
         end
         
     end
 end

 %% Plot results
figure; 
subplot(1,2,1);
barwitherr(std([100.*EEG_TPR_per_block 100.*EEG_EMG_TPR_per_block]),mean([100.*EEG_TPR_per_block 100.*EEG_EMG_TPR_per_block]),0.5);
axis([0.25 2.5 0 100]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'EEG','EEG+EMG'},'FontWeight','bold');
xlabel('','FontSize',12);
ylabel('% TPR','FontSize',12);

subplot(1,2,2)
barwitherr(std([100.*EEG_FPR_per_block 100.*EEG_EMG_FPR_per_block]),mean([100.*EEG_FPR_per_block 100.*EEG_EMG_FPR_per_block]),0.5);
axis([0.25 2.5 0 100]);
set(gca,'XTick',[1 2]);
set(gca,'XTickLabel',{'EEG','EEG+EMG'},'FontWeight','bold');
xlabel('','FontSize',12);
ylabel('% FPR','FontSize',12);


title('Subject BNBO','FontSize',12);

 
 