% Subject Details
Subject_name = 'JF';
Sess_num = '2';
Cond_num = 3;  % 1 - Active; 2 - Passive; 3 - Triggered; 4 - Observation 
Block_num = 160;

folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_name '\' Subject_name '_Session' num2str(Sess_num) '\']; % change2
load([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_performance_optimized_causal.mat']);      % Always use causal for training classifier
                 
channels  = Performance.classchannels;
smart_window_length = Performance.smart_window_length;
smart_Cov_Mat = Performance.smart_Cov_Mat ;
smart_Mu_move = Performance.smart_Mu_move;

save([folder_path Subject_name '_ses' num2str(Sess_num) '_cond' num2str(Cond_num) '_block' num2str(Block_num) '_classifier_parameters.mat'],...
          'channels',...
          'smart_window_length',...
          'smart_Cov_Mat',...
          'smart_Mu_move');





% % Analyzing closed data
% clear;
% Subject_names = {'BNBO','PLSH','LSGR','ERWS','JF'};
% Sess_nums = 4;
% 
% subj_n = 1;
% ses_n = Sess_nums(1);
% 
% folder_path = ['C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_' Subject_names{subj_n} '\' Subject_names{subj_n} '_Session' num2str(ses_n) '\'];
% fileid = [folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'];
% cl_ses_data = dlmread([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_cloop_statistics.csv'],',',7,1); 
% unique_blocks = unique(cl_ses_data(:,1));
% 
% m = 1;
% block_n = unique_blocks(m);
% load([folder_path Subject_names{subj_n} '_ses' num2str(ses_n) '_block' num2str(block_n) '_closeloop_results.mat']);
% load('C:\NRI_BMI_Mahi_Project_files\All_Subjects\Subject_BNBO\BNBO_Session2\BNBO_ses2_cond1_block160_performance_optimized_causal.mat');
% 
% % Decide classifier with best accuracy
% [max_acc_val,max_acc_index] = max(Performance.eeg_accur); 
% Best_BMI_classifier = Performance.eeg_svm_model{max_acc_index};
% prob_threshold = min(all_cloop_prob_threshold);

    
    