function [ALLEEG, EEG, CURRENTSET, ALLCOM] = SegmentEEGRestData(Subject_name, Sess_num, block_no,readbv_files,remove_emg_channel_nos, read_file_location, ...
read_file_name, save_file_location, ALLEEG, EEG, CURRENTSET, ALLCOM)
            % Import raw BrainVision files (.eeg, .vhdr, .vmrk)  to EEGLAB dataset
            if readbv_files == 1
                EEG = pop_loadbv(read_file_location, [read_file_name '.vhdr'], [], 1:64);
                EEG.setname=[read_file_name '_eeg_raw'];
                EEG = eeg_checkset( EEG );

                EEG=pop_chanedit(EEG, 'lookup','C:\EEGLAB\plugins\dipfit2.3\standard_BESA\standard-10-5-cap385.elp');
                EEG = eeg_checkset( EEG );
                EEG = pop_saveset( EEG, 'filename',[read_file_name '_eeg_raw.set'],'filepath',read_file_location);
                EEG = eeg_checkset( EEG );
                % Update EEGLAB window
                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
                %    eeglab redraw;
            end
            

            %% Segment raw EEG to determine baseline (30s)
                EEG = pop_loadset( [read_file_name '_eeg_raw.set'], read_file_location); 
                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                eeglab redraw;

                % Define Paramteres
                raw_eeg_Fs = EEG.srate;      
                raw_eeg = EEG.data;
                [eeg_nbchns,eeg_pnts] = size(EEG.data);

        % %         % 60 Hz - notch filter
        % %         wo = 60/(raw_eeg_Fs/2);
        % %         quality_factor = 35; 
        % %         [b_notch,a_notch] = iirnotch(wo,wo/quality_factor); % iirnotch(wo,bw); wo = 60/(Fs/2);  bw = wo/35;
        % %         %fvtool(b_notch,a_notch,'Analysis','freq','Fs',raw_eeg_Fs)
        % %         notch_filtered_eeg = zeros(size(raw_eeg));
        % %         for i = 1:eeg_nbchns
        % %             notch_filtered_eeg(i,:) = filtfilt(b_notch,a_notch,double(raw_eeg(i,:))); % filtering with zero-phase delay
        % %         end
        % %         EEG.data = notch_filtered_eeg;

                block_start_trigger_latency = [];
                for j=1:length(EEG.event)-1
                    if (strcmp(EEG.event(j).type,'S 42'))% S 10 %change
                        block_start_trigger_latency = [block_start_trigger_latency; EEG.event(j).latency/raw_eeg_Fs];
                        baseline_start = EEG.xmin*1000; 
                        baseline_stop = block_start_trigger_latency - 15;  % 15 seconds when getting ready to start robot 
                        
                        break;
                    end
                end
                if isempty(block_start_trigger_latency) || ((baseline_stop - baseline_start) >= 200) % trigger not found or greater than 200 sec. 
                    errordlg('Trial start trigger "S 42" not found');
                    eeglab redraw;
                    baseline_start = input('Enter start time for 30 sec baseline segment in seconds: ');
                    baseline_stop = baseline_start + 30.00;
                end

                % Plot pre-trial start EEG data and 'manually' select interval
        %         eegplot(EEG.data(EEG_channel_nos,:), 'srate', EEG.srate, 'spacing', 200, 'eloc_file', EEG.chanlocs, 'limits', [EEG.xmin*1000 block_start_trigger_latency],...
        %             'winlength', 30, 'title', 'EEG channel activities using eegplot()'); 

                EEG = pop_select(EEG,'time',[baseline_start baseline_stop], 'nochannel', remove_emg_channel_nos);
                EEG.setname=[Subject_name '_session' num2str(Sess_num)  '_block' num2str(block_no) '_resting_eeg'];
                EEG = eeg_checkset( EEG );
                EEG = pop_saveset( EEG, 'filename',[Subject_name '_session' num2str(Sess_num)  '_block' num2str(block_no) '_resting_eeg.set'],...
                    'filepath',save_file_location);
                EEG = eeg_checkset( EEG );

                [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG); % copy it to ALLEEG
                %eeglab redraw;
end