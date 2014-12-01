% script import markers
Fs_eeg = 500; 
eeg_time_correction = 1.38;     %sec, BNBO_ses4_block6 = 2.78, BNBO_ses5_block5 = 1.38
ind400 = find(marker_block(:,2) == 400);
ind300 = ind400 - 1;
markers = double([[marker_block(ind300,2) marker_block(ind300,1)];
                    [marker_block(ind400,2) marker_block(ind400,1)]]);
                
markers(:,2) = markers(:,2)./Fs_eeg - eeg_time_correction;
dlmwrite('BNBO_ses5_block5_markers.txt',markers,'delimiter','\t','precision','%.4f');
