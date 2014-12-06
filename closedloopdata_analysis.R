# BMI Mahi closed-loop data analysis
#setwd("/home//nikunj//Documents//R_programs")
library(signal)
library(R.matlab)
library(pracma)
############################## FUNCTIONS

ExtractUniqueTriggers <- function(TrigIn, lookback = 1, polarity = 0){
  Unique_Index <- NULL
  TrigOut <- numeric(length(TrigIn))
  
  if (polarity == 1){
    index <- which(TrigIn == 5) 
    for(i in seq_along(index)){
    
      if((i == 1) && (TrigIn[index[i]]==5)){
        Unique_Index <- c(Unique_Index, index[i])
      TrigOut[index[i]] <- 0;        
      next
      }
      
      if ((TrigIn[index[i]]==5)  && (TrigIn[index[i]-1]==0)){
        Unique_Index <- c(Unique_Index,index[i])
        TrigOut[index[i]] <- 1;
      }
    }
  }
  else{
    index <- which(TrigIn == 0) 
    for(i in seq_along(index)){
      
      if ((TrigIn[index[i]]==0)  && (TrigIn[index[i]-1]==5)){
        Unique_Index <- c(Unique_Index,index[i])
        TrigOut[index[i]] <- 1;
      }
    }
    
  }
  TrigOut
}

find_next_stimulus_index <- function(stimulus_trigs,catch_trigs){
  #function(a,b){
  #b[sapply(sapply(seq_along(a),function(x) which(b[-seq(x)]>a[x])+x),"[",1)]
  nearest_index_to_catch <- numeric(length(catch_trigs))
  for(j in seq_along(catch_trigs)){
    nearest_index_to_catch[j] <- stimulus_trigs[which.min(abs(stimulus_trigs - catch_trigs[j]))]
  }
  nearest_index_to_catch
}

find_next_response_index <- function(response_trigs,catch_trigs){
  #function(a,b){
  #b[sapply(sapply(seq_along(a),function(x) which(b[-seq(x)]>a[x])+x),"[",1)]
  nearest_index_to_catch <- numeric(length(catch_trigs))
  for(j in seq_along(catch_trigs)){
    nearest_index_to_catch[j] <- response_trigs[which.min(abs(response_trigs - catch_trigs[j]))+1]
  }
  nearest_index_to_catch
}


################################### MAIN PROGRAM ############################################################# 
directory <- "C:/NRI_BMI_Mahi_Project_files/All_Subjects/"

# Changes to be made
Subject_name <- "BNBO"        #1 JF 0.0232
closeloop_Sess_num <- 3       #2
Block_num <- c(5)         #3
# Velocity Thresholds: JF - 0.0232, LSGR - 0.008, PLSH - 0.0183, ERWS - 0.0123, BNBO - 0.0267
velocity_threshold <- 0.0267  #4

Total_num_of_trials <- NULL
Successful_trials <- NULL
Catch_trials <- NULL
Failed_Catch_trials <- NULL

cl_session_stats <- data.frame(
                              Block_number = numeric(),
                              Start_of_trial = numeric(),
                              End_of_trial = numeric(),
                              Valid_or_catch = numeric(),
                              Intent_detected = numeric(),
                              Time_to_trigger = numeric(),
                              Number_of_attempts = numeric(),
                              EEG_decisions = numeric(),
                              EEG_EMG_decisions = numeric(),
                              MRCP_slope = numeric(),
                              MRCP_neg_peak = numeric(),
                              MRCP_AUC = numeric(),
                              MRCP_mahalanobis = numeric(),
                              Likert_score = numeric()
                              )

folderid <- paste(c("Subject_",Subject_name,"/",Subject_name,"_Session",toString(closeloop_Sess_num),"/"),collapse = '')

for (bc in seq_along(Block_num)){

      # Load kinematics file
      #cl_kinematics_data <- read.table("JF_ses4_block5_closeloop_kinematics.txt",header = TRUE, skip = 14)
      
      fileid <- paste(c(Subject_name,"_ses",toString(closeloop_Sess_num),"_block",toString(Block_num[bc])),collapse = '')
#      if (Block_num[bc] == 7){
#        cl_kinematics_data <- read.table(paste(c(directory,folderid,fileid,"_closeloop_kinematics.txt"),collapse = ''),header = TRUE, skip = 14,nrows = 395346)
#      }
#      else{
      cl_kinematics_data <- read.table(paste(c(directory,folderid,fileid,"_closeloop_kinematics.txt"),collapse = ''),header = TRUE, skip = 14) #,nrows = 395346)
#      }
      kin_header <- scan(paste(c(directory,folderid,fileid,"_closeloop_kinematics.txt"),collapse = ''),what ="character",skip = 11,nlines = 1)
      block_likert <- type.convert(kin_header[!is.na(type.convert(kin_header,na.strings = c("Survey","Responses:")))])
      cl_BMI_data <- readMat(paste(c(directory,folderid,fileid,"_closeloop_results.mat"),collapse = ''),fixNames = FALSE)
      print(c("Working with block", toString(Block_num[bc])))
      
      # Change column-names for kinematics data
      # Trig1 - stimulus
      # Trig2 - response
      # Trig_Mov - Movement onset
      colnames(cl_kinematics_data)[c(1,17,18,19)] <- c("time","Target_shown","Target_reached","Move_onset")
      colnames(cl_kinematics_data)[c(21,22)] <- c("Catch","Timeout") # Timeout and catch columns must be swapped - Ted
      colnames(cl_kinematics_data)[c(7,12)] <- c("Elbow_velocity","Elbow_torque")
      
      # Filter the velocity and compute magnitude
      filt_vel <- abs(sgolayfilt(cl_kinematics_data$Elbow_velocity,p = 1,n = 201))
      vel_triggers <- numeric(length(filt_vel))
      vel_triggers[filt_vel >= velocity_threshold] <- 5   # Digitize 0 or 5 as trigger levels
      vel_triggers_initial <- ExtractUniqueTriggers(vel_triggers,polarity = 1)
      trig_indices <- which(vel_triggers_initial == 1)
      diff_trig_indices <- diff(trig_indices)
      trig_indices[c(FALSE, (diff_trig_indices < 1000))] <- 0
      vel_trigger_mod <- numeric(length(vel_triggers_initial))
      vel_trigger_mod[trig_indices] <- 1
      
      # Get first sample whe a trigger signal was generated
      cl_kinematics_data$Target_shown <- ExtractUniqueTriggers(cl_kinematics_data$Target_shown)
      cl_kinematics_data$Target_reached <- ExtractUniqueTriggers(cl_kinematics_data$Target_reached)
      
      # Correct for start of block triggers - set them to zero in Target_shown, Target_reached
      temp_intersect <- intersect(which(cl_kinematics_data$Target_shown == 1), which(cl_kinematics_data$Target_reached == 1))
      cl_kinematics_data$Target_shown[temp_intersect] <- 0
      cl_kinematics_data$Target_reached[temp_intersect] <- 0
      
      # Extract unique movement onset, catch trials and time out triggers
      cl_kinematics_data$Move_onset <- ExtractUniqueTriggers(cl_kinematics_data$Move_onset)
      cl_kinematics_data$Catch <- ExtractUniqueTriggers(cl_kinematics_data$Catch)
      cl_kinematics_data$Timeout <- ExtractUniqueTriggers(cl_kinematics_data$Timeout)
      
      if (Subject_name == "JF"){
        # Remove extra Triggers from Target_shown and Move_onset
#        cl_kinematics_data[which(cl_kinematics_data$Catch == 1),"Target_shown"] <- 0
        cl_kinematics_data[which(cl_kinematics_data$Timeout == 1),"Move_onset"] <- 0
      }
      
      # Plot all triggers - How to create better plots?
      mycolors <- c("green","magenta","blue","red","black","green")
      #plot.ts(cl_kinematics_data[,c("Catch","Target_shown","Target_reached","Move_onset","Timeout")],plot.type = "single",col = mycolors, xy.labels = "")
      data_to_plot <- data.frame(filt_vel,vel_trigger_mod,cl_kinematics_data[,c("Target_shown","Target_reached","Move_onset","Timeout")])
      plot.ts(data_to_plot,plot.type = "single",col = mycolors, xy.labels = "",ylim = c(0,2))
            
      # Remove Catch trials from Valid trials count i.e. correct for Catch Trials
      if (Subject_name == "JF"){
        # DO NOT Combine Target_reached + Timeout
        end_trial <- cl_kinematics_data$Target_reached
        all_response_indices <- which(end_trial==1)
        catch_indices <- which(cl_kinematics_data$Catch == 1)
        all_stimulus_indices <- which(cl_kinematics_data$Target_shown == 1)
        nearest_stimulus_indices <- catch_indices    # Since Catch and Target shown triggers overlap
        nearest_response_indices <- find_next_response_index(all_response_indices,catch_indices)
      }
      else{      
        # Combine Target_reached + Timeout
        end_trial <- cl_kinematics_data$Target_reached + cl_kinematics_data$Timeout
        all_response_indices <- which(end_trial==1)
        catch_indices <- which(cl_kinematics_data$Catch == 1)
        all_stimulus_indices <- which(cl_kinematics_data$Target_shown == 1)
        nearest_stimulus_indices <- find_next_stimulus_index(all_stimulus_indices,catch_indices)
        nearest_response_indices <- find_next_response_index(all_response_indices,catch_indices)
      }
      
      # Create cl_trial_stats that contains 14 factors
      cl_trial_stats <- data.frame(
                              Block_number = rep_len(Block_num[bc],length(all_stimulus_indices)),
                              Start_of_trial = all_stimulus_indices,
                              End_of_trial = all_response_indices,
                              Valid_or_catch = rep_len(1,length(all_stimulus_indices)),
                              Intent_detected = numeric(length(all_stimulus_indices)),
                              Time_to_trigger = numeric(length(all_stimulus_indices)),
                              Number_of_attempts = numeric(length(all_stimulus_indices)),
                              EEG_decisions = numeric(length(all_stimulus_indices)),
                              EEG_EMG_decisions = numeric(length(all_stimulus_indices)),
                              MRCP_slope = numeric(length(all_stimulus_indices)),
                              MRCP_neg_peak = numeric(length(all_stimulus_indices)),
                              MRCP_AUC = numeric(length(all_stimulus_indices)),
                              MRCP_mahalanobis = numeric(length(all_stimulus_indices)),
                              Likert_score = block_likert
      )
      
      for (i in seq_along(nearest_stimulus_indices)){
        cl_trial_stats$Valid_or_catch[which(all_stimulus_indices == nearest_stimulus_indices[i])] <- 2      
      }
      adj_start_of_trial <- floor(cl_trial_stats$Start_of_trial/2)    # Downsample to 500 Hz
      adj_end_of_trial <- floor(cl_trial_stats$End_of_trial/2)
      
      for (m in seq_along(all_stimulus_indices)){
        kinematic_data_trial_interval <- c(all_stimulus_indices[m],all_response_indices[m])
        
        if(1 %in% cl_kinematics_data$Move_onset[kinematic_data_trial_interval[1]:kinematic_data_trial_interval[2]]){
          # Intent was detected
          cl_trial_stats$Intent_detected[m] <- 1
          cl_trial_stats$Time_to_trigger[m] <- 
            which(cl_kinematics_data$Move_onset[kinematic_data_trial_interval[1]:kinematic_data_trial_interval[2]] == 1)/1000  # Fs = 1000 Hz for kinematics data
        }
        else{ 
          #Timeout occured, intent was not detected
          cl_trial_stats$Intent_detected[m] <- 0
          cl_trial_stats$Time_to_trigger[m] <- (kinematic_data_trial_interval[2] - kinematic_data_trial_interval[1])/1000
        }
        
        cl_trial_stats$Number_of_attempts[m] <- 
          length(which(vel_trigger_mod[kinematic_data_trial_interval[1]:kinematic_data_trial_interval[2]] == 1))
      }



      # Determine values of features when Intent was detected i.e. last EEG_GO decision(marker 300)
      # We have all EEG_GO decision available - use move_counts 
      # Subtract start/stop prediction index
      # This subtraction also compensates for the delay between intiation of EEG and kinematic data capture
      #cl_BMI_data$marker_block[,1] <- cl_BMI_data$marker_block[,1] - cl_BMI_data$marker_block[min(which(cl_BMI_data$marker_block[,2]==50)),1]
      adj_marker_block_time_stamps <- cl_BMI_data$marker_block[,1] - cl_BMI_data$marker_block[min(which(cl_BMI_data$marker_block[,2]==50)),1]
      for (k in seq_along(adj_start_of_trial)){
          bmi_data_trial_interval <- intersect(which(adj_marker_block_time_stamps >= adj_start_of_trial[k]),
                    which(adj_marker_block_time_stamps < adj_end_of_trial[k]))
          if (300 %in% cl_BMI_data$marker_block[bmi_data_trial_interval,2]){
            cl_trial_stats$EEG_decisions[k] <- 1
          }
          if (400 %in% cl_BMI_data$marker_block[bmi_data_trial_interval,2]){
            cl_trial_stats$EEG_EMG_decisions[k] <- 1
            # Copy feature vectors for EEG_GO only when EEG_EMG_GO occurs  
            f_index <- bmi_data_trial_interval[max(which(cl_BMI_data$marker_block[bmi_data_trial_interval,2] == 300))]
            spatial_avg_index <- floor((cl_BMI_data$marker_block[f_index,1]/500)*20) # Resample to 20 Hz - Index for Overall_spatial_chan_avg
            feature_index <- floor((adj_marker_block_time_stamps[f_index]/500)*20) # Resample to 20 Hz - Index for all_feature_vectors
            feature_index <- feature_index + 1 # Correction of 1 sample added after manually calculating the features (Dec 5,14)
            
            # Directly get features
            cl_trial_stats[k,c("MRCP_slope","MRCP_neg_peak","MRCP_AUC","MRCP_mahalanobis")] <- t(cl_BMI_data$all_feature_vectors[,feature_index])            
            # Segment the Overall Spatial Avg and use it to derive features
            # ~~~Need to know window length for the subject, BNBO - 0.65s
            opt_window_length <- 0.65
            resamp_Fs <- 20
            spatial_avg_epoch <- cl_BMI_data$Overall_spatial_chan_avg[(spatial_avg_index - opt_window_length*resamp_Fs):spatial_avg_index]
            epoch_time <- seq(from = -1*opt_window_length,to = 0,by = 1/resamp_Fs)
            
            cal_feature_vec <- t(c((spatial_avg_epoch[length(spatial_avg_epoch)] - spatial_avg_epoch[1])/(epoch_time[length(epoch_time)] - epoch_time[1]),
              min(spatial_avg_epoch),
              trapz(epoch_time,spatial_avg_epoch),
              0))
            
            #cat("Cal: ",cal_feature_vec,"\t","Meas: ", toString(cl_trial_stats[k,c("MRCP_slope","MRCP_neg_peak","MRCP_AUC","MRCP_mahalanobis")]))
          }
          # Segment EEG signal during when Intent is detected
          # Write time stamp to .txt file
      }
      
      
# Correct for Catch Trials after finding nearest stimulus and response indices above
#      for (i in seq_along(nearest_stimulus_indices)){
        #new_block$valid_or_catch_flag[which(all_stimulus_indices == nearest_stimulus_indices[i])] <- 2
        #cl_kinematics_data$Move_onset[which(cl_kinematics_data[nearest_stimulus_indices[i]:nearest_response_indices[i],"Move_onset"] == 1)] <- 2
                      
        
#        cl_kinematics_data[nearest_stimulus_indices[i]:nearest_response_indices[i],"Move_onset"] <- 0
        # Remove attempted movements during catch trials
#        vel_trigger_mod[nearest_stimulus_indices[i]:nearest_response_indices[i]] <- 0      
        #vel_trigger_mod[which(vel_trigger_mod[nearest_stimulus_indices[i]:nearest_response_indices[i]] == 1)] <- 2
        
#      }
      
      # Remove catch trial triggers from end_trial and Target_shown
#      end_trial[intersect(all_response_indices,nearest_response_indices)] <- 0
#      cl_kinematics_data$Target_shown[intersect(all_stimulus_indices,nearest_stimulus_indices)] <- 0
#      data_to_plot <- data.frame(filt_vel,vel_trigger_mod,cl_kinematics_data[,c("Move_onset","Target_shown")],end_trial)
#      plot.ts(data_to_plot,plot.type = "single",col = mycolors, xy.labels = "",ylim = c(0,2))
      
      
      #if (Subject_name == "JF"){
      #Total_num_of_trials <- c(Total_num_of_trials, as.numeric(length(all_stimulus_indices)))
      #}
      #else{
      #Total_num_of_trials <- c(Total_num_of_trials, as.numeric(length(all_stimulus_indices)) - as.numeric(length(catch_indices)))
      #}
      #Successful_trials <- c(Successful_trials, as.numeric(length(which(cl_kinematics_data$Move_onset==1))))
      
            
      # Find number of move attempts and successful move/timeout for each trial 
      # by looking at the interval between target_shown and end_trial
#      trial_start_indices <- which(cl_kinematics_data$Target_shown == 1)
#      trial_complete_indices <- which(end_trial==1)
      #preallocate space to data.frame, computationally efficient
#      block_statistics <- data.frame(Start_of_trial = numeric(length(trial_start_indices)),
#                                     End_of_trial = numeric(length(trial_start_indices)),
#                                     Number_of_move_attempts = numeric(length(trial_start_indices)),
#                                     Successful_move_attempts = numeric(length(trial_start_indices)))
      
      
     
#      Successful_trials <- c(Successful_trials, as.numeric(length(which(cl_kinematics_data$Move_onset == 1))))
#      Total_num_of_trials <- c(Total_num_of_trials, as.numeric(length(which(cl_kinematics_data$Target_shown ==  1))))      
#      Catch_trials <- c(Catch_trials, as.numeric(length(which(cl_kinematics_data$Catch == 1))))
      #Failed Catch Trials
#      cat("Number of move attempts: ", sum(block_statistics$Number_of_move_attempts), "\n")
#      cat("Successful move attempts: ", sum(block_statistics$Successful_move_attempts), "\n")
      
      cl_session_stats <- rbind.data.frame(cl_session_stats,cl_trial_stats)
      block_ind <- which(cl_session_stats$Block_number == Block_num[bc])
      
      Total_num_of_trials <- c(Total_num_of_trials,length(which(cl_session_stats[block_ind,"Valid_or_catch"] == 1)))
      Successful_trials <- c(Successful_trials,length(intersect(which(cl_session_stats[block_ind,"Valid_or_catch"] == 1), 
                                            which(cl_session_stats[block_ind,"Intent_detected"] == 1))))
      Catch_trials <- c(Catch_trials,length(which(cl_session_stats[block_ind,"Valid_or_catch"] == 2)))
      Failed_Catch_trials <- c(Failed_Catch_trials,length(intersect(which(cl_session_stats[block_ind,"Valid_or_catch"] == 2), 
                                                   which(cl_session_stats[block_ind,"Intent_detected"] == 1))))

}
cat("Block Numbers:       ", Block_num, "\n")
cat("Successful Trials:   ", Successful_trials, "\n")
cat("Total Num of trials: ", Total_num_of_trials, "\n")
cat("Failed Catch trials: ", Failed_Catch_trials, "\n")
cat("Total Catch Trials:  ", Catch_trials,"\n")

save_filename <- paste(c(directory,folderid,Subject_name,"_ses",toString(closeloop_Sess_num),"_cloop_statistics.csv"),collapse = '')
fileConn <- file(save_filename)
cat("Block Numbers,", toString(Block_num), "\n", #file = fileConn, sep = ',', append = T)
    "Successful Trials,", toString(Successful_trials), "\n", 
    "Total Num of trials,", toString(Total_num_of_trials), "\n",
    "Failed Catch trials,", toString(Failed_Catch_trials), "\n",
    "Total Catch Trials,", toString(Catch_trials),"\n\n",file = fileConn, sep = '', append = T)  
close(fileConn)
#fileConn <- file(save_filename)
write.table(x = cl_session_stats,file = save_filename,append = T,col.names = T, sep = ',')
#close(fileConn)


