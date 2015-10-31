# BMI Mahi closed-loop data analysis
#setwd("/home//nikunj//Documents//R_programs")
#library(signal)
#library(R.matlab)
#library(pracma)
#library(abind)
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
    #nearest_index_to_catch[j] <- stimulus_trigs[which.min(abs(stimulus_trigs - catch_trigs[j]))]
    # Corrected by Ted's help
    nearest_index_to_catch[j] <- stimulus_trigs[which((stimulus_trigs-catch_trigs[j]) >= 0)[1]]
  }
  nearest_index_to_catch
}

find_next_response_index <- function(response_trigs,catch_trigs){
  #function(a,b){
  #b[sapply(sapply(seq_along(a),function(x) which(b[-seq(x)]>a[x])+x),"[",1)]
  nearest_index_to_catch <- numeric(length(catch_trigs))
  for(j in seq_along(catch_trigs)){
    # nearest_index_to_catch[j] <- response_trigs[which.min(abs(response_trigs - catch_trigs[j]))+1]
    # Corrected by Ted's help
    nearest_index_to_catch[j] <- response_trigs[which((response_trigs-catch_trigs[j]) >= 0)[1]]
  }
  nearest_index_to_catch
}


################################### MAIN PROGRAM ############################################################# 
analyze_closedloop_session_data <- function(directory,Subject_name,closeloop_Sess_num,closedloop_Block_num){
#setwd(directory)
#directory <- "C:/NRI_BMI_Mahi_Project_files/All_Subjects/"

# Changes to be made

#  Subject_name <- "S9007"            #1 
#  closeloop_Sess_num <- 14           #2
#  closedloop_Block_num <- c(1)      #3
  velocity_threshold <- (1.19)*(pi/180)      #4  # Velocity Thresholds: JF - 0.0232, LSGR - 0.008, PLSH - 0.0183, ERWS - 0.0123, BNBO - 0.0267
  Cond_num  <- 1                    #5 1 - Backdrive, 3-Triggered modes
  use_simulated_closeloop_results <- 0
  simulation_mode <- "UD" # UD - use simulated user-driven mode; UT - use simulated user-triggered mode
  
  Training_Sess_num <- "2"
  Training_Block_num <- "160"

  # Load Classifier Model used during closed-loop
  Training_folderid <- paste(c("Subject_",Subject_name,"/",Subject_name,"_Session",Training_Sess_num,"/"),collapse = '')
  Training_fileid <- paste(c(Subject_name,"_ses",Training_Sess_num,"_cond",toString(Cond_num),"_block",Training_Block_num),collapse = '')
  Classifier <- readMat(paste(c(directory,Training_folderid,Training_fileid,"_classifier_parameters.mat"),collapse = ''),fixNames = FALSE)
  
  
  Total_num_of_trials <- NULL
  Successful_trials <- NULL
  Successful_EEG_EMG_trials <- NULL
  Catch_trials <- NULL
  Failed_Catch_trials <- NULL
  Intent_EEG_epochs_session <- NULL
  
  cl_session_stats <- data.frame(
                                Session_number = numeric(),
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
                                feature_index = numeric(),
                                Corrected_spatial_chan_avg_index = numeric(),
                                Correction_applied_in_samples = numeric(),
                                Likert_score = numeric(),
                                Target = numeric(),
                                Kinematic_onset_sample_num = numeric(),
                                Target_is_hit = numeric(),
                                EEG_Kinematic_latency_ms = numeric()
                                )
  
  folderid <- paste(c("Subject_",Subject_name,"/",Subject_name,"_Session",toString(closeloop_Sess_num),"/"),collapse = '')
  sim_results_folderid <- paste(c("Subject_",Subject_name,"/",Subject_name,"_Session",toString(closeloop_Sess_num),"B/"),collapse = '')
  
  for (bc in seq_along(closedloop_Block_num)){
  
        # Load kinematics file
        #cl_kinematics_data <- read.table("JF_ses4_block5_closeloop_kinematics.txt",header = TRUE, skip = 14)
        
        fileid <- paste(c(Subject_name,"_ses",toString(closeloop_Sess_num),"_block",toString(closedloop_Block_num[bc])),collapse = '')
        kin_fileid <- paste(c(Subject_name,"_CLses",toString(closeloop_Sess_num),"_block",toString(closedloop_Block_num[bc])),collapse = '')
        kin_filename <- dir(path = paste(c(directory,folderid),collapse = ''),pattern = paste(c(kin_fileid,"_kinematics"),collapse = ''))
        sim_results_fileid <- paste(c(Subject_name,"_ses",toString(closeloop_Sess_num),"B_block",toString(closedloop_Block_num[bc]),"_",simulation_mode),collapse = '')
        
  #      if (closedloop_Block_num[bc] == 7){
  #        cl_kinematics_data <- read.table(paste(c(directory,folderid,fileid,"_closeloop_kinematics.txt"),collapse = ''),header = TRUE, skip = 14,nrows = 395346)
  #      }
  #      else{
        cl_kinematics_data <- read.table(paste(c(directory,folderid,kin_filename),collapse = ''),header = TRUE, skip = 14) #,nrows = 395346)
  #      }
        kin_header <- scan(paste(c(directory,folderid,kin_filename),collapse = ''),what ="character",skip = 11,nlines = 1)
        block_likert <- type.convert(kin_header[!is.na(type.convert(kin_header,na.strings = c("Survey","Responses:")))])
        if(use_simulated_closeloop_results == 1){
          cl_BMI_data <- readMat(paste(c(directory,sim_results_folderid,sim_results_fileid,"_closeloop_results.mat"),collapse = ''),fixNames = FALSE)
        }else{
          # cl_BMI_data <- readMat(paste(c(directory,folderid,fileid,"_closeloop_results.mat"),collapse = ''),fixNames = FALSE)
          # Read files that have time-stamps in filenames
          closeloop_mat_filename <- dir(path = paste(c(directory,folderid),collapse = ''),pattern = paste(c(fileid,"_closeloop_results"),collapse = ''))
          if (length(closeloop_mat_filename)==0){
            BMI_mat_file_exists <- 0
          }
          else{
            BMI_mat_file_exists <- 1
            cl_BMI_data <- readMat(paste(c(directory,folderid,closeloop_mat_filename),collapse = ''),fixNames = FALSE)
          }
        }
  
        if (BMI_mat_file_exists == 1){
          cl_BMI_data$marker_block[,1] <- as.double(cl_BMI_data$marker_block[,1]) # convert marker block in double precision
          marker_block_index <- which(cl_BMI_data$marker_block[,2] == 300)        # Used to find feature_index
          #move_counts_index <- which(cl_BMI_data$move_counts == max(cl_BMI_data$all_cloop_cnts_threshold))  # Also used to find feature_index
          # Correction for calculating move_counts_index
          move_counts_index <- which((cl_BMI_data$all_cloop_cnts_threshold - cl_BMI_data$move_counts) <= 0)
        }
        print(c("Working with block", toString(closedloop_Block_num[bc])))
        
        # Change column-names for kinematics data
        # Trig1 - stimulus
        # Trig2 - response
        # Trig_Mov - Movement onset
        colnames(cl_kinematics_data)[c(1,17,18,19)] <- c("time","Target_shown","Target_reached","Move_onset")
        colnames(cl_kinematics_data)[c(22,21)] <- c("Catch","Timeout") # Timeout is 21 and catch is 22 on 9/13/2015. Previously these were swapped
        colnames(cl_kinematics_data)[c(2,7,12)] <- c("Elbow_position","Elbow_velocity","Elbow_torque")
        
        # Get first sample when a trigger signal was generated
        cl_kinematics_data$Target_shown <- ExtractUniqueTriggers(cl_kinematics_data$Target_shown)
        cl_kinematics_data$Target_reached <- ExtractUniqueTriggers(cl_kinematics_data$Target_reached)
  
  
        # Extract unique movement onset, catch trials and time out triggers
        cl_kinematics_data$Move_onset <- ExtractUniqueTriggers(cl_kinematics_data$Move_onset)
        cl_kinematics_data$Catch <- ExtractUniqueTriggers(cl_kinematics_data$Catch)
        cl_kinematics_data$Timeout <- ExtractUniqueTriggers(cl_kinematics_data$Timeout)
  
        
        # Correct for start and end of block triggers - set them to zero in Target_shown, Target_reached 
        temp_intersect <- intersect(which(cl_kinematics_data$Target_shown == 1), which(cl_kinematics_data$Target_reached == 1))
        if(length(temp_intersect) < 2){
          if (which(which(cl_kinematics_data$Target_shown == 1) == temp_intersect) == 1){
            cl_kinematics_data <- cl_kinematics_data[-c(1:temp_intersect[1]),] # Added 9-13-2015
          }else if((which(which(cl_kinematics_data$Target_shown == 1) == temp_intersect) > 1)){
            cl_kinematics_data <- cl_kinematics_data[-c(temp_intersect[1]:nrow(cl_kinematics_data)),]        
          }else{
            #do nothing
          }        
        }else{
          cl_kinematics_data <- cl_kinematics_data[-c(1:temp_intersect[1]),] # Added Dec 8,2014
          temp_intersect <- intersect(which(cl_kinematics_data$Target_shown == 1), which(cl_kinematics_data$Target_reached == 1))
          cl_kinematics_data <- cl_kinematics_data[-c(temp_intersect[1]:nrow(cl_kinematics_data)),]
        }
        
        # Remove extra Target_reached triggers generated for likert input - 9/13/2015
        target_shown_instants <- which(cl_kinematics_data$Target_shown == 1)
        target_reached_instances <- which(cl_kinematics_data$Target_reached == 1)
        new_Target_reached <- numeric(length(cl_kinematics_data$Target_reached))
        for (j in seq_along(target_shown_instants)){        
          val_instant <- which((target_reached_instances - target_shown_instants[j]) >= 0)
            if(!isempty(val_instant)){
              nearest_target_reached_instance <- target_reached_instances[val_instant[1]]
              if(1 %in% cl_kinematics_data$Timeout[target_shown_instants[j]:nearest_target_reached_instance] == 1){
                # do nothing
              }
              else{
                # use this as target_reached trigger
                new_Target_reached[nearest_target_reached_instance] <- 1
              }
            }      
        }
        cl_kinematics_data$Target_reached <- new_Target_reached  
  
        # Filter the velocity and compute magnitude
        #filt_vel <- abs(sgolayfilt(cl_kinematics_data$Elbow_velocity,p = 1,n = 201))  # No filtering to avoid any time delays
        filt_vel <- abs(cl_kinematics_data$Elbow_velocity)
        vel_triggers <- numeric(length(filt_vel))
        vel_triggers[filt_vel >= velocity_threshold] <- 5   # Digitize 0 or 5 as trigger levels
        vel_triggers_initial <- ExtractUniqueTriggers(vel_triggers,polarity = 1)
        trig_indices <- which(vel_triggers_initial == 1)
        diff_trig_indices <- diff(trig_indices)
        trig_indices[c(FALSE, (diff_trig_indices < 1000))] <- 0
        vel_trigger_mod <- numeric(length(vel_triggers_initial))
        vel_trigger_mod[trig_indices] <- 1
              
        if (Subject_name == "JF"){
          # Remove extra Triggers from Target_shown and Move_onset
  #        cl_kinematics_data[which(cl_kinematics_data$Catch == 1),"Target_shown"] <- 0
          cl_kinematics_data[which(cl_kinematics_data$Timeout == 1),"Move_onset"] <- 0
          temp_intersect <- intersect(which(cl_kinematics_data$Target_shown == 1), which(cl_kinematics_data$Target_reached == 1))
          cl_kinematics_data$Target_shown[temp_intersect] <- 0       # No longer requried for other subjects, Dec 8,2014
          cl_kinematics_data$Target_reached[temp_intersect] <- 0
        }
        
        # Plot all triggers - How to create better plots?
        mycolors <- c("green","magenta","blue","black","green","red","yellow","black")
        #plot.ts(cl_kinematics_data[,c("Catch","Target_shown","Target_reached","Move_onset","Timeout")],plot.type = "single",col = mycolors, xy.labels = "")
        data_to_plot <- data.frame(filt_vel,cl_kinematics_data[,c("Target_shown","Target_reached","Move_onset","Timeout","Catch")])
        #plot.ts(data_to_plot,plot.type = "single",col = mycolors, xy.labels = "",ylim = c(0,0.5))
              
        # Remove Catch trials from Valid trials count i.e. correct for Catch Trials
        if (Subject_name == "JF"){
          # DO NOT Combine Target_reached/Move_onset + Timeout
          end_trial <- cl_kinematics_data$Target_reached
          #end_trial <- cl_kinematics_data$Move_onset      # Added Dec 8,2014
          all_response_indices <- which(end_trial==1)
          catch_indices <- which(cl_kinematics_data$Catch == 1)
          all_stimulus_indices <- which(cl_kinematics_data$Target_shown == 1)
          nearest_stimulus_indices <- catch_indices    # Since Catch and Target shown triggers overlap
          nearest_response_indices <- find_next_response_index(all_response_indices,catch_indices)
        }
        else{      
          # Combine Target_reached + Timeout
          Target_hit <- cl_kinematics_data$Target_reached + cl_kinematics_data$Timeout
          # Instead Combine Move_onset + Timeout Added Dec 8,2014
          end_trial <- cl_kinematics_data$Move_onset + cl_kinematics_data$Timeout
          all_response_indices <- which(end_trial==1)
          all_target_hits <- which(Target_hit==1)
          catch_indices <- which(cl_kinematics_data$Catch == 1)
          all_stimulus_indices <- which(cl_kinematics_data$Target_shown == 1)
          nearest_stimulus_indices <- find_next_stimulus_index(all_stimulus_indices,catch_indices)
          nearest_response_indices <- find_next_response_index(all_response_indices,catch_indices)
        }
        
        # Create cl_trial_stats that contains all factors
        cl_trial_stats <- data.frame(
                                Session_nos = rep_len(closeloop_Sess_num,length(all_stimulus_indices)),
                                Block_number = rep_len(closedloop_Block_num[bc],length(all_stimulus_indices)),
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
                                feature_index = numeric(length(all_stimulus_indices)),
                                Corrected_spatial_chan_avg_index = numeric(length(all_stimulus_indices)),
                                Correction_applied_in_samples = numeric(length(all_stimulus_indices)),
                                Likert_score = block_likert,
                                Target = cl_kinematics_data$Target[all_stimulus_indices],
                                Kinematic_onset_sample_num = numeric(length(all_stimulus_indices)),
                                Target_is_hit = all_target_hits,
                                EEG_Kinematic_latency_ms = numeric(length(all_stimulus_indices))
        )
        
        for (i in seq_along(nearest_stimulus_indices)){
          cl_trial_stats$Valid_or_catch[which(all_stimulus_indices == nearest_stimulus_indices[i])] <- 2      
        }
       
        for (m in seq_along(all_stimulus_indices)){
          kinematic_data_trial_interval <- c(all_stimulus_indices[m],all_response_indices[m])
          
          if(1 %in% cl_kinematics_data$Move_onset[kinematic_data_trial_interval[1]:kinematic_data_trial_interval[2]]){
            # Intent was detected
            cl_trial_stats$Intent_detected[m] <- 1
            cl_trial_stats$Time_to_trigger[m] <- 
              which(cl_kinematics_data$Move_onset[kinematic_data_trial_interval[1]:kinematic_data_trial_interval[2]] == 1)/1000  # Fs = 1000 Hz for kinematics data
          }[]
          else{ 
            #Timeout occured, intent was not detected
            cl_trial_stats$Intent_detected[m] <- 0
            cl_trial_stats$Time_to_trigger[m] <- (kinematic_data_trial_interval[2] - kinematic_data_trial_interval[1])/1000
          }
          
          kinematic_data_trial_interval_new <- c(all_stimulus_indices[m],all_target_hits[m]) # Added for calculating EEG-Onset latency
          
          cl_trial_stats$Number_of_attempts[m] <- 
            length(which(vel_trigger_mod[kinematic_data_trial_interval_new[1]:kinematic_data_trial_interval_new[2]] == 1))
          
          if(cl_trial_stats$Number_of_attempts[m] == 0){
            cl_trial_stats$Kinematic_onset_sample_num[m] <- kinematic_data_trial_interval_new[1]
          }
          else{
          cl_trial_stats$Kinematic_onset_sample_num[m] <- 
            kinematic_data_trial_interval[1] + max(which(vel_trigger_mod[kinematic_data_trial_interval_new[1]:kinematic_data_trial_interval_new[2]] == 1))
          }
        }
  
        adj_start_of_trial <- round(cl_trial_stats$Start_of_trial/2) -  250   # Downsample to 500 Hz
        adj_end_of_trial <- round(cl_trial_stats$End_of_trial/2) + 350
  
        # Determine values of features when Intent was detected i.e. last EEG_GO decision(marker 300)
        # We have all EEG_GO decision available - use move_counts 
        # Subtract start/stop prediction index
        # This subtraction also compensates for the delay between intiation of EEG and kinematic data capture
        epoch_start_time <- -2.5
        epoch_end_time <- 1
        resamp_Fs <- 20
        #upsamp_Fs <- 500
        #upsampled_Overall_spatial_chan_avg <- resample(x = cl_BMI_data$Overall_spatial_chan_avg,p = upsamp_Fs,q = 20)
  
        # marker_block is not accurate metric -- Use move_counts instead calculating features
        if (BMI_mat_file_exists == 1){          
          adj_marker_block_time_stamps <- cl_BMI_data$marker_block[,1] - cl_BMI_data$marker_block[min(which(cl_BMI_data$marker_block[,2]==50)),1]
          for (k in seq_along(adj_start_of_trial)){
              bmi_data_trial_interval <- intersect(which(adj_marker_block_time_stamps >= adj_start_of_trial[k]),
                        which(adj_marker_block_time_stamps < adj_end_of_trial[k]))
              
              if (length(bmi_data_trial_interval) == 0){
                
                cat("BMI data interval not found!!\n")
                Intent_EEG_epochs_trial <- array(0, 
                                                 dim = c(dim(cl_BMI_data$processed_eeg)[1]+2,
                                                         (epoch_end_time-epoch_start_time)*resamp_Fs+1,
                                                         1))
                Intent_EEG_epochs_session <- abind(Intent_EEG_epochs_session,Intent_EEG_epochs_trial,along = 3)
                cl_trial_stats$Corrected_spatial_chan_avg_index[k] <- 0
                cl_trial_stats$Correction_applied_in_samples[k] <- 0
                cl_trial_stats$feature_index[k] <- 0
                next
              }
                
                
              if (300 %in% cl_BMI_data$marker_block[bmi_data_trial_interval,2]){
                cl_trial_stats$EEG_decisions[k] <- 1
              }
              
              if (400 %in% cl_BMI_data$marker_block[bmi_data_trial_interval,2]){
                cl_trial_stats$EEG_EMG_decisions[k] <- 1
              }
              
              if (cl_trial_stats$Intent_detected[k]){
                # Copy feature vectors for EEG_GO only when EEG_EMG_GO occurs
                ind300 <- which(cl_BMI_data$marker_block[bmi_data_trial_interval,2] == 300)
                if (length(ind300) == 0){
                  ind300 <- which(cl_BMI_data$marker_block[bmi_data_trial_interval-1,2] == 300)
                  marker_block_300 <- bmi_data_trial_interval[max(ind300)]-1
                }
                else{
                  marker_block_300 <- bmi_data_trial_interval[max(ind300)]              
                }
                cl_trial_stats$EEG_Kinematic_latency_ms[k] <- adj_marker_block_time_stamps[marker_block_300]*2 - cl_trial_stats$Kinematic_onset_sample_num[k]
                
                feature_index <- move_counts_index[which(marker_block_index == marker_block_300)]   # Added Dec10,2014
                
                if (use_simulated_closeloop_results == 1){
                  # Do not find EEG features because of some bug
                  next # added 5-21-2015
                }
                
                ## No longer used - Dec10,2014
                # Note: Correct way to resample is new_time_stamp = round((old_time_stamp/old_frequency)*new_frequency)
                # f_index <- marker_block_300
                #feature_index <- round((adj_marker_block_time_stamps[f_index]/500)*20) # Resample to 20 Hz - Index for all_feature_vectors
                #feature_index <- feature_index + 2 # Incorrect - Correction of 5 sample added after manually calculating the features (Dec 5,14)
                #spatial_avg_index <- round((cl_BMI_data$marker_block[f_index,1]/500)*20) # Resample to 20 Hz - Index for Overall_spatial_chan_avg        
                
                # Directly get classification features - (Dec9,2014) No longer used because of imprecision in sample number. Instead used Overall_spatial_chan_avg
                cl_trial_stats[k,c("MRCP_slope","MRCP_neg_peak","MRCP_AUC","MRCP_mahalanobis")] <- t(cl_BMI_data$all_feature_vectors[,feature_index])            
                
                # Segment the Overall Spatial Avg and use it to derive features
                spatial_avg_sample_correction <- round(((cl_BMI_data$marker_block[which(cl_BMI_data$marker_block[,2]==50)[1],1] - cl_BMI_data$marker_block[1,1])/500)*20)
                spatial_avg_index <- feature_index + spatial_avg_sample_correction
                spatial_avg_epoch <- cl_BMI_data$Overall_spatial_chan_avg[(spatial_avg_index - Classifier$smart_window_length*resamp_Fs):spatial_avg_index]
                epoch_time <- seq(from = -1*Classifier$smart_window_length,to = 0,by = 1/resamp_Fs)
                
                cal_feature_vec <- t(c((spatial_avg_epoch[length(spatial_avg_epoch)] - spatial_avg_epoch[1])/(epoch_time[length(epoch_time)] - epoch_time[1]),
                  min(spatial_avg_epoch),
                  trapz(epoch_time,spatial_avg_epoch),
                  sqrt((spatial_avg_epoch - Classifier$smart_Mu_move)%*%(inv(Classifier$smart_Cov_Mat))%*%(t(spatial_avg_epoch - Classifier$smart_Mu_move)))
                  ))
                           
                #cl_trial_stats[k,c("MRCP_slope","MRCP_neg_peak","MRCP_AUC","MRCP_mahalanobis")] <- cal_feature_vec
                # Calculating correction for Overall_spatial_chan_avg
                intersection_index <- 
                  intersect(
                    intersect(which(round(cl_BMI_data$all_feature_vectors[1,],4) %in% round(cal_feature_vec[1],4)),
                              which(round(cl_BMI_data$all_feature_vectors[2,],4) %in% round(cal_feature_vec[2],4))),
                    intersect(which(round(cl_BMI_data$all_feature_vectors[3,],4) %in% round(cal_feature_vec[3],4)),
                              which(round(cl_BMI_data$all_feature_vectors[4,],4) %in% round(cal_feature_vec[4],4))))
                
                    if (length(intersection_index) != 0){
                      
                      correction_for_spatial_avg_index <- intersection_index - feature_index
                      spatial_avg_index <- spatial_avg_index - correction_for_spatial_avg_index
                      
                      spatial_avg_epoch <- cl_BMI_data$Overall_spatial_chan_avg[(spatial_avg_index - Classifier$smart_window_length*resamp_Fs):spatial_avg_index]
                      epoch_time <- seq(from = -1*Classifier$smart_window_length,to = 0,by = 1/resamp_Fs)
                      
                      cal_feature_vec <- t(c((spatial_avg_epoch[length(spatial_avg_epoch)] - spatial_avg_epoch[1])/(epoch_time[length(epoch_time)] - epoch_time[1]),
                                             min(spatial_avg_epoch),
                                             trapz(epoch_time,spatial_avg_epoch),
                                             sqrt((spatial_avg_epoch - Classifier$smart_Mu_move)%*%(inv(Classifier$smart_Cov_Mat))%*%(t(spatial_avg_epoch - Classifier$smart_Mu_move)))
                      ))
                      
                      cat("Cal: ",cal_feature_vec,"\t","Meas: ", toString(t(cl_BMI_data$all_feature_vectors[,feature_index])),"\t",
                          "spatial_index: ", toString(spatial_avg_index), "\n")
                                
                      # Segment processed_eeg and Overall_spatial_avg arrays according to the instant when intent was detected
                      # Segment duration = [-2.5s to +1s] w.r.t instant when intent is detected
                      Intent_EEG_epochs_trial <- array(data = NA, 
                                                       dim = c(length(Classifier$channels)+2,
                                                               (epoch_end_time-epoch_start_time)*resamp_Fs+1,
                                                               1))
                      Intent_EEG_epochs_trial[1,,1] <- seq(from = epoch_start_time,to = epoch_end_time,by = 1/resamp_Fs)
                      Intent_EEG_epochs_trial[2,,1] <- cl_BMI_data$Overall_spatial_chan_avg[(spatial_avg_index + epoch_start_time*resamp_Fs):
                                                                                              (spatial_avg_index + epoch_end_time*resamp_Fs)]
                                  
                      Intent_EEG_epochs_trial[3:dim(Intent_EEG_epochs_trial)[1],,1] <- 
                        cl_BMI_data$processed_eeg[,(spatial_avg_index + epoch_start_time*resamp_Fs):
                                                    (spatial_avg_index + epoch_end_time*resamp_Fs)]
                      
                      # Add row names
                      rownames(Intent_EEG_epochs_trial) <- c(list("time","Spatial_Avg"),as.list(Classifier$channels))
                      
                      # To check if processed EEG channels and Spatial Avg match
                      # plot(Intent_EEG_epochs_trial[1,,],Intent_EEG_epochs_trial[2,,])
                      # lines(Intent_EEG_epochs_trial[1,,],colMeans(Intent_EEG_epochs_trial[3:6,,1]),col="red")
                      
                      # Append (bind) array to global array
                      Intent_EEG_epochs_session <- abind(Intent_EEG_epochs_session,Intent_EEG_epochs_trial,along = 3)
                      
                      cl_trial_stats$feature_index[k] <- feature_index
                      cl_trial_stats$Corrected_spatial_chan_avg_index[k] <- spatial_avg_index # Write time stamp to .csv file
                      cl_trial_stats$Correction_applied_in_samples[k] <- correction_for_spatial_avg_index
                    }
                    else{
                      # Note: intersection_index is NULL then correction to spatial_chan_avg_index is 0
                      cat("Feature vector intersection not found!!\n")
                      Intent_EEG_epochs_trial <- array(0, 
                                                       dim = c(dim(cl_BMI_data$processed_eeg)[1]+2,
                                                               (epoch_end_time-epoch_start_time)*resamp_Fs+1,
                                                               1))
                      Intent_EEG_epochs_session <- abind(Intent_EEG_epochs_session,Intent_EEG_epochs_trial,along = 3)
                      cl_trial_stats$Corrected_spatial_chan_avg_index[k] <- 0
                      cl_trial_stats$Correction_applied_in_samples[k] <- 0
                      cl_trial_stats$feature_index[k] <- feature_index
                    }
                
              }
              else{
                Intent_EEG_epochs_trial <- array(0, 
                                                 dim = c(dim(cl_BMI_data$processed_eeg)[1]+2,
                                                         (epoch_end_time-epoch_start_time)*resamp_Fs+1,
                                                         1))
                Intent_EEG_epochs_session <- abind(Intent_EEG_epochs_session,Intent_EEG_epochs_trial,along = 3)
                cl_trial_stats$Corrected_spatial_chan_avg_index[k] <- 0
                cl_trial_stats$Correction_applied_in_samples[k] <- 0
                cl_trial_stats$feature_index[k] <- 0
              }
              
          }
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
        block_ind <- which(cl_session_stats$Block_number == closedloop_Block_num[bc])
        
        Total_num_of_trials <- c(Total_num_of_trials,length(which(cl_session_stats[block_ind,"Valid_or_catch"] == 1)))
        Successful_trials <- c(Successful_trials,length(intersect(which(cl_session_stats[block_ind,"Valid_or_catch"] == 1), 
                                              which(cl_session_stats[block_ind,"Intent_detected"] == 1))))
        Successful_EEG_EMG_trials <- c(Successful_EEG_EMG_trials,length(intersect(which(cl_session_stats[block_ind,"Valid_or_catch"] == 1), 
                                              which(cl_session_stats[block_ind,"EEG_EMG_decisions"] == 1))))
  
        Catch_trials <- c(Catch_trials,length(which(cl_session_stats[block_ind,"Valid_or_catch"] == 2)))
        Failed_Catch_trials <- c(Failed_Catch_trials,length(intersect(which(cl_session_stats[block_ind,"Valid_or_catch"] == 2), 
                                                     which(cl_session_stats[block_ind,"Intent_detected"] == 1))))
  
        cl_kinematic_params <- data.frame(
          #Block_number = rep_len(closedloop_Block_num[bc],length(all_stimulus_indices)),
          #Start_of_trial = round(all_stimulus_indices/2),
          #End_of_trial = round((cl_kinematics_data$Target_reached + cl_kinematics_data$Timeout)/2),
          #Valid_or_catch = cl_trial_stats$Valid_or_catch,
          Elbow_pos = resample(cl_kinematics_data$Elbow_position,500,1000),
          Elbow_vel = resample(cl_kinematics_data$Elbow_velocity,500,1000),
          time = resample(cl_kinematics_data$time,500,1000)
        ) # Used for creating Raster Plot !!
    
  
  }
  cat("Block Numbers:       ", closedloop_Block_num, "\n")
  cat("Successful Trials:   ", Successful_trials, "\n")
  #cat("Successful EEG- EMG Trials:   ", Successful_EEG_EMG_trials, "\n")
  cat("Total Num of trials: ", Total_num_of_trials, "\n")
  cat("Failed Catch trials: ", Failed_Catch_trials, "\n")
  cat("Total Catch Trials:  ", Catch_trials,"\n")
  cat("Are EEG_EMG_decisions and Intent_detected identifical? ", 
      identical(cl_session_stats$Intent_detected,cl_session_stats$EEG_EMG_decisions), "  ",
      which(cl_session_stats$Intent_detected != cl_session_stats$EEG_EMG_decisions), "\n")
  cat("Total blocks = ", length(closedloop_Block_num), 
      ", TPR = ", 100*sum(Successful_trials)/sum(Total_num_of_trials),
      "%, FPR = ", 100*sum(Failed_Catch_trials)/sum(Catch_trials), "%")
  
  if (use_simulated_closeloop_results == 1){
    save_filename <- paste(c(directory,sim_results_folderid,Subject_name,"_ses",toString(closeloop_Sess_num),"B_",simulation_mode,"_cloop_statistics.csv"),collapse = '')
  }else{
    save_filename <- paste(c(directory,folderid,Subject_name,"_ses",toString(closeloop_Sess_num),"_cloop_results.csv"),collapse = '')  
  }
  fileConn <- file(save_filename)
  cat("Block Numbers,", toString(closedloop_Block_num), "\n", #file = fileConn, sep = ',', append = T)
      "Successful Trials,", toString(Successful_trials), "\n", 
      "Total Num of trials,", toString(Total_num_of_trials), "\n",
      "Failed Catch trials,", toString(Failed_Catch_trials), "\n",
      "Total Catch Trials,", toString(Catch_trials),"\n\n",file = fileConn, sep = '', append = T)  
  close(fileConn)
  #fileConn <- file(save_filename)
  write.table(x = cl_session_stats,file = save_filename,append = T,col.names = T,row.names = F, sep = ',')
  #close(fileConn)
  
  save_matfile <- paste(c(directory,folderid,Subject_name,"_ses",toString(closeloop_Sess_num),"_cloop_eeg_epochs.mat"),collapse = '')
  writeMat(save_matfile,Intent_EEG_epochs_session = Intent_EEG_epochs_session, append = FALSE)
  
  # Used for generating raster plot - Raster plot for paper
  #save_matfile1 <- paste(c(directory,folderid,Subject_name,"_ses",toString(closeloop_Sess_num),"_block",toString(closedloop_Block_num),"_cloop_kinematic_params.mat"),collapse = '')
  #writeMat(save_matfile1,cl_kinematic_params = cl_kinematic_params, append = FALSE)
  cl_session_stats
}



