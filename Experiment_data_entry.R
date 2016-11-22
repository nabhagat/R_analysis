# Program for automatically enlisting the files generated during BMI_MAHI Clinical trial
################ Libraries
library(signal)
library(R.matlab)
library(pracma)
library(abind)
################ Main Program ########################
directory <- "D:/NRI_Project_Data/Clinical_study_Data/"
Subject_name <- "S9011"            #1 
Subject_velocity_threshold <- 1.16 # Unit - degrees/sec
Subject_cap_size <- 56 # Unit - cm
Subject_impaired_side <- "left"
Calibration_Cond_num  <- c(1)      # 1 - User-driven (UD), 3 - User-triggered (UT). Not - If both UD and UT were used then enter c(1,3)
BMIClassifier_Training_Sess_num <- c(2)
BMIClassifier_Training_Block_num <- c(170)
EMG_channel_nos <- c(42,41,51,17,45,46,55,22)

block_wise_file_extensions <- c(".eeg",".vhdr",".vmrk",           # BrainVision Recorder
                               ".mat",                            # MATLAB
                               ".set",".fdt",                     # EEGLAB Toolbox
                               ".txt",                            # Kinematics - MAHI Exo 
                               ".avi"                             # Video files
                              )
session_wise_file_extensions <- c(".txt",                         # EEG electrode impedance
                                  ".elp",                         # Electrode location file - CAPTRACK digitizer
                                  ".csv",                         # Closed-loop session results 
                                  ".mat"                          # Closed-loop EEG epochs
                                  )
# Finds the Session numbers for a subject
setwd(paste(c(directory,"Subject_",Subject_name,"/"),collapse = ''))
all_sessions_directory_names <- dir(path = ".",pattern = paste(c(Subject_name,"_Session"),collapse = ''))
Session_nos <- sort(as.numeric(gsub(pattern = paste(c(Subject_name,"_Session"),collapse = ''),"",x = all_sessions_directory_names)))

header_summary_file <- data.frame(col1 = c("Subject:","Total sessions","Impaired side",
                                           "EEG Cap size","Velocity threshold (rad/s)","EMG_channel_nos","",""),
                                  col2 = c(Subject_name,length(Session_nos),Subject_impaired_side,
                                           Subject_cap_size,Subject_velocity_threshold,paste(EMG_channel_nos,collapse = ' '),"","")
)

save_filename <- paste(c(directory,"Subject_",Subject_name,"/",Subject_name,"_experiment_summary_",
                         format(Sys.time(), "%m-%d-%y_%H-%M-%S"),".csv"),collapse = '')
save_results_file <- paste(c(directory,"Subject_",Subject_name,"/",Subject_name,"_session_wise_results_",
                             format(Sys.time(), "%m-%d-%y_%H-%M-%S"),".csv"),collapse = '')
save_results_file_header <- 1
  
#if (file.exists(save_filename)){
#  file.remove(save_filename)
#}

write.table(x = header_summary_file,file = save_filename,append = T,col.names = F, row.names = F, sep = ',')

for (ses_num in seq_along(Session_nos)){
  setwd(paste(c(directory,"Subject_",Subject_name,"/",Subject_name,"_Session",Session_nos[ses_num]),collapse = ''))
  
  
  if ((Session_nos[ses_num] == 1) || (Session_nos[ses_num] == 2)){          
  #if ((Session_nos[ses_num] == 1) || (Session_nos[ses_num] == 2) || (Session_nos[ses_num] == 3)){
    # Different naming convention and add EEGLAB dataset names
    # Based on .vhdr file naming convention
    begin_filename_identifier <- paste(c(Subject_name,"_ses",Session_nos[ses_num],"_cond", Calibration_Cond_num, "_block0"),collapse = '')
    end_filename_identifier <- ".vhdr"
    # next commented on 8-31-2016
  }
  else{
    # Use closed-loop naming format
    # Based on .vhdr file naming convention
    begin_filename_identifier <- paste(c(Subject_name,"_ses",Session_nos[ses_num],"_closeloop_block0"),collapse = '')
    end_filename_identifier <- ".vhdr"
    
    # Based on kinematics naming convention
    #begin_filename_identifier <- paste(c(Subject_name,"_CLses",Session_nos[ses_num],"_block"),collapse = '')
    #end_filename_identifier <- "_kinematics.txt
  }
  
  
  # Finds the block numbers within each session
  files_with_vhdr_extension <- dir(path = ".",pattern = end_filename_identifier,recursive = FALSE)
  #files_with_vhdr_extension <- files_with_vhdr_extension[grep(pattern = begin_filename_identifier,"",x = files_with_vhdr_extension)]
  Block_nos <- sort(as.numeric(gsub(pattern = end_filename_identifier,"", x = 
                                      gsub(pattern = begin_filename_identifier,"",x = files_with_vhdr_extension))))
  
  # Call closedloopdata_analysis.R to analyze data for each theraoy session only
  if (Session_nos[ses_num] > 2){
    # Commented on 9-6-2016. Instead directly read .csv file with session results
    # session_results <- analyze_closedloop_session_data(directory = directory,
    #                                                   Subject_name = Subject_name,
    #                                                   closeloop_Sess_num = Session_nos[ses_num],
    #                                                   closedloop_Block_num = Block_nos[Block_nos > 0],    #Block0 is for robotic assessmenet - so ignore
    #                                                   Subject_velocity_threshold,
    #                                                   Calibration_Cond_num,
    #                                                   BMIClassifier_Training_Sess_num,
    #                                                   BMIClassifier_Training_Block_num,
    #                                                   Subject_impaired_side
    #                                                   )
    
    session_results <- read.csv(paste(c(Subject_name,"_ses",toString(Session_nos[ses_num]),"_cloop_results.csv"),collapse = ''),
                                header = TRUE, skip = 6)
    
    
    if (save_results_file_header == 1){
      write.table(x = session_results,file = save_results_file,append = T,col.names = T, row.names = F, sep = ',')
      save_results_file_header <- 0
    }
    else{
      write.table(x = session_results,file = save_results_file,append = T,col.names = F, row.names = F, sep = ',')
    }
  }

  
  
  # Make table to save filenames within a session
  for (sc in seq_along(session_wise_file_extensions)){
    loop_ext <- session_wise_file_extensions[sc]
    
    # .elp files
    if (loop_ext == ".elp"){
      if (length(file.exists(dir(".",pattern = ".elp")))){
        electrode_location_filename <- dir(".",pattern = ".elp")
      }else{
        electrode_location_filename <- "NA"
      }
    }
    
    # .txt files for Electrode Impedance
    if (loop_ext == ".txt"){
      impedance_filenames <- as.character()
      if (file.exists(dir(".",pattern = "*impedance*"))){
        impedance_files <- dir(".",pattern = "*impedance*")
        if (length(grep("start",impedance_files))!= 0){
          impedance_filenames[1] <- impedance_files[grep("start",impedance_files)] 
        }
        else {
          impedance_filenames[1] <- "NA"
        }
        if (length(grep("break",impedance_files))!= 0){
          impedance_filenames[2] <- impedance_files[grep("break",impedance_files)] 
        }
        else {
          impedance_filenames[2] <- "NA"
        }  
        if (length(grep("stop",impedance_files))!= 0){
          impedance_filenames[3] <- impedance_files[grep("stop",impedance_files)] 
        }
        else if (length(grep("end",impedance_files))!= 0){
          impedance_filenames[3] <- impedance_files[grep("end",impedance_files)] 
        }
        else {
          impedance_filenames[3] <- "NA"
        }  
      }
      else{
        impedance_filenames <- "NA"
      }
    }
    
    if (loop_ext == ".csv"){
      if(length(file.exists(dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_cloop_results.csv"),collapse = '')))) != 0){
        session_results_filename <- dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_cloop_results.csv"),collapse = ''))
      }
      else{
        session_results_filename <- "NA"
      }
      
      if(length(file.exists(dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_closeloop_emg_thresholds.csv"),collapse = '')))) != 0){
        emg_thresholds_filename <- dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_closeloop_emg_thresholds.csv"),collapse = ''))
      }
      else{
        emg_thresholds_filename <- "NA"
      }
      
    }
    
    if (loop_ext == ".mat"){
      if ((Session_nos[ses_num] == 1) || (Session_nos[ses_num] == 2)){
        if(length(file.exists(dir(".",pattern = paste(c("*_performance_optimized_conventional_smart",loop_ext),collapse = '')))) != 0){
          BMIclassifier_filename <- dir(".",pattern = paste(c("*_performance_optimized_conventional_smart",loop_ext),collapse = ''))
        }
        else{
          BMIclassifier_filename <- "NA"
        }
      }
      else{
            if(length(file.exists(dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_cloop_eeg_epochs.mat"),collapse = '')))) != 0){
              eeg_epochs_filename <- dir(".",pattern = paste(c("_ses",Session_nos[ses_num],"_cloop_eeg_epochs.mat"),collapse = ''))
            }
            else{
                  eeg_epochs_filename <- "NA"
            }
      }
    }
  }
  
  if ((Session_nos[ses_num] == 1) || (Session_nos[ses_num] == 2)){
      Session_specific_files <- data.frame(col1 = c("","Session No:","Total blocks","Electrode locations (.elp)",
                                                    "Electrode Impedance (.txt)"," "," ",
                                                    "BMI Classifier (.mat)", ""),
                                           col2 = c("",Session_nos[ses_num],length(Block_nos),electrode_location_filename,
                                                    impedance_filenames,
                                                    BMIclassifier_filename, ""))        
      
      # Make table to save filenames for each block within a session
      Block_wise_files <- data.frame(File_Types = block_wise_file_extensions)
      #Block_wise_files <- lapply(Block_wise_files,as.character)
  
      for (bc in seq_along(Block_nos)){
        # Add a new column for new block
        Block_wise_files[,paste(c("Block",Block_nos[bc]),collapse = '')] <- "NA"
        
        for (bc_ext in seq_along(block_wise_file_extensions)){
          loop_ext <- block_wise_file_extensions[bc_ext]
        
          # .eeg, .vhdr, .vmrk files
          if (loop_ext == ".eeg" || loop_ext == ".vhdr" || loop_ext == ".vmrk"){
            if (Block_nos[bc] <= 9){
              check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num, "_block000",Block_nos[bc],loop_ext),collapse = ''))
            }
            else {
              check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num, "_block00",Block_nos[bc],loop_ext),collapse = ''))
              }  
          }
          
          # # .mat files
          if (loop_ext == ".mat") {
            # Include wildchars
            check_file_exists <- character(0) #dir(".",paste(c("*_performance_optimized_conventional_smart",loop_ext),collapse = ''))
          }
          
          # .txt files - Kinematics
          if (loop_ext == ".txt") {
            # Include wildchars
            check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num,"_block",Block_nos[bc],"_kinematics*"),collapse = ''))
          }
          
          # .avi files
          if (loop_ext == ".avi" || loop_ext == ".mp4") {
            # Include wildchars
            check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num,"_block",Block_nos[bc],"_video*"),collapse = ''))
          }
          
          # .fdt, .set files
          if (loop_ext == ".set" || loop_ext == ".fdt") {
           check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num,"_block",Block_nos[bc],"_eeg_raw",loop_ext),collapse = ''))
          }
          
          if(length(file.exists(check_file_exists)) != 0){
            Block_wise_files[grep(loop_ext,Block_wise_files$File_Types),paste(c("Block",Block_nos[bc]),collapse = '')] <- check_file_exists
          }
        
        }
      }
      
      if(Session_nos[ses_num] == 2){
        # remove the extra file extensions after you have completed processing session 2  
        block_wise_file_extensions <- block_wise_file_extensions[c(-5,-6)] # Remove ".set" and ".fdt" extensions as no longer used - 9/19/2016
      }
  }
  else{
      Session_specific_files <- data.frame(col1 = c("","Session No:","Total blocks","Electrode locations (.elp)",
                                                     "Electrode Impedance (.txt)"," "," ",
                                                     "Closed-loop results (.csv)","Closed-loop EEG epochs (.mat)",
                                                     "Blockwise EMG Thresholds (.csv)", " "),
                                            col2 = c("",Session_nos[ses_num],length(Block_nos),electrode_location_filename,
                                                     impedance_filenames,
                                                     session_results_filename,eeg_epochs_filename,
                                                     emg_thresholds_filename, ""))        
      
      # Make table to save filenames for each block within a session
      Block_wise_files <- data.frame(File_Types = block_wise_file_extensions)
      #Block_wise_files <- lapply(Block_wise_files,as.character)
  
      for (bc in seq_along(Block_nos)){
        # Add a new column for new block
        Block_wise_files[,paste(c("Block",Block_nos[bc]),collapse = '')] <- "NA"
        
        for (bc_ext in seq_along(block_wise_file_extensions)){
          loop_ext <- block_wise_file_extensions[bc_ext]
        
          # .eeg, .vhdr, .vmrk files
          if (loop_ext == ".eeg" || loop_ext == ".vhdr" || loop_ext == ".vmrk"){
            if (Block_nos[bc] <= 9){
              check_file_exists <- paste(c(Subject_name,"_ses",ses_num,"_closeloop_block000",Block_nos[bc],loop_ext),collapse = '') 
            }
            else {
              check_file_exists <- paste(c(Subject_name,"_ses",ses_num,"_closeloop_block00",Block_nos[bc],loop_ext),collapse = '')             
              }  
          }
          
          # .mat files
          if (loop_ext == ".mat") {
            # Include wildchars
            check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_block",Block_nos[bc],"_closeloop_results*"),collapse = ''))
          }
          
          # .txt files - Kinematics
          if (loop_ext == ".txt") {
            # Include wildchars
            if (Block_nos[bc] == 0){
              check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_cond", Calibration_Cond_num,"_block",Block_nos[bc],"_kinematics*"),collapse = '')) 
            }
            else{
              check_file_exists <- dir(".",paste(c(Subject_name,"_CLses",ses_num,"_block",Block_nos[bc],"_kinematics*"),collapse = '')) 
            }
          }
          
          # .avi files
          if (loop_ext == ".avi" || loop_ext == ".mp4") {
            # Include wildchars
            check_file_exists <- dir(".",paste(c(Subject_name,"_ses",ses_num,"_block",Block_nos[bc],"_closeloop_video*"),collapse = ''))
          }
          
          # .fdt, .set files
          if (loop_ext == ".set" || loop_ext == ".fdt") {
            check_file_exists <- paste(c(Subject_name,"_ses",ses_num,"_block",Block_nos[bc],"_closeloop_eeglab",loop_ext),collapse = '')
          }
          
          if(length(file.exists(check_file_exists)) != 0){
            Block_wise_files[grep(loop_ext,Block_wise_files$File_Types),paste(c("Block",Block_nos[bc]),collapse = '')] <- check_file_exists
          }
        }
      }
  }
  
  #fileConn <- file(save_filename)
  #cat("\n\n",file = fileConn, sep = '', append = T)  
  #close(fileConn)
  write.table(x = Session_specific_files,file = save_filename,append = T,col.names = F, row.names = F, sep = ',')
  #fileConn <- file(save_filename)
  #cat("\n\n",file = fileConn, sep = '', append = T)  
  #close(fileConn)
  write.table(x = Block_wise_files,file = save_filename,append = T,col.names = T, row.names = F, sep = ',')
}
