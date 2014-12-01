# Code to fit linear regression to accuracy and error_min for all subjects 
# and determine statistical significance of slope of linear regression

library('R.matlab')

# Read MAT variables
acc_per_session <- as.data.frame(readMat("C:/NRI_BMI_Mahi_Project_files/All_Subjects//acc_per_session.mat", fixNames = FALSE))
err_per_session <- as.data.frame(readMat("C:/NRI_BMI_Mahi_Project_files/All_Subjects//err_per_session.mat", fixNames = FALSE))

rownames(acc_per_session) <- c("BNBO","ERWS","JF","LSGR","PLSH")
rownames(err_per_session) <- c("BNBO","ERWS","JF","LSGR","PLSH")

acc_per_session["JF",c(1,2)] <- NA
acc_per_session["ERWS",1] <- NA
err_per_session["JF",c(1,2)] <- NA
err_per_session["ERWS",1] <- NA
days <- c(3,4,5)

# help(summary.lm)

for(i in 1:5){
  acc_regression <- lm(100*as.numeric(acc_per_session[i,]) ~ days,na.action = na.omit)  
  print(rownames(acc_per_session[i,]))
  print(summary(acc_regression))
}

print("######################################################################")

for(i in 1:5){
  err_regression <- lm(as.numeric(err_per_session[i,]) ~ days,na.action = na.omit)  
  print(rownames(err_per_session[i,]))
  print(summary(err_regression))
}

