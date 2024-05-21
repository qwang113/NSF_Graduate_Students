rm(list = ls())
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
# for (out.rm in 0:1) {
#   if(out.rm)
#   {
#     # Remove outliers
#     nh = 200
#     osh_res <- read.csv(paste("D:/77/UCSC/study/Research/temp/NSF_dat/ESN_res_",nh, ".csv", sep = ""))
#     annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
#     nsf_wide_car <- nsf_wide_car[-annoying_cases[1:100],]
#   }
  
  wide_y <- nsf_wide_car[,-c(1:9, ncol(nsf_wide_car))]
  # 
  # dummy_car <- model.matrix(~nsf_wide_car$HD2021.Carnegie.Classification.2021..Graduate.Instructional.Program - 1)
  # # dummy_school <- model.matrix(~nsf_wide$UNITID - 1)
  # # dummy_matrix <- cbind(dummy_school, dummy_car)
  # dummy_matrix <- dummy_car
  
  osh_pred <- matrix(NA, nrow = nrow(wide_y), ncol = length(2012:2021))
  
  #All schools pooled together
  # for (year in 2012:2021) {
  #   long_yt <- unlist(as.vector(wide_y[,2:(year-1972)]))
  #   long_yt_lag <- log(unlist(as.vector(wide_y[,1:(year-1972-1)]))+1)
  #   obs_dat <- data.frame(cbind(long_yt, long_yt_lag))
  #   curr_model <- glm(long_yt~., family = poisson(link = "log"), data = obs_dat)
  #   pred_name <- colnames(obs_dat)[2]
  #   pred_mat <- data.frame(long_yt_lag = log(wide_y[,year-1972]+1)) 
  #   osh_pred[,year-2011] <- predict(curr_model, newdata = pred_mat, type = "response")
  # }
  # 
  # osh_res <- wide_y[,41:50] - osh_pred
  # mean(unlist(as.vector(osh_res^2)))
  
  #Individually regression: AR1
  for (year in 2012:2021) {
    for (school in 1:nrow(wide_y)) {
      print(paste("now doing year", year, "school", school))
      prev_obs <- unlist(wide_y[school,2:(year-1972)]) 
      lag_obs <- log(unlist(wide_y[school, 1:(year-1972-1)])+1)
      model <- glm(prev_obs ~ lag_obs, family = poisson(link = "log"))
      osh_pred[school, year-2011] <- predict(model, newdata = data.frame("lag_obs"=unlist(log(wide_y[school, year-1972]+1))),
                                             type = "response")
    }
  }
  
  osh_res <- wide_y[,41:50] - osh_pred
  mean(unlist(osh_res)^2)
  
  # if(out.rm == 0){
    # # print("Full dataset")
    # write.csv( as.data.frame(osh_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv", row.names = FALSE)
    # write.csv( as.data.frame(osh_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res.csv", row.names = FALSE)
  # }else{
  #   print("Not Full dataset")
  #   write.csv( as.data.frame(osh_pred), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred_outrm.csv", row.names = FALSE)
  #   write.csv( as.data.frame(osh_res), "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_res_outrm.csv", row.names = FALSE)
  # }
  
# }


# File path: "D:/77/UCSC/study/Research/temp/NSF_dat/pois_autoreg_pred.csv"
# 
# mean(unlist(as.vector(osh_res^2)))
# var(unlist(as.vector( wide_y[,41:50])))
# source(here::here("utility_functions.R"))
# annoying_cases <- order(apply(unlist(as.matrix(osh_res^2)), 1, mean), decreasing = TRUE)
# ordered_mse <- rowMeans(unlist(as.matrix(osh_res^2)))[annoying_cases]
# how_annoying <- 1799
# show_obs(annoying_cases[how_annoying])
# show_pred(annoying_cases[how_annoying])
# ordered_mse[how_annoying]
# sqrt(ordered_mse[how_annoying])


# Delete some bad observations
# CNN with/without
# Visualilze the prediction trace of different models
# Median/Median sqared error for ESN, PCA, PoisAutoReg with/without special caeses





