nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
cases <- c(126614106, 228778109, 204796109, 196088907, 147767906)
special_dat <- nsf_wide_car[which(nsf_wide_car$ID %in% cases),-c(1:9, ncol(nsf_wide_car))]
pois_coef <- array(NA, dim = c(5,10,2))
pois_pred <- matrix(NA, nrow = 5, ncol = 10)
esn_pred <- matrix(NA, nrow = 5, ncol = 10)
for (year in 2012:2021) {
  for (i in 1:5) {
    obs_i <- unlist(special_dat[i, 2:(year-1972)])
    lag_i <- unlist(special_dat[i, 1:(year-1972-1)])
    pois_model <- glm(obs_i~lag_i, family = poisson(link = "log"))
    pois_coef[i,year-2011,] <- pois_model$coefficients
    pois_pred[i,year-2011] <- predict(pois_model,data.frame("lag_i" = obs_i[length(obs_i)]), type = "response")
  }
}


######################################ESN Pooled#################################################


a <- 0.1
nh = 50
nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
nsf_wide_car <- nsf_wide_car[which(nsf_wide_car$ID %in% cases),]
one_step_ahead_pred_y <- matrix(NA, nrow = nrow(nsf_wide_car), ncol = length(2012:2021))
for (year in 2012:2021) {
  #Initialize
  nu <- 1
  W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
  ar_col <- matrix(runif(nh,-a,a), nrow = 1)
  lambda_scale <- max(abs(eigen(W)$values))
  curr_H <- matrix(tanh(nsf_wide_car[,10]), nrow = 5)%*% ar_col
  Y <- nsf_wide_car[,10]
  pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
  print("Calculating Recurrent H Matrix. . .")
  for (i in 2:(year-1972+1)) {
    setTxtProgressBar(pb,i)
    new_H <- apply( 
      # nu/lambda_scale*
      curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W 
      + log(nsf_wide_car[,i+8]+1)%*%ar_col
      , c(1,2), tanh)
    Y <- c(Y, nsf_wide_car[,i+9])
    curr_H <- rbind(curr_H, new_H)
  }
  colnames(curr_H) <- paste("node", 1:ncol(curr_H))
  obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
  pred_H <- curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
  print(paste("Now doing year",year))
  years_before <- year - 1972
  obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
  one_step_ahead_model <- glm(obs_y~., family = poisson(link="log"), data = data.frame(cbind(obs_y, obs_H)),
                              control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
  one_step_ahead_pred_y[,year-2011] <- predict(one_step_ahead_model, newdata = data.frame(pred_H), type = "response")
}

esn_pre_pooled <- one_step_ahead_pred_y
one_step_ahead_res <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - one_step_ahead_pred_y
mean(unlist(as.vector(one_step_ahead_res))^2)
var(unlist(as.vector(nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))])))





######################################ESN Separate#################################################

one_step_ahead_pred_y <- matrix(NA, nrow = length(cases), ncol = length(2012:2021))
a <- 0.1
nh = 10
for (cases_i in 1:5) {
  nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
  nsf_wide_car <- nsf_wide_car[which(nsf_wide_car$ID == cases[cases_i]),]
  for (year in 2012:2021) {
    #Initialize
    nu <- 1
    W <- matrix(runif(nh^2, -a,a), nh, nh) # Recurrent weight matrix, handle the output from last hidden unit
    ar_col <- matrix(runif(nh,-a,a), nrow = 1)
    lambda_scale <- max(abs(eigen(W)$values))
    curr_H <- tanh(matrix(nsf_wide_car[,10])%*% ar_col)
    Y <- nsf_wide_car[i,10]
    pb <- txtProgressBar(min = 1, max = length(2:(year-1972+1)), style = 3)
    print("Calculating Recurrent H Matrix. . .")
    for (i in 2:(year-1972+1)) {
      setTxtProgressBar(pb,i)
      new_H <- apply( 
        # nu/lambda_scale*
        curr_H[(nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H),]%*%W 
        + log(nsf_wide_car[,i+8]+1)%*%ar_col
        , c(1,2), tanh)
      Y <- c(Y, nsf_wide_car[,i+9])
      curr_H <- rbind(curr_H, new_H)
    }
    colnames(curr_H) <- paste("node", 1:ncol(curr_H))
    obs_H <- curr_H[-c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),]
    pred_H <- matrix(curr_H[c((nrow(curr_H)-nrow(nsf_wide_car)+1):nrow(curr_H)),], nrow = 1)
    print(paste("Now doing year",year))
    years_before <- year - 1972
    obs_y <- Y[1:(years_before*nrow(nsf_wide_car))]
    one_step_ahead_model <- glm(obs_y~., family = poisson(link="log"), data = data.frame(cbind(obs_y, obs_H)),
                                control = glm.control(epsilon = 1e-8, maxit = 10000000, trace = TRUE))
    colnames(pred_H) <- colnames(obs_H)
    one_step_ahead_pred_y[cases_i,year-2011] <- predict(one_step_ahead_model, newdata = data.frame(pred_H), type = "response")
  }
}


esn_pre_sep <- one_step_ahead_pred_y

nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
nsf_wide_car <- nsf_wide_car[which(nsf_wide_car$ID %in% cases),]


pois_res <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - pois_pred
esn_res_pooled <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - esn_pre_pooled
esn_res_sep <- nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))] - esn_pre_sep


print(apply(pois_res^2,1, mean))
print(apply(esn_res_pooled^2,1, mean))
print(apply(esn_res_sep^2,1, mean))


mean(unlist(as.vector(one_step_ahead_res))^2)



var(unlist(as.vector(nsf_wide_car[,c((2012-1972+10):(ncol(nsf_wide_car)-1))])))





case_study_show <- function(curr_id = 1){
  
  library(ggplot2)
  
  
  nsf_wide_car <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_wide_car.csv")
  nsf_long <- read.csv("D:/77/UCSC/study/Research/temp/NSF_dat/nsf_final_long.csv")
  
  nsf_wide_car <- nsf_wide_car[which(nsf_wide_car$ID %in% cases),]
  
  curr_y <- unlist(nsf_wide_car[curr_id,-c(1:9,ncol(nsf_wide_car))])
  
  colorss <- c("Poisson" = "purple", "ESN_pooled" = "green", "ESN_sep" = "orange")
  obs_trace <-
    ggplot() +
    geom_line(aes(x = 1972:2021, y = curr_y), col = "blue", linewidth = 2) +
    geom_point(aes(x = 1972:2021, y = curr_y), col = "red", size = 2) +
    labs(x = "Year", y = "Count", title = paste(nsf_long$col_sch[min(which(nsf_long$ID == nsf_wide_car$ID[curr_id]))]),
         # ifelse(out.rm, "(Outlier Removed)","Full Dataset")), 
         color = "Legend") +
    theme(plot.title = element_text(hjust = 0.5)) +
    geom_line(aes(x = 2012:2021, y = as.numeric(pois_pred[curr_id,]), color = "Poisson"), lwd = 1) +
    geom_point(aes(x = 2012:2021, y = as.numeric(pois_pred[curr_id,]), color = "Poisson"), size = 2) +
    geom_line(aes(x = 2012:2021, y = as.numeric(esn_pre_pooled[curr_id,]), color = "ESN_pooled"), lwd = 1) +
    geom_point(aes(x = 2012:2021, y = as.numeric(esn_pre_pooled[curr_id,]), color = "ESN_pooled"), size = 2) +
    # geom_line(aes(x = 2012:2021, y = as.numeric(esn_pre_sep[curr_id,]), color = "ESN_sep"), lwd = 1) +
    # geom_point(aes(x = 2012:2021, y = as.numeric(esn_pre_sep[curr_id,]), color = "ESN_sep"), size = 2) +
    geom_vline(xintercept = 2012, color = "black", lwd = 1, linetype= "dashed")
  
  return(obs_trace)
}



cowplot::plot_grid( case_study_show(1),case_study_show(2),
                    case_study_show(3),case_study_show(4),case_study_show(5),ncol = 1)


