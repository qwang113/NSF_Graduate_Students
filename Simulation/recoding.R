dat <- read.csv(here::here("nsf_final_wide_car.csv"))
y_mat <- as.matrix(dat[,-c(1:9, ncol(dat))])

nh <- 50
a <- 0.001
nu = 0.5

y_pred <- matrix(NA, nrow = nrow(y_mat), ncol = length(41:50))

for (curr_year in 41:50) {
  W <- matrix(runif(nh^2, -a, a), nrow = nh, ncol = nh)
  lam <- max(abs(eigen(W)$values))
  curr_x <- matrix(c(1, rep(0, nrow(y_mat))), ncol = 1)
  nx <- length(curr_x)
  U <- matrix(runif(nh*nx, -a, a), nrow = nh, ncol = nx)
  curr_ux <- U %*% curr_x
  curr_H <- tanh(curr_ux)
  all_H <- curr_H
  
  for (year in 2:(curr_year)) {
    curr_x <- matrix(c(1, log(wide_y[,year-1]+1)), ncol = 1)
    curr_ux <- U %*% curr_x
    new_H <- tanh(nu/lam * W %*% curr_H + curr_ux)
    all_H <- cbind(all_H, new_H)
    curr_H <- new_H
  }
  
  # h_pca_res <- prcomp(t(all_H))
  # num_use <- min(which( cumsum(h_pca_res$sdev^2)/sum(h_pca_res$sdev^2) >= 0.99 ))
  # h_pc <- h_pca_res$x[,1:num_use]
  
  
  for (school in 1:nrow(y_mat)) {
    print(paste("Now doing year",curr_year, "school",school))
    obs_y <- matrix(y_mat[school, 1:(curr_year-1)], ncol = 1)
    obs_h <- t(all_H[,-1])
    # obs_h_pc <- h_pc[1:(curr_year-1),]
    # curr_dat <- data.frame(cbind(obs_y, obs_h_pc))
    colnames(curr_dat)[1] <- "y"
    curr_model <- glm(y ~., data = curr_dat, family = poisson(link = "log"))
    # curr_model_pen.cv <- cv.glmnet(x = t(obs_h[,-1]), y = obs_y, family = poisson(link = "log"), alpha = 0)
    # plot(curr_model_pen.cv)
    # curr_model_pen <- glmnet(x = t(obs_h[,-1]), y = obs_y, family = poisson(link = "log"), alpha = 0,
                             # lambda = curr_model_pen.cv$lambda.min)
    # sum(coef(curr_model_pen)^2)
    # pred_h_pc <- data.frame(matrix(h_pc[curr_year,], nrow = 1))
    # colnames(pred_h_pc) <- colnames(curr_dat)[-1]
    
    pred_h <- data.frame(matrix(curr_H, nrow = 1))
    colnames(pred_h) <- colnames(curr_dat)[-1]

    # pred_y[school, curr_year-40] <- predict(curr_model, pred_h_pc, type = "response")
    # pred_y[school, curr_year-40] <- predict(curr_model_pen, newx = t(new_H), type = "response)
    pred_y[school, curr_year-40] <- predict(curr_model, pred_h, type = "response")
  }
}

pred_mse <- mean((y_mat[,41:50] - pred_y)^2)
pred_mse
var(as.vector(y_mat[,41:50]))
