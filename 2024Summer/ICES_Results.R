library(tidyverse)
library(glmnet)
library(tscount)


schools <- read_csv(here::here("nsf_final_wide_car.csv")) %>% filter(state=="CA")
schoolsM <- as.matrix(schools[-c(81,97),10:59])
schoolsM <- as.matrix(schools[,10:59])


Pois_ESN <- function(Xin, Yin, Xpred, nh=120, lambda=0.1, nu=0.35, rv=0.01, aw=0.1, pw=0.1, au=0.1, pu=0.1, reps=1){
  Preds <- array(NA, dim=c(nrow(Xpred), ncol(Yin), reps))
  for(r in 1:reps){
    ## Fit
    p <- ncol(Xin)
    W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,pw), nrow=nh)
    W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
    U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,pu), nrow=nh)
    H <- matrix(NA, nrow=nrow(Xin), ncol=nh)
    tmp <- U %*% Xin[1,]
    H[1,] <- tanh(tmp) 
    for(i in 2:nrow(H)){
      tmp <- W%*%H[i-1,] + U%*%Xin[i,]
      H[i,] <- tanh(tmp)
    }
    Hnew <- H

    
    Hpred <- matrix(NA, nrow=nrow(Xpred), ncol=nh)
    tmp <- W%*%H[nrow(H), ] + U%*%Xpred[1,]
    Hpred[1,] <- tanh(tmp)
    if(nrow(Hpred)>1){
      for(i in 2:nrow(Hpred)){
        tmp <- W%*%Hpred[i-1, ] + U%*%Xpred[i,]
        Hpred[i,] <- tanh(tmp)
      }
    }
    for(j in 1:ncol(Yin)){
      lasso <- glmnet(Hnew, Yin[,j], family="poisson", lambda=lambda)
      V <- lasso$beta
      Ypred <- predict(lasso, Hpred, type="response")
      Preds[,j,r] <- Ypred
      #print(j)
    }
    
  }
  return(Preds)
}





Pois_ESN_SS <- function(Xin, Yin, Xpred, nh=120, lambda=0.5, nu=0.35, rv=0.01, aw=.1, pw=.1, au=.1, pu=.1, reps=1){
  Preds <- array(NA, dim=c(nrow(Xpred), ncol(Yin), reps))
  for(r in 1:reps){
    ## Fit
    p <- 2
    W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,pw), nrow=nh)
    W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
    U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,pu), nrow=nh)
    H <- matrix(NA, nrow=nrow(Xin), ncol=nh)
    for(j in 1:ncol(Yin)){
      tmp <- U %*% c(1,Xin[1,j])
      H[1,] <- tanh(tmp) 
      for(i in 2:nrow(H)){
        tmp <- W%*%H[i-1,] + U%*%c(1,Xin[i,j])
        H[i,] <- tanh(tmp)
      }
      Hnew <- H
      
      Hpred <- matrix(NA, nrow=nrow(Xpred), ncol=nh)
      tmp <- W%*%H[nrow(H), ] + U%*%c(1,Xpred[1,j])
      Hpred[1,] <- tanh(tmp)
      if(nrow(Hpred)>1){
        for(i in 2:nrow(Hpred)){
          tmp <- W%*%Hpred[i-1, ] + U%*%c(1,Xpred[i,j])
          Hpred[i,] <- tanh(tmp)
        }
      }
      
      lasso <- glmnet(Hnew, Yin[,j], family="poisson", lambda=lambda)
      Ypred <- predict(lasso, Hpred, type="response")
      Preds[,j,r] <- Ypred
      #print(j)
    }
    print(r)
  }
  
  return(Preds)
}



rCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1)){
  ## This function will simulate from the cMLG distribution
  m <- length(kappa)
  w <- log(rgamma(m, shape=alpha, rate=kappa))
  return(as.numeric(solve(t(H)%*%H)%*%t(H)%*%w))
}
rTruncCMLG <- function(H=matrix(rnorm(6),3), alpha=c(1,1,1), kappa=c(1,1,1), cut=0){
  ## This function simulates from a cMLG distribution truncated from below
  repeat {
    temp <- rCMLG(H=H, alpha=alpha, kappa=kappa)
    if(all(temp > cut)) break
  }
  return(temp)
}


bPois <- function(X, Y, alpha=1000, w=1000, p=1000, sigb=.5, iter=10, burn=5, eps=1){
  r <- ncol(X)
  n <- length(Y)
  aBeta <- c((Y+eps), rep(alpha,r))
  # aK <- c(rep(alpha, r), w)
  # kK <- c(rep(alpha, r), p)
  betaOut <- matrix(NA, nrow=iter, ncol=r)
  beta <- rep(1,p)
  sigKinv <- .1
  for(i in 1:iter){
    # Regression coefficients
    Hbeta <- rbind(X,  alpha^(-0.5)*(sigKinv)*Diagonal(r))
    Kbeta <- c(rep(1,n), rep(alpha,r))
    temp <- rCMLG(H=Hbeta, alpha=aBeta, kappa=Kbeta)
    beta <- betaOut[i,] <- temp
  }
  return(list(Beta=betaOut[-c(1:burn),]))
}



Pois_ESN_Bayes <- function(Xin, Yin, Xpred, nh=120, lambda=0.5, nu=0.35, rv=0.01, aw=.1, pw=.1, au=.1, pu=.1, reps=1){
  Preds <- array(NA, dim=c(nrow(Xpred), ncol(Yin), reps))
  for(r in 1:reps){
    ## Fit
    p <- 2
    W <- matrix(runif(nh*nh, min=-aw, max=aw), nrow=nh) * matrix(rbinom(nh*nh,1,pw), nrow=nh)
    W <- (nu/max(abs(eigen(W, only.values=T)$values))) * W
    U <- matrix(runif(nh*p, min=-au, max=au), nrow=nh) * matrix(rbinom(nh*p,1,pu), nrow=nh)
    H <- matrix(NA, nrow=nrow(Xin), ncol=nh)
    for(j in 1:ncol(Yin)){
      tmp <- U %*% c(1,Xin[1,j])
      H[1,] <- tanh(tmp) 
      for(i in 2:nrow(H)){
        tmp <- W%*%H[i-1,] + U%*%c(1,Xin[i,j])
        H[i,] <- tanh(tmp)
      }
      Hnew <- H
      
      Hpred <- matrix(NA, nrow=nrow(Xpred), ncol=nh)
      tmp <- W%*%H[nrow(H), ] + U%*%c(1,Xpred[1,j])
      Hpred[1,] <- tanh(tmp)
      if(nrow(Hpred)>1){
        for(i in 2:nrow(Hpred)){
          tmp <- W%*%Hpred[i-1, ] + U%*%c(1,Xpred[i,j])
          Hpred[i,] <- tanh(tmp)
        }
      }
      

      fit <- bPois(X=Hnew, Y=Yin[,j])
      Ypred <- exp(Hpred%*%t(fit$Beta))
      Preds[,j,r] <- median(Ypred)
      print(j)
    }
  }
  
  return(Preds)
}




set.seed(1)
MSE <- MSLE <- COV <- IS <- matrix(NA, nrow=5, ncol=5)
for(TY in 46:50){
  PredY <- TY
  train <- schoolsM[,1:(PredY-1)]
  test <- schoolsM[,PredY]
  Xin <- t(train[,1:(ncol(train)-1)])
  Yin <- t(train[,2:(ncol(train))])
  Xpred <- t(matrix(schoolsM[,PredY-1]))
  
  aw <- pw <- au <- pu <- .1
  fit <- Pois_ESN_SS(Xin=log(Xin+1), Yin=Yin, Xpred=log(Xpred+1), nh=120, lambda=2, nu=0.35, aw=aw, pw=pw, au=au, pu=pu, reps=100)
  fit2 <- Pois_ESN_Bayes(Xin=log(Xin+1), 
                         Yin=Yin, Xpred=log(Xpred+1), nh=120, lambda=2, nu=0.35, aw=aw, pw=pw, au=au, pu=pu, reps=2)
  
  predsAR <- lAR <- hAR <- rep(NA, ncol(Xin))
  for(i in 1:length(predsAR)){
    tmp <- tsglm(train[i,],  model=list(past_obs=1, past_mean=1), dist="poisson")
    predsAR[i] <- predict(tmp, n.ahead=1)$pred
    lAR[i] <- predict(tmp, n.ahead=1)$interval[1]
    hAR[i] <- predict(tmp, n.ahead=1)$interval[2]
    print(i)
  }
  
  predsSing <- fit[1,,1]
  predsEns <- apply(fit[1,,], 1, mean)
  predsBayes <- apply(fit2[1,,], 1, mean)
  
  lEns <- apply(fit[1,,], 1, function(x) quantile(rpois(109,x), probs=0.025))
  hEns <- apply(fit[1,,], 1, function(x) quantile(rpois(109,x), probs=0.975))
  
  lBayes <- apply(fit2[1,,], 1, function(x) quantile(rpois(109,x), probs=0.025))
  hBayes <- apply(fit2[1,,], 1, function(x) quantile(rpois(109,x), probs=0.975))
  
  
  MSE[1,(TY-45)] <- mean((test-rowMeans(schoolsM))^2)
  MSE[2,(TY-45)] <- mean(((predsAR-test))^2, na.rm=T)
  MSE[3,(TY-45)] <- mean(((predsSing-test))^2, na.rm=T)
  MSE[4,(TY-45)] <- mean(((predsEns-test))^2, na.rm=T)
  MSE[5,(TY-45)] <- mean(((predsBayes-test))^2, na.rm=T)
  
  MSLE[1,(TY-45)] <- mean((log(1+rowMeans(schoolsM))-log(1+test))^2, na.rm=T)
  MSLE[2,(TY-45)] <- mean((log(1+predsAR)-log(1+test))^2, na.rm=T)
  MSLE[3,(TY-45)] <- mean((log(1+predsSing)-log(1+test))^2, na.rm=T)
  MSLE[4,(TY-45)] <- mean((log(1+predsEns)-log(1+test))^2, na.rm=T)
  MSLE[5,(TY-45)] <- mean((log(1+predsBayes)-log(1+test))^2, na.rm=T)
  
  COV[2,(TY-45)] <- mean(test < hAR & test > lAR)
  COV[4,(TY-45)] <- mean(test < hEns & test > lEns)
  COV[5,(TY-45)] <- mean(test < hBayes & test > lBayes)
  
  IS[2,(TY-45)] <- mean((hAR -lAR) + 2/.05*(lAR - test)*(test<lAR) + 
             2/.05*(test-hAR)*(test>hAR))
  IS[4,(TY-45)] <- mean((hEns -lEns) + 2/.05*(lEns - test)*(test<lEns) + 
                          2/.05*(test-hEns)*(test>hEns))
  IS[5,(TY-45)] <- mean((hBayes -lBayes) + 2/.05*(lBayes - test)*(test<lBayes) + 
                          2/.05*(test-hBayes)*(test>hBayes))
    
  
  print(TY)
}

MSE
MSLE
COV
IS