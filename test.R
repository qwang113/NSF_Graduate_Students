x <- rnorm(800)
y <- rnorm(800)
z <- rpois(800, lambda = exp(x*2 + y*1.5))
dat <- as.data.frame(cbind(x,y,z))

toy_model <- glmnet(x = cbind(x,y), y = z, family = poisson(link = "log"))
