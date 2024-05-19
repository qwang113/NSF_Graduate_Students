rm(list = ls())
library(sp)
library(fields)
library(ggplot2)
library(Rfast)
set.seed(0)
gridsize = 30
timesize = 15
rho <- 0.9
sig = 1
long <- seq(from = 0, to = 1, length.out = gridsize)
lat <- seq(from = 0, to = 1, length.out = gridsize)
timestep <- 1:timesize
coords = expand.grid(long, lat, timestep)
dist_sp <- rdist(coords[,1:2])
dist_time <- rdist(coords[,3])
cov_sp <- Matern(dist_sp, range = 0.5)
cov_time <- rho^(dist_time)
cov_mat <- cov_sp * cov_time * sig^2

conti_y <- Rfast::rmvnorm(1, mu = rep(1, gridsize^2*timesize), sigma = cov_mat)
count_y <- rpois(length(conti_y), lambda = exp(conti_y))

y <- matrix(count_y, ncol = gridsize^2)

# Reshape y to a data frame
df <- data.frame(
  long = coords[, 1],
  lat = coords[, 2],
  time = coords[, 3],
  value = as.vector(y)
)
# Assuming df is your data frame from the previous step

# Convert to wide format
df_wide <- df %>%
  pivot_wider(names_from = time, values_from = value, names_prefix = "time_")

# Plot
ggplot(df, aes(x = long, y = lat, fill = value)) +
  geom_tile() +
  facet_wrap(~time) +
  scale_fill_viridis_c() +
  theme_minimal() +
  labs(title = "Spatio-Temporal Data Heatmap",
       x = "Longitude",
       y = "Latitude",
       fill = "Value")
write.csv(df, "D:/77/UCSC/study/Research/temp/sim.csv", row.names = FALSE)
write.csv(df_wide, "D:/77/UCSC/study/Research/temp/sim_wide.csv", row.names = FALSE)

