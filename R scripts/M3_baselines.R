library(Mcomp)
library(forecast)
library(data.table)
library(magrittr)

path <- ''

M3S <- subset(M3, 'monthly')
M3S_freq <- 12
M3S_h <- 6
M3S_o <- 13

# M3S <- subset(M3, 'quarterly')
# M3S_freq <- 4
# M3S_h <- 4
# M3S_o <- 5

# M3S <- subset(M3, 'yearly')
# M3S_freq <- 1
# M3S_h <- 2
# M3S_o <- 5

# M3S <- subset(M3, 'other')
# M3S_freq <- 1
# M3S_h <- 2
# M3S_o <- 7

##############
#ets forecasts
##############
library(doParallel)
ncores <- detectCores()
cl <- makeCluster(ncores, outfile = "")
registerDoParallel(cl)

pb <- txtProgressBar(min = 1, max = length(M3S), style = 3)

ets_forecasts_df <- foreach(id = c(1:length(M3S)), .combine = 'rbind', .packages = 'forecast') %dopar% {
  
  ets_forecasts_mean_df <- data.frame()
  ets_forecasts_sd_df <- data.frame()
  ets_insample_sd_df <- data.frame()
  
  ts <- M3S[[id]]
  
  counter <- 1
  for (fc_origin in c(1:M3S_o)){
    
    if (fc_origin == 1){
      ts_origin <- ts$x
    } else {
      ts_origin <- ts(c(ts$x, ts$xx[1:(fc_origin-1)]), frequency = M3S_freq)
    }
    
    ets_fit <- ets(ts_origin)
    ets_forecasts <- forecast(ets_fit, h = M3S_h, PI = TRUE, level = c(0.9))
    
    ets_forecasts_mean_df[counter, 1] <- id
    ets_forecasts_mean_df[counter, 2] <- fc_origin
    ets_forecasts_mean_df[counter, c(3:(3+M3S_h-1))] <- ets_forecasts$mean
    ets_forecasts_mean_df[counter, (3+M3S_h)] <- 'mean_forecast'
    
    ets_forecasts_sd_df[counter, 1] <- id
    ets_forecasts_sd_df[counter, 2] <- fc_origin
    ets_forecasts_sd_df[counter, c(3:(3+M3S_h-1))] <- (ets_forecasts$upper - ets_forecasts$mean)/qnorm(0.95)
    ets_forecasts_sd_df[counter, (3+M3S_h)] <- 'sd_forecast'
    
    ets_insample_sd_df[counter, 1] <- id
    ets_insample_sd_df[counter, 2] <- fc_origin
    ets_insample_sd_df[counter, c(3:(3+M3S_h-1))] <- rep(accuracy(ets_fit)[2], M3S_h)
    ets_insample_sd_df[counter, (3+M3S_h)] <- 'sd_insample'
    
    counter <- counter+1
    
  }
  
  setTxtProgressBar(pb, id)
  #output
  return(rbind(ets_forecasts_mean_df, ets_forecasts_sd_df, ets_insample_sd_df))
  
}

stopCluster(cl)
registerDoSEQ()

colnames(ets_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M3S_h), 'type')
ets_forecasts_dt <- ets_forecasts_df %>% data.table()

fwrite(ets_forecasts_dt, paste0(path, 'M3M_ETS_probabilistic.csv'))

##############
#arima forecasts
##############
library(doParallel)
ncores <- detectCores()
cl <- makeCluster(ncores, outfile = "")
registerDoParallel(cl)

pb <- txtProgressBar(min = 1, max = length(M3S), style = 3)

arima_forecasts_df <- foreach(id = c(1:length(M3S)), .combine = 'rbind', .packages = 'forecast') %dopar% {
  
  arima_forecasts_mean_df <- data.frame()
  arima_forecasts_sd_df <- data.frame()
  arima_insample_sd_df <- data.frame()
  
  ts <- M3S[[id]]
  
  counter <- 1
  for (fc_origin in c(1:M3S_o)){
    
    if (fc_origin == 1){
      ts_origin <- ts$x
    } else {
      ts_origin <- ts(c(ts$x, ts$xx[1:(fc_origin-1)]), frequency = M3S_freq)
    }
    
    arima_fit <- auto.arima(ts_origin)
    arima_forecasts <- forecast(arima_fit, h = M3S_h, level = c(0.9))
    
    arima_forecasts_mean_df[counter, 1] <- id
    arima_forecasts_mean_df[counter, 2] <- fc_origin
    arima_forecasts_mean_df[counter, c(3:(3+M3S_h-1))] <- arima_forecasts$mean
    arima_forecasts_mean_df[counter, (3+M3S_h)] <- 'mean_forecast'
    
    arima_forecasts_sd_df[counter, 1] <- id
    arima_forecasts_sd_df[counter, 2] <- fc_origin
    arima_forecasts_sd_df[counter, c(3:(3+M3S_h-1))] <- (arima_forecasts$upper - arima_forecasts$mean)/qnorm(0.95)
    arima_forecasts_sd_df[counter, (3+M3S_h)] <- 'sd_forecast'
    
    arima_insample_sd_df[counter, 1] <- id
    arima_insample_sd_df[counter, 2] <- fc_origin
    arima_insample_sd_df[counter, c(3:(3+M3S_h-1))] <- rep(accuracy(arima_fit)[2], M3S_h)
    arima_insample_sd_df[counter, (3+M3S_h)] <- 'sd_insample'
    
    counter <- counter+1
  }
  
  setTxtProgressBar(pb, id)
  #output
  return(rbind(arima_forecasts_mean_df, arima_forecasts_sd_df, arima_insample_sd_df))
  
}

close(pb)
stopCluster(cl)
registerDoSEQ()

colnames(arima_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M3S_h), 'type')
arima_forecasts_dt <- arima_forecasts_df %>% data.table()

fwrite(arima_forecasts_dt, paste0(path, 'M3M_arima_probabilistic.csv'))

##############
#theta forecasts
##############
library(doParallel)
ncores <- detectCores()
cl <- makeCluster(ncores, outfile = "")
registerDoParallel(cl)

pb <- txtProgressBar(min = 1, max = length(M3S), style = 3)

theta_forecasts_df <- foreach(id = c(1:length(M3S)), .combine = 'rbind', .packages = 'forecast') %dopar% {
  
  theta_forecasts_mean_df <- data.frame()
  theta_forecasts_sd_df <- data.frame()
  theta_insample_sd_df <- data.frame()
  
  ts <- M3S[[id]]
  
  counter <- 1
  for (fc_origin in c(1:M3S_o)){
    
    if (fc_origin == 1){
      ts_origin <- ts$x
    } else {
      ts_origin <- ts(c(ts$x, ts$xx[1:(fc_origin-1)]), frequency = M3S_freq)
    }
    
    theta_forecasts <- thetaf(ts_origin, h = M3S_h, level = c(0.9))
    
    theta_forecasts_mean_df[counter, 1] <- id
    theta_forecasts_mean_df[counter, 2] <- fc_origin
    theta_forecasts_mean_df[counter, c(3:(3+M3S_h-1))] <- theta_forecasts$mean
    theta_forecasts_mean_df[counter, (3+M3S_h)] <- 'mean_forecast'
    
    theta_forecasts_sd_df[counter, 1] <- id
    theta_forecasts_sd_df[counter, 2] <- fc_origin
    theta_forecasts_sd_df[counter, c(3:(3+M3S_h-1))] <- (theta_forecasts$upper - theta_forecasts$mean)/qnorm(0.95)
    theta_forecasts_sd_df[counter, (3+M3S_h)] <- 'sd_forecast'
    
    theta_insample_sd_df[counter, 1] <- id
    theta_insample_sd_df[counter, 2] <- fc_origin
    theta_insample_sd_df[counter, c(3:(3+M3S_h-1))] <- rep(accuracy(x = theta_forecasts$x,
                                                          f = theta_forecasts$fitted)[2], M3S_h)
    theta_insample_sd_df[counter, (3+M3S_h)] <- 'sd_insample'
    
    counter <- counter+1
  }
  
  setTxtProgressBar(pb, id)
  #output
  return(rbind(theta_forecasts_mean_df, theta_forecasts_sd_df, theta_insample_sd_df))
  
}

stopCluster(cl)
registerDoSEQ()

colnames(theta_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M3S_h), 'type')
theta_forecasts_dt <- theta_forecasts_df %>% data.table()

fwrite(theta_forecasts_dt, paste0(path, 'M3M_theta_probabilistic.csv'))
