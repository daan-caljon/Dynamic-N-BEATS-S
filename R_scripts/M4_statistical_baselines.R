library(Mcomp)
library(forecast)
library(data.table)
library(magrittr)
library(stringr)

setwd(choose.dir()) #set to current directory

read_path <- 'M4DataSet/'
write_path <- ''

# ################################################
M4S_actuals_train <- fread(paste0(read_path, 'Monthly-train.csv'))
M4S_actuals_test <- fread(paste0(read_path, 'Monthly-test.csv'))
M4S_freq <- 12
M4S_h <- 6
M4S_oh <- 18
M4S_o <- 13
M4S_abbr <- 'M'
# ################################################
# M4S_actuals_train <- fread(paste0(read_path, 'Yearly-train.csv'))
# M4S_actuals_test <- fread(paste0(read_path, 'Yearly-test.csv'))
# M4S_freq <- 1
# M4S_h <- 2
# M4S_oh <- 6
# M4S_o <- 5
# M4S_abbr <- 'Y'
# ################################################
# M4S_actuals_train <- fread(paste0(read_path, 'Quarterly-train.csv'))
# M4S_actuals_test <- fread(paste0(read_path, 'Quarterly-test.csv'))
# M4S_freq <- 4
# M4S_h <- 4
# M4S_oh <- 8
# M4S_o <- 5
# M4S_abbr <- 'Q'
# ################################################
# M4S_actuals_train <- fread(paste0(read_path, 'Weekly-train.csv'))
# M4S_actuals_test <- fread(paste0(read_path, 'Weekly-test.csv'))
# M4S_freq <- 1
# M4S_h <- 4
# M4S_oh <- 13
# M4S_o <- 10
# M4S_abbr <- 'W'
# ################################################
# M4S_actuals_train <- fread(paste0(read_path, 'Daily-train.csv'))
# M4S_actuals_test <- fread(paste0(read_path, 'Daily-test.csv'))
# M4S_freq <- 7
# M4S_h <- 7
# M4S_oh <- 14
# M4S_o <- 8
# M4S_abbr <- 'D'
# ################################################
# M4S_actuals_train <- fread(paste0(read_path, 'Hourly-train.csv'))
# M4S_actuals_test <- fread(paste0(read_path, 'Hourly-test.csv'))
# M4S_freq <- 24
# M4S_h <- 24
# M4S_oh <- 48
# M4S_o <- 25
# M4S_abbr <- 'H'
# ################################################

setnames(M4S_actuals_train, names(M4S_actuals_train), c('item_id', 1:(ncol(M4S_actuals_train)-1)))
setnames(M4S_actuals_test, names(M4S_actuals_test), c('item_id', 1:(ncol(M4S_actuals_test)-1)))
M4S_actuals_train[, item_id := str_remove(item_id, M4S_abbr) %>% as.numeric()]
M4S_actuals_test[, item_id := str_remove(item_id, M4S_abbr) %>% as.numeric()]

M4S_actuals_all <- merge.data.table(M4S_actuals_train, M4S_actuals_test, by = 'item_id')
setnames(M4S_actuals_all, names(M4S_actuals_all), c('item_id', 1:(ncol(M4S_actuals_train)+ncol(M4S_actuals_test)-2)))
M4S_actuals_all <- melt.data.table(M4S_actuals_all, id.vars = 'item_id')
M4S_actuals_all <- M4S_actuals_all[!is.na(value)]
setorder(M4S_actuals_all, variable)
M4S_actuals_all[, variable:= c(1:.N) , by = list(item_id)]

library(doParallel)
ncores <- detectCores()
cl <- makeCluster(ncores, outfile = "")
registerDoParallel(cl)

M4S_actuals_all_df <- data.frame(M4S_actuals_all)

start_ts <- 1
end_ts <- length(unique(M4S_actuals_all$item_id))

iterations <- end_ts - start_ts + 1
pb <- txtProgressBar(min = 1, max = iterations , style = 3)

##############
#ETS forecasts
##############

starttime <- Sys.time()

ets_forecasts_df <- foreach(id = c(start_ts:end_ts), .combine = 'rbind', .packages = 'forecast') %dopar% {

  ets_forecasts_mean_df <- data.frame()
  # ets_forecasts_sd_df <- data.frame()
  # ets_insample_sd_df <- data.frame()

  ts <- M4S_actuals_all_df[which(M4S_actuals_all_df$item_id == id),]$value

    counter <- 1
    for (fc_origin in 1:M4S_o){
      
      ts_origin <- ts(ts[1:(length(ts)-(M4S_oh-(fc_origin-1)))], frequency = M4S_freq)

      ets_fit <- ets(ts_origin)
      ets_forecasts <- forecast(ets_fit, h = M4S_h, PI = FALSE)
      # ets_forecasts <- forecast(ets_fit, h = M4S_h, PI = TRUE, level = c(0.9))

      ets_forecasts_mean_df[counter, 1] <- id
      ets_forecasts_mean_df[counter, 2] <- fc_origin
      ets_forecasts_mean_df[counter, c(3:(3+M4S_h-1))] <- ets_forecasts$mean
      ets_forecasts_mean_df[counter, (3+M4S_h)] <- 'mean_forecast'

      # ets_forecasts_sd_df[counter, 1] <- id
      # ets_forecasts_sd_df[counter, 2] <- fc_origin
      # #ets_forecasts_sd_df[counter, c(3:(3+M4S_h-1))] <- (ets_forecasts$upper - ets_forecasts$mean)/qnorm(0.9) - error in first version
      # ets_forecasts_sd_df[counter, c(3:(3+M4S_h-1))] <- (ets_forecasts$upper - ets_forecasts$mean)/qnorm(0.95)
      # ets_forecasts_sd_df[counter, (3+M4S_h)] <- 'sd_forecast'
      # 
      # ets_insample_sd_df[counter, 1] <- id
      # ets_insample_sd_df[counter, 2] <- fc_origin
      # ets_insample_sd_df[counter, c(3:(3+M4S_h-1))] <- rep(accuracy(ets_fit)[2], M4S_h)
      # # ets_insample_sd_df[counter, c(3:(3+M4S_h-1))] <- rep(accuracy(x = ets_forecasts$x,
      # #                                                     f = ets_forecasts$fitted)[2], M4S_h)
      # ets_insample_sd_df[counter, (3+M4S_h)] <- 'sd_insample'

      counter <- counter+1
    }

    setTxtProgressBar(pb, (id-start_ts+1))
    #output
    return(ets_forecasts_mean_df)
    # rbind(ets_forecasts_mean_df, ets_forecasts_sd_df, ets_insample_sd_df)

}

close(pb)
stopCluster(cl)
registerDoSEQ()

colnames(ets_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M4S_h), 'type')
ets_forecasts_dt <- ets_forecasts_df %>% data.table()

Sys.time() - starttime

fwrite(ets_forecasts_dt, paste0(write_path, 'M4', M4S_abbr, '_ETS.csv'))

##############
#arima forecasts
##############

starttime <- Sys.time()

arima_forecasts_df <- foreach(id = c(start_ts:end_ts), .combine = 'rbind', .packages = 'forecast') %dopar% {

  arima_forecasts_mean_df <- data.frame()
  # arima_forecasts_sd_df <- data.frame()
  # arima_insample_sd_df <- data.frame()

  ts <- M4S_actuals_all_df[which(M4S_actuals_all_df$item_id == id),]$value

  counter <- 1
  for (fc_origin in 1:M4S_o){

    ts_origin <- ts(ts[1:(length(ts)-(M4S_oh-(fc_origin-1)))], frequency = M4S_freq)

    arima_fit <- auto.arima(ts_origin)#,
                            #truncate = 120,
                            #stepwise = TRUE,
                            #approximation = TRUE,
                            #trace = TRUE)
    arima_forecasts <- forecast(arima_fit, h = M4S_h, PI = FALSE)#, level = c(0.9))

    # arima_forecasts %>% plot(xlim=c(45,49))

    arima_forecasts_mean_df[counter, 1] <- id
    arima_forecasts_mean_df[counter, 2] <- fc_origin
    arima_forecasts_mean_df[counter, c(3:(3+M4S_h-1))] <- arima_forecasts$mean
    arima_forecasts_mean_df[counter, c(3+M4S_h)] <- 'mean_forecast'

    # arima_forecasts_sd_df[counter, 1] <- id
    # arima_forecasts_sd_df[counter, 2] <- fc_origin
    # arima_forecasts_sd_df[counter, c(3:(3+M4S_h-1))] <- (arima_forecasts$upper - arima_forecasts$mean)/qnorm(0.95)
    # arima_forecasts_sd_df[counter, c(3+M4S_h)] <- 'sd_forecast'

    # arima_insample_sd_df[counter, 1] <- id
    # arima_insample_sd_df[counter, 2] <- fc_origin
    # arima_insample_sd_df[counter, c(3:(3+M4S_h-1))] <- rep(accuracy(arima_fit)[2], M4S_h)
    # # arima_insample_sd_df[counter, c(3:(3+M4S_h-1))] <- rep(accuracy(x = arima_forecasts$x,
    # #                                                       f = arima_forecasts$fitted)[2], M4S_h)
    # arima_insample_sd_df[counter, (3+M4S_h)] <- 'sd_insample'

    counter <- counter+1
  }

  setTxtProgressBar(pb, (id-start_ts+1))
  #output
  return(arima_forecasts_mean_df)
  # return(rbind(arima_forecasts_mean_df, arima_forecasts_sd_df, arima_insample_sd_df))

}

close(pb)
stopCluster(cl)
registerDoSEQ()

colnames(arima_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M4S_h), 'type')
arima_forecasts_dt <- arima_forecasts_df %>% data.table()

Sys.time() - starttime

fwrite(arima_forecasts_dt, paste0(write_path, 'M4', M4S_abbr, '_arima.csv'))

# ##############
# #theta forecasts
# ##############

starttime <- Sys.time()

theta_forecasts_df <- foreach(id = c(start_ts:end_ts), .combine = 'rbind', .packages = 'forecast') %dopar% {

  theta_forecasts_mean_df <- data.frame()
  # theta_forecasts_sd_df <- data.frame()
  # theta_insample_sd_df <- data.frame()

  ts <- M4S_actuals_all_df[which(M4S_actuals_all_df$item_id == id),]$value

  counter <- 1
  for (fc_origin in 1:M4S_o){

    ts_origin <- ts(ts[1:(length(ts)-(M4S_oh-(fc_origin-1)))], frequency = M4S_freq)

    theta_forecasts <- thetaf(ts_origin, h = M4S_h)#, level = c(0.9))

    theta_forecasts_mean_df[counter, 1] <- id
    theta_forecasts_mean_df[counter, 2] <- fc_origin
    theta_forecasts_mean_df[counter, c(3:(3+M4S_h-1))] <- theta_forecasts$mean
    theta_forecasts_mean_df[counter, (3+M4S_h)] <- 'mean_forecast'

    # theta_forecasts_sd_df[counter, 1] <- id
    # theta_forecasts_sd_df[counter, 2] <- fc_origin
    # theta_forecasts_sd_df[counter, c(3:(3+M4S_h-1))] <- (theta_forecasts$upper - theta_forecasts$mean)/qnorm(0.95)
    # theta_forecasts_sd_df[counter, (3+M4S_h)] <- 'sd_forecast'
    # 
    # theta_insample_sd_df[counter, 1] <- id
    # theta_insample_sd_df[counter, 2] <- fc_origin
    # theta_insample_sd_df[counter, c(3:(3+M4S_h-1))] <- rep(accuracy(x = theta_forecasts$x,
    #                                                       f = theta_forecasts$fitted)[2], M4S_h)
    # theta_insample_sd_df[counter, (3+M4S_h)] <- 'sd_insample'

    counter <- counter+1
  }

  #output
  return(theta_forecasts_mean_df)
  # rbind(theta_forecasts_mean_df, theta_forecasts_sd_df, theta_insample_sd_df)

}

stopCluster(cl)
registerDoSEQ()

colnames(theta_forecasts_df) <- c('item_id', 'fc_origin', as.character(1:M4S_h), 'type')
theta_forecasts_dt <- theta_forecasts_df %>% data.table()

Sys.time() - starttime

fwrite(theta_forecasts_dt, paste0(write_path, 'M4', M4S_abbr, '_theta.csv'))
