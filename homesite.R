setwd("/Users/homw/Documents/petp/homesite/")

###################
# Packages
###################
library(readr)
library(xgboost)
library(ggplot2)
library(dplyr)
library(tidyr)
library(pROC)

###################
# Reading the data
###################
train <- read_csv("train.csv")
test <- read_csv("test.csv")
sub <-read_csv("sample_submission.csv")

#To load back immediately
TRN <- train
TST <- test
#rm(TRN,TST)


#Parse the date field in train and test
parse_date <- function(df){
  date <- as.Date(df$Original_Quote_Date)
  df$month <- as.integer(format(date, "%m"))
  df$year <- as.integer(format(date, "%y"))
  df$wday <- weekdays(date)
  df <- df[,!(names(df) %in% c("Original_Quote_Date"))]
  df <- df[,-1] #Remove quote number
  return (df)
}
train <- parse_date(train)
test <- parse_date(test)

###################
#find the columns with NAs
###################
x <- apply(train, 2, function(x) sum(is.na(x)))
sum(is.na(train))
x <- order(x, decreasing = T)
summary(train[,c(125,161)])

sum(is.na(test))
y <- apply(test, 2, function(x) sum(is.na(x)))
t <- y[y>0]
names(t)
summary(test[,names(t)])
####################
#replace NAs with -1 to retain information in 0 value
train[is.na(train)] <- -99
test[is.na(test)] <- -99

#count -1 or "" => No response fields
label <- train$QuoteConversion_Flag
train <- train[,!(names(train) %in% c("QuoteConversion_Flag"))]

train$count_na <- apply(train, 1, function(t) {length(which(t==-1)) + length(which(t==""))})
train$count_zero <- apply(train, 1, function(t) {length(which(t== 0))})
train$count_miss <- apply(train, 1, function(t) {length(which(t== -99))})

test$count_na <- apply(test, 1, function(t) {length(which(t==-1)) + length(which(t==""))})
test$count_zero <- apply(test, 1, function(t) {length(which(t== 0))})
test$count_miss <- apply(test, 1, function(t) {length(which(t== -99))})
table(train$wday)

#convert all the fields to numeric to fit xgboost model 
features <- names(train)
for(f in features){
  if (class(train[[f]]) == "character"){
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels = levels))
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}

sum(is.na(test))
sum(is.na(train))

# fit a simple xgb
# divide into train, validation for model fitting. Hold out set for crossvalidation
#choose only the selected features
train_raw <- train
train <- train[,c(features_sel,"count_miss")]
test_raw <- test
test <- test[,c(features_sel,"count_miss")]


set.seed(9702)
st <- sample(1:nrow(train), 0.6*nrow(train))
dtrain <- train[st,]
dval <- train[-st,]
hold_out <- sample(1:nrow(dval), 0.25*nrow(dval))
dhold_out <- train[-st,][hold_out,]
dval <- dval[-hold_out,]
sv <- as.integer(row.names(dval))

dtrain <- xgb.DMatrix(data.matrix(dtrain), label = label[st])
dval <- xgb.DMatrix(data.matrix(dval), label = label[sv])
dhold_out <- xgb.DMatrix(data.matrix(dhold_out), label = label[hold_out])
watchlist = list(train = dtrain, val = dval)

fit <- fit.model(watchlist, dtrain)

pred <- predict(fit, data.matrix(train[-st,]))
act <- label[-st]
auc(response = act,predictor = pred)
imp.features <- xgb.importance(feature_names = names(train), model = fit)
xgb.plot.importance(imp.features[1:10])

# setup a 5 fold crossvalidation
#train <- train1
#train <- train[sample(nrow(train),20000),]
auc.cv <- 1:5
folds <- cut(seq(1,nrow(train)), breaks = 5, labels = FALSE)
set.seed(97008)
shuffled <- sample(nrow(train))
for (i in 1:5){
  sh <- which(folds == i, arr.ind = TRUE)
  i2 <- ifelse(((i+1) %% 5) == 0, 5,((i+1) %% 5))
  sv <- which(folds == i2, arr.ind = TRUE)
  st <- which(!(folds %in% c(i, i2)))
  train_s <- train[shuffled,]
  label_s <- label[shuffled]
  
  dtrain <- xgb.DMatrix(data.matrix(train_s[st,]), label = label_s[st])
  dval <- xgb.DMatrix(data.matrix(train_s[sv,]), label = label_s[sv])
  dhold_out <- xgb.DMatrix(data.matrix(train_s[sh,]), label = label_s[sh])
  watchlist = list(val = dval, train = dtrain)
  
  fit <- fit.model(watchlist, dtrain)
  
  pred <- predict(fit, data.matrix(train_s[sh,]))
  act <- label_s[sh]
  auc.cv[i] <- auc(response = act,predictor = pred)[1]
  cat("\n")
  cat(auc.cv[i],"of fold ", i, "\n")
}


fit.model <- function(watchlist, dtrain){
  param <- list("objective" = "binary:logistic",
                "booster" = "gbtree",
                "eval_metric" = "auc",    # evaluation metric 
                "nthread" = 4,   # number of threads to be used 
                "max_depth" = 4,    # maximum depth of tree 
                "eta" = 0.03,    # step size shrinkage 
                "gamma" = 0,    # minimum loss reduction 
                "subsample" = 0.4,    # part of data instances to grow tree 
                "colsample_bytree" = 0.6  # subsample ratio of col
  )
  
  fit <- xgb.train(params = param, data = dtrain, nrounds = 3000, watchlist = watchlist, print.every.n = 50, early.stop.round = 200, maximize = TRUE)
  
  return(fit)
}

st <- sample(nrow(train), 0.96*nrow(train))
dtrain <- xgb.DMatrix(data.matrix(train[st,]), label = label[st])
dval <- xgb.DMatrix(data.matrix(train[-st,]), label = label[-st])
watchlist = list(val = dval, train = dtrain)

fit <- fit.model(watchlist, dtrain)
#print(fit)
#test data
pred <- predict(fit, data.matrix(test))
sub$QuoteConversion_Flag <- pred
write.csv(sub, "sub7.csv", row.names = FALSE)

imp.features <- xgb.importance(feature_names = names(train), model = fit)
xgb.plot.importance(imp.features[1:10])
table(train$year)
features_sel <- imp.features$Feature[1:100]
