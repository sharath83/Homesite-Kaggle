# train
# test
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
#replace NAs with -1 to retain information in 0 value
train[is.na(train)] <- -99
test[is.na(test)] <- -99

#count -1 or "" => No response fields
label <- train$QuoteConversion_Flag
train <- train[,!(names(train) %in% c("QuoteConversion_Flag"))]
#-------------------------

#build model on each type of variable and predict on test
tr.new <- train
ts.new <- test
#Sales
#select releavnt features from sales
tr.new$SalesF16 <- tr.new$SalesField13+ tr.new$SalesField14+tr.new$SalesField15
ts.new$SalesF16 <- ts.new$SalesField13+ ts.new$SalesField14+ts.new$SalesField15
tr.new <- tr.new[,grepl("Sales", names(tr.new))]
ts.new <- ts.new[,grepl("Sales", names(ts.new))]

sales.f <- names(tr.new)[!(names(tr.new) %in% c("SalesField7","SalesField8","SalesField11","SalesField12","SalesField2B","SalesField13","SalesField14","SalesField15"))]

set.seed(9999)
v <- sample(nrow(tr.new), 0.3*nrow(tr.new))
val <- tr.new[v,]
#tr.new <- tr.new[-v,]
lab.val <- label[v]
lab.tr <- label[-v]

fit.sales <- glm(lab.tr ~.-SalesField4 + as.factor(SalesField4) - SalesField5 + as.factor(SalesField5) - SalesField6 + as.factor(SalesField6) - SalesField9 + as.factor(SalesField9), data = tr.new[-v,sales.f], family = binomial)
summary(fit.sales)
#table(tr.new$SalesF16)
val.pred <- predict(fit.sales, val[,sales.f], type = "response")
glmnet::auc(lab.val, val.pred)

sales.pred <- predict(fit.sales, ts.new[,sales.f], type = "response")
train.pred <- predict(fit.sales, tr.new[,sales.f], type = "response")


sales.pred[1:10]
train.pred[1:10]
fit.sales$fitted.values[1:10]
train$sales.pred <- train.pred
test$sales.pred <- sales.pred

#property
tr.new <- train[,grepl("Property", names(train))]
ts.new <- test[,grepl("Property", names(test))]
prop.f <- names(tr.new)[!(names(tr.new) %in% c("PropertyField3","PropertyField4","PropertyField5","PropertyField6","PropertyField8","PropertyField9","PropertyField10","PropertyField14","PropertyField17","PropertyField18","PropertyField19","PropertyField20","PropertyField23","PropertyField24","PropertyField25","PropertyField26A","PropertyField26B","PropertyField29","PropertyField36","PropertyField38"))]

tr.new <- tr.new[,prop.f]
ts.new <- ts.new[,prop.f]

fit.prop <- glm(lab.tr ~., data = tr.new[-v,], family = binomial)
summary(fit.prop)

ts.new$PropertyField7 <- factor(ts.new$PropertyField7, levels = unique(tr.new$PropertyField7))
ts.new$PropertyField30 <- factor(ts.new$PropertyField30, levels = unique(tr.new$PropertyField30))
ts.new$PropertyField37 <- factor(ts.new$PropertyField37, levels = unique(tr.new$PropertyField37))

val.pred <- predict(fit.prop, tr.new[v,], type = "response")
glmnet::auc(lab.val, val.pred)

prop.pred <- predict(fit.prop, ts.new, type = "response")
train.pred <- predict(fit.prop, tr.new, type = "response")

prop.pred[1:10]
train.pred[1:10]
fit.prop$fitted.values[1:10]

train$prop.pred <- train.pred
test$prop.pred <- prop.pred
rm(fit.prop)

#coverage
tr.new <- train[,grepl("Coverage", names(train))]
ts.new <- test[,grepl("Coverage", names(test))]
cover.f <- names(tr.new)[!(names(tr.new) %in% c("CoverageField11A","CoverageField11B","CoverageField2A","CoverageField2B"))]

tr.new <- tr.new[,cover.f]
ts.new <- ts.new[,cover.f]

fit.cover <- glm(lab.tr ~., data = tr.new[-v,], family = binomial)
summary(fit.cover)

val.pred <- predict(fit.cover, ts.new[v,], type = "response")
glmnet::auc(lab.val, val.pred)
auc(lab.tr, fit.cover$fitted.values)


cover.pred[1:10]
fit.cover$fitted.values[1:10]
train$cover.pred <- fit.cover$fitted.values
test$cover.pred <- cover.pred
rm(fit.cover)

library(randomForest)
tr.new <- as.data.frame(apply(tr.new,2,as.factor))
ts.new <- as.data.frame(apply(ts.new,2,as.factor))

fit.cover <- randomForest(x=tr.new[-v,],y = as.factor(lab.tr), ntree = 100,mtry = 4,do.trace = TRUE,replace = TRUE)
val.pred <- predict(fit.cover, tr.new[v,], type="prob")
glmnet::auc(lab.val, val.pred[,2])

train.pred <- predict(fit.cover, tr.new, type="prob")[,2]
cover.pred <- predict(fit.cover, ts.new, type="prob")[,2]
length(cover.pred[cover.pred>0])

train$cover.pred <- train.pred
test$cover.pred <- cover.pred

#Personal
tr.new <- train[,grepl("Personal", names(train))]
ts.new <- test[,grepl("Personal", names(test))]

fit.personal <- glm(label ~., data = tr.new, family = binomial)
summary(fit.personal)
ts.new$PersonalField16 <- factor(ts.new$PersonalField16, levels = unique(tr.new$PersonalField16))
ts.new$PersonalField17 <- factor(ts.new$PersonalField17, levels = unique(tr.new$PersonalField17))
ts.new$PersonalField18 <- factor(ts.new$PersonalField18, levels = unique(tr.new$PersonalField18))
ts.new$PersonalField19 <- factor(ts.new$PersonalField19, levels = unique(tr.new$PersonalField19))

personal.pred <- predict(fit.personal, ts.new, type = "response")
personal.pred[1:10]
fit.personal$fitted.values[1:10]
train$personal.pred <- fit.personal$fitted.values
test$personal.pred <- personal.pred
rm(fit.personal)

#Try random forest for personal features
tr.new <- apply(tr.new, 2, as.factor)
ts.new <- apply(ts.new, 2, as.factor)

fit.personal <- randomForest(x=tr.new[-v,],y = as.factor(lab.tr), ntree = 100,mtry = 4,do.trace = TRUE,replace = TRUE)
val.pred <- predict(fit.personal, tr.new[v,], type = "prob")
glmnet::auc(lab.val, val.pred[,2])

train.pred <- predict(fit.personal, tr.new, type="prob")[,2]
personal.pred <- predict(fit.personal, ts.new, type="prob")[,2]
length(personal.pred[personal.pred>0])

train$personal.pred <- train.pred
test$personal.pred <- personal.pred


#Geopraphic
tr.new <- train[,grepl("Geographic", names(train))]
ts.new <- test[,grepl("Geographic", names(test))]

tr.new <- as.data.frame(apply(tr.new,2,as.factor))
ts.new <- as.data.frame(apply(ts.new,2,as.factor))

fit.geo <- randomForest(x=tr.new[-v,],y = as.factor(lab.tr), ntree = 100,mtry = 5,do.trace = TRUE,replace = TRUE)

val.pred <- predict(fit.geo, tr.new[v,], type = "prob")
glmnet::auc(lab.val, val.pred[,2])
auc(lab.tr, train.pred)

train.pred <- predict(fit.geo, tr.new[-v,], type="prob")[,2]
geo.pred <- predict(fit.geo, ts.new, type="prob")[,2]
length(geo.pred[geo.pred>0])

train$geo.pred <- train.pred
test$geo.pred <- personal.pred

for (f in names(tr.new)){
  if ((length(unique(levels(tr.new[[f]]))) != length(unique(levels(ts.new[[f]]))))){
    print(f)
    cat("\n")
  }
}
ts.new$GeographicField25A <- factor(ts.new$GeographicField25A, levels = unique(tr.new$GeographicField25A))
ts.new$GeographicField27A <- factor(ts.new$GeographicField27A, levels = unique(tr.new$GeographicField27A))
sum(is.na(ts.new))
summary(ts.new$GeographicField25A)
ts.new[is.na(ts.new)] <- "-1"

geo.pred <- predict(fit.geo, ts.new, type = "prob")
geo.pred[1:10,2]
sales.pred[1:10]

train.geo.pred <- predict(fit.geo, tr.new, type="prob")
train.geo.pred[1:10,2]

auc(label,train.geo.pred)
test$geo.pred <- geo.pred[,2]
train$geo.pred <- train.geo.pred[,2]
rm(fit.geo, train.geo.pred,geo.pred,sales.pred,prop.pred,cover.pred,personal.pred)

names(train)[298:304]
names(test)[300:304]

train <- train[,-304] #removing geo.pred since its validation score is bad
test <- test[,-304]

tr.preds <- train[,grepl(".pred", names(train))]
ts.preds <- test[,grepl(".pred", names(test))]
#Build xgb
train$count_miss <- apply(train, 1, function(t) {length(which(t==-1)) + length(which(t==""))})
train$count_zero <- apply(train, 1, function(t) {length(which(t== 0))})
train$count_na <- apply(train, 1, function(t) {length(which(t== -99))})

test$count_miss <- apply(test, 1, function(t) {length(which(t==-1)) + length(which(t==""))})
test$count_zero <- apply(test, 1, function(t) {length(which(t== 0))})
test$count_na <- apply(test, 1, function(t) {length(which(t== -99))})

#converting fields to numeric type. Xgb handles only numeric variables
features <- names(train)
for(f in features){
  if (class(train[[f]]) == "character"){
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels = levels))
    test[[f]] <- as.integer(factor(test[[f]], levels=levels))
  }
}
sum(is.na(train))
sum(is.na(test))
##-------------
train <- read_csv("train.csv")
label <- train$QuoteConversion_Flag
train <- read_csv("train_h.csv")
test <- read_csv("test_h.csv")

set.seed(102020)
val = sample(nrow(train),10000)
dtrain <- xgb.DMatrix(data.matrix(train[-val,selected.f]),label = label[-val])
dval <- xgb.DMatrix(data.matrix(train[val,selected.f]),label = label[val])
watchlist = list(val = dval, train = dtrain)
fit <- fit.model(watchlist, dtrain)

imp.features <- xgb.importance(feature_names = names(train), model = fit)
sub.pred <- predict(fit, data.matrix(test))
sub7 <- read.csv("sub7.csv")
sub8 <- read.csv("sub8.csv")
sub.pred[1:10]
sub$QuoteConversion_Flag[1:10]
sub$QuoteConversion_Flag <- sub.pred
write.csv(sub, "sub8.csv", row.names = FALSE)
sub9 <- sub
sub9$QuoteConversion_Flag <- sub7$QuoteConversion_Flag*.91+0.09*sub8$QuoteConversion_Flag
write.csv(sub9, "sub10.csv", row.names = FALSE)

fit.model <- function(watchlist, dtrain){
  param <- list("objective" = "binary:logistic",
                "booster" = "gbtree",
                "eval_metric" = "auc",    # evaluation metric 
                "nthread" = 4,   # number of threads to be used 
                "max_depth" = 5,    # maximum depth of tree 
                "eta" = 0.05,    # step size shrinkage 
                "gamma" = 0,    # minimum loss reduction 
                "subsample" = 0.4,    # part of data instances to grow tree 
                "colsample_bytree" = 0.6  # subsample ratio of col
  )
  
  fit <- xgb.train(params = param, data = dtrain, nrounds = 1000, watchlist = watchlist, print.every.n = 20, early.stop.round = 60, maximize = TRUE)
  
  return(fit)
}

naa <- test[rowSums(is.na(test)) >0 ,]
test[is.na(test)] <- 0

#setting up CV
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
  auc.cv[i] <- glmnet::auc(act,pred)
  cat("\n")
  cat(auc.cv[i],"of fold ", i, "\n")
}


#------Submission-------
set.seed(1020208)
val = sample(nrow(train),50000)
dtrain <- xgb.DMatrix(data.matrix(train[-val,]),label = label[-val])
dval <- xgb.DMatrix(data.matrix(train[val,]),label = label[val])
watchlist = list(val = dval, train = dtrain)
fit <- fit.model(watchlist, dtrain)
sub.pred <- predict(fit, data.matrix(test))
sub$QuoteConversion_Flag <- sub.pred
write.csv(sub, "sub12.csv", row.names = FALSE)

sub10 <- read.csv("sub10.csv")
sub15 <- sub
sub15$QuoteConversion_Flag <- sub14$QuoteConversion_Flag*.91 + sub$QuoteConversion_Flag*.09

write.csv(sub15, "sub15.csv", row.names = FALSE)
a <- glmnet::auc(label[1:1.19e05], as.numeric(train$sales.pred[1:1.19e05]))
summary(train$sales.pred)

length(train$cover.pred[train$cover.pred==0])
train$cover.pred[1:10]

write.csv(train,"train_h.csv", row.names = FALSE)
write.csv(test,"test_h.csv", row.names = FALSE)
imp.features <- xgb.importance(names(train),model = fit)
selected.f <- imp.features$Feature[1:100]
selected.f <- selected.f[!(selected.f %in% "geo.pred")]
