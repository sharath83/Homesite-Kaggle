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

train10 <- train[sample(nrow(train),10000),]
lab <- train10$QuoteConversion_Flag
train10 <- train10[,-1]
# Types of features. understand the behaviour of each type
sales <- train10[,grepl("Sales", names(train10))]
sales <- sales[,-18]
sales$f1345 <- sales$SalesField13+ sales$SalesField14+sales$SalesField15

fit.sales <- glm(lab ~.-SalesField7-SalesField8-SalesField11-SalesField12-SalesField2B- SalesField4 + as.factor(SalesField4) - SalesField5 + as.factor(SalesField5) - SalesField6 + as.factor(SalesField6)- SalesField10 + as.factor(SalesField10) - SalesField9 + as.factor(SalesField9), data = as.data.frame(cbind(sales[,-c(15:17)], lab)), family = binomial)
summary(fit.sales)

sales.f <- names(sales)[!(names(sales) %in% c("SalesField7","SalesField8","SalesField11","SalesField12","SalesField2B","SalesField13","SalesField14","SalesField15"))]

tr.new <- train
tr.new$SalesF16 <- tr.new$SalesField13+ tr.new$SalesField14+tr.new$SalesField15
sales <- tr.new[1,grepl("Sales", names(tr.new))]
sales.f <- names(sales)[!(names(sales) %in% c("SalesField7","SalesField8","SalesField11","SalesField12","SalesField2B","SalesField13","SalesField14","SalesField15"))]

samp <- sample(nrow(tr.new), 0.3*nrow(tr.new))
tr <- tr.new[-samp,sales.f]
ts <- tr.new[samp,sales.f]
trY <- label[-samp]
tsY <- label[samp]

fit.sales <- glm(trY ~.-SalesField4 + as.factor(SalesField4) - SalesField5 + as.factor(SalesField5) - SalesField6 + as.factor(SalesField6) - SalesField9 + as.factor(SalesField9), data = as.data.frame(cbind(tr, trY)), family = binomial)
summary(fit.sales)
pred <- predict(fit.sales, ts, type = "response")
auc(tsY, pred)
auc(trY, fit.sales$fitted.values)
#create SalesF16, select sales.f features and fit.sales
sales.en <- cbind(ts,pred)

#Property fields
prop <- train10[,grepl("Property", names(train10))]
prop <- as.data.frame(cbind(prop,lab))
prop$PropertyField26 <- prop$PropertyField26A + prop$PropertyField26B
prop <- prop[prop$PropertyField32 != "",]

prop.fit <- glm(lab~.-PropertyField7, data = prop[,prop.f], family = binomial)
summary(prop.fit)
table(prop$PropertyField26B)
t <- as.numeric(factor(prop$PropertyField7, levels = unique(train$PropertyField7)))
auc(prop$lab, prop.fit$fitted.values)

prop.f <- names(prop)[!(names(prop) %in% c("PropertyField3","PropertyField4","PropertyField5","PropertyField6","PropertyField8","PropertyField9","PropertyField10","PropertyField14","PropertyField17","PropertyField18","PropertyField19","PropertyField20","PropertyField23","PropertyField24","PropertyField25","PropertyField26A","PropertyField26B","PropertyField29","PropertyField36","PropertyField38"))]
prop.f <- prop.f[-29]

tr <- tr.new[-samp,prop.f]
ts <- tr.new[samp,prop.f]
trY <- label[-samp]
tsY <- label[samp]
fit.prop <- glm(trY ~.-PropertyField7+as.numeric(factor(PropertyField7, levels = unique(train$PropertyField7))), data = as.data.frame(cbind(tr, trY)), family = binomial)
summary(fit.prop)
pred <- predict(fit.prop, ts, "response")
auc(tsY, pred)
prop.en <- cbind(ts, pred)
#Coverage fields
cover <- train10[,grepl("Coverage", names(train10))]
cover <- as.data.frame(cbind(cover,lab))
fit.cover <- glm(lab~.-CoverageField11A-CoverageField11B-CoverageField2A-CoverageField2B, data = cover, family = binomial)
summary(fit.cover)
auc(lab, fit.cover$fitted.values)
cover.f <- names(cover)[!(names(cover) %in% c("CoverageField11A","CoverageField11B","CoverageField2A","CoverageField2B", "lab"))]

tr <- tr.new[-samp,cover.f]
ts <- tr.new[samp,cover.f]
trY <- label[-samp]
tsY <- label[samp]

fit.cover <- glm(trY~., data = tr, family = binomial)
summary(fit.cover)
pred <- predict(fit.cover, ts, type = "response")
auc(tsY, pred)
cover.en <- cbind(ts,pred)

#Personal field
names(test)
personal <- train10[,grepl("Personal", names(train10))]
personal$PersonalField19 <- as.numeric(factor(personal$PersonalField19, levels = unique(train$PersonalField19)))
personal$PersonalField18 <- as.numeric(factor(personal$PersonalField18, levels = unique(train$PersonalField18)))
personal$PersonalField17 <- as.numeric(factor(personal$PersonalField17, levels = unique(train$PersonalField17)))


fit.personal <- glm(lab~., data = personal, "binomial")
summary(fit.personal)
pROC::auc(lab, fit.personal$fitted.values)
#length(personal[personal == ""])
fit.personal$coefficients[-1,4]

ts$PersonalField16 <- factor(ts$PersonalField16, levels = unique(c(tr$PersonalField16)))
tr$PersonalField16 <- factor(tr$PersonalField16, levels = unique(c(train$PersonalField16, test$PersonalField16)))

ts$PersonalField17 <- factor(ts$PersonalField17, levels = unique(c(tr$PersonalField17)))
tr$PersonalField17 <- factor(tr$PersonalField17, levels = unique(c(train$PersonalField17, test$PersonalField17)))

ts$PersonalField18 <- factor(ts$PersonalField18, levels = unique(c(tr$PersonalField18)))
tr$PersonalField18 <- factor(tr$PersonalField18, levels = unique(c(train$PersonalField18, test$PersonalField18)))

ts$PersonalField19 <- factor(ts$PersonalField19, levels = unique(c(tr$PersonalField19)))
tr$PersonalField19 <- factor(tr$PersonalField19, levels = unique(c(train$PersonalField19, test$PersonalField19)))


table(tr$PersonalField16)
table(ts$PersonalField16)
tr <- tr.new[-samp,names(personal)]
ts <- tr.new[samp,names(personal)]
fit.personal <- glm(trY~., data = tr, family = binomial)
summary(fit.personal)
pred <- predict(fit.personal, ts, type = "response")
auc(tsY, pred)
personal.en <- cbind(ts,pred)

#Geographic
geo <- train10[,grepl("Geographic", names(train10))]
geo10 <- apply(geo[,1:10],2,function(x){as.factor(x)})
geo20 <- apply(geo[,30:40],2,function(x){as.factor(x)})
fit.geo <- glm(lab~., data = as.data.frame(geo[,30:40]), family = binomial)

geo <- as.data.frame(cbind(geo,lab))
geo <- geo[!(duplicated(geo)),]
tsne <- Rtsne(geo)
summary(fit.geo)
summary(geo)
auc(lab, fit.geo$fitted.values)

library(Rtsne)
library(randomForest)

geo <- apply(geo,2,as.factor)
geo <- as.data.frame(geo)
rf.geo <- randomForest(x=geo,y = as.factor(lab), ntree = 200,mtry = 15,do.trace = TRUE,replace = TRUE)
geo.f <- names(geo)#[order(importance(rf.geo), decreasing = T)][1:50]

p <- predict(rf.geo, geo, type = "prob")
pROC::auc(lab, p[,2])
##
tr <- tr.new[-samp,geo.f]
ts <- tr.new[samp,geo.f]
tr <- as.data.frame(apply(tr,2,as.factor))
ts <- as.data.frame(apply(ts,2,as.factor))
rf.geo <- randomForest(x=tr,y = as.factor(trY), ntree = 300,mtry = 15,do.trace = TRUE,replace = TRUE)

#ensure similar levels in ts

t <- lapply(names(ts), function(x){factor(ts[,x], levels = unique(tr[,x]))})
t <- as.data.frame(t)
names(t) <- names(ts)
pred <- predict(rf.geo, t, type = "prob")
#pred <- pred[,2]
auc(trY, pred[,2])

# tr1 <- tr.new[-samp,geo.f]
# ts1 <- tr.new[samp,geo.f]
# fitglm <- glm(trY~., tr, family = "binomial")
# summary(fit)
# pROC::auc(trY, fit$fitted.values)
# pred1 <- predict(fit, ts, type= "response")
# auc(tsY,pred1)
geo.en <- cbind(ts,pred[,2])

## prepare ensemble data for xgb
toMatch <- c("Geo", "Sale", "Personal", "Property", "Coverage")
t <- unique (grep(paste(toMatch,collapse="|"), 
             names(tr.new), value=TRUE))
ts.en <- tr.new[samp,!(names(tr.new) %in% t)]

names(sales.en)[ncol(sales.en)] <- "sales.pred"
names(personal.en)[ncol(personal.en)] <- "personal.pred"
names(prop.en)[ncol(prop.en)] <- "prop.pred"
names(cover.en)[ncol(cover.en)] <- "cover.pred"
names(geo.en)[ncol(geo.en)] <- "geo.pred"

ts.en <- cbind(ts.en,sales.en,personal.en,prop.en,cover.en,geo.en)
ts.en <- cbind(ts.en,sales.en$sales.pred,personal.en$personal.pred,prop.en$prop.pred,cover.en$cover.pred,geo.en$geo.pred)
#fit xgb on ensemble data
meanp <- mean(ts.en[,grepl(".pred",names(ts.en))])
t <- ts.en[,grepl(".pred",names(ts.en))]
meanp <- apply(t,1, mean)
auc(tsY, meanp)

features <- names(ts.en)
for(f in features){
  if (class(ts.en[[f]]) == "factor" | class(ts.en[[f]]) == "character"){
    ts.en[[f]] <- as.character(ts.en[[f]])
    levels <- unique(c(ts.en[[f]]))
    ts.en[[f]] <- as.integer(factor(ts.en[[f]], levels=levels))
  }
}
sum(is.na(ts.en))
ts.en[is.na(ts.en)] <- 0

ts.en$count_miss <- apply(ts.en, 1, function(t) {length(which(t==-1))})
ts.en$count_zero <- apply(ts.en, 1, function(t) {length(which(t== 0))})
ts.en$count_na <- apply(ts.en, 1, function(t) {length(which(t== -99))})

val <- sample(nrow(ts.en),20000)


dtrain <- xgb.DMatrix(data.matrix(ts.en[-val,]),label = tsY[-val])
dval <- xgb.DMatrix(data.matrix(ts.en[val,]),label = tsY[val])
watchlist = list(val = dval, train = dtrain)
fit <- fit.model(watchlist, dtrain)

imp.features <- xgb.importance(feature_names = names(ts.en), model = fit)

pred <- predict(fit,data.matrix(ts.en[-val,]))
auc(tsY[-val],pred)
fitglm <- glm(tsY~., data = ts.en, "binomial")
auc(tsY, fitglm$fitted.values)

fit.model <- function(watchlist, dtrain){
  param <- list("objective" = "binary:logistic",
                "booster" = "gbtree",
                "eval_metric" = "auc",    # evaluation metric 
                "nthread" = 4,   # number of threads to be used 
                "max_depth" = 8,    # maximum depth of tree 
                "eta" = 0.03,    # step size shrinkage 
                "gamma" = 0,    # minimum loss reduction 
                "subsample" = 0.4,    # part of data instances to grow tree 
                "colsample_bytree" = 0.6  # subsample ratio of col
  )
  
  fit <- xgb.train(params = param, data = dtrain, nrounds = 3000, watchlist = watchlist, print.every.n = 20, early.stop.round = 50, maximize = TRUE)
  
  return(fit)
}




