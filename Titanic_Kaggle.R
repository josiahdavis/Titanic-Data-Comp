# Kaggle Competition
rm(list = ls()); gc()
set.seed(98143)
library(caret)
library(rpart)
library(party)
library(plyr)
library(randomForest)
library(neuralnet)
library(ada)
library(e1071)
library(missForest)

# TO-DO: 
# Add spouse variable

setwd("C:/Users/josdavis/Documents/GitHub/Titanic-Data-Comp")
test <- read.csv("test.csv", header = TRUE)
train <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Want to determine the number of family members
# Combine data sets 

test$Survived <- 5 # Create the variable for now
test$dat <- "test"
train$dat <- "train"
combined <- rbind(train, test)

# Get last name
combined$last.name <- strsplit(as.character(combined$Name), ",")
combined$last.name <- as.factor(unlist(combined$last.name)[seq(1, 2618, 2)]) #take every other element

# Count the number of names by last name
combined <- ddply(combined, c("last.name"), function(x)cbind(x, family.no = length(unique(x$Name)) - 1))

# Get Titles (for age imputation mainly)
getTitle <- function(data) {
  title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
  title.comma.end <- title.dot.start + attr(title.dot.start, "match.length")-1
  data$Title <- substr(data$Name, title.dot.start+2, title.comma.end-1)
  return (data$Title)
}   
combined$Title <- as.factor(getTitle(combined))
rm(getTitle)

# Impute missing values (does not change order)
combined.i <- missForest(combined[c("Pclass", "Sex", "Age", "Parch",
                                  "SibSp", "Fare", "Embarked", "family.no", "Title")], verbose = FALSE)$ximp

combined <- cbind(combined.i, combined[c("Name", "PassengerId", "dat", "Survived", "Ticket", "Cabin")])
rm(combined.i)

# Did any of the passengers on the ticket live?
combined <- ddply(combined, c("Ticket"), function(x)cbind(x, any.lived = any(x$Survived == 1)))

# Did any of the passengers on the ticket die?
combined <- ddply(combined, c("Ticket"), function(x)cbind(x, any.died = any(x$Survived == 0)))

# Split the data back to original training and testing sets
test <- combined[combined$dat == "test",]
train <- combined[combined$dat == "train",]

# train <- merge(combined[combined$dat == "train",], train[,c("Survived", "Name")],
#                  by = "Name", all.x = TRUE, all.y = TRUE)

rm(combined)

# Create a numeric version as well
# train.n <- train[c("Survived", "Pclass", "Sex", "Age",
#                             "SibSp", "Fare", "Embarked", "family.no")]
# train.n$Sex <- ifelse(train$Sex == "male", 1, 0)
# train.n$Embarked <- as.numeric(train$Embarked)
# train.n <- as.matrix(train.n)

formula <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + 
                      Embarked + Fare + family.no + any.lived")

# Create an empty matrix for the modeling results
iterations = 3
results = data.frame(matrix(0, 5 * iterations, 4, dimnames = 
                              list(NULL, c("Model", "Accuracy", "Sensitivity", "Specificity"))))

el <- vector("list", iterations)

models <- list(logit = el, 
               forest = el, 
               boost = el,
               nb = el,
               ensemble = el)
rm(el)

for(i in 1:iterations){

  #################
  # Split up into train/test sets
  #################
  idx <- createDataPartition(train[,3], times = 1, p = 0.60, list = FALSE)
  train.1 <- train[idx,]; train.2 <- train[-idx,]
  #train.1.n <- train.n[idx,]; train.2.n <- train.n[-idx,]
  rm(idx)
  
  #################
  # Logit
  #################
  m.logit <- glm(Survived~Pclass+Sex+Age+SibSp+Fare+family.no+any.lived, family=binomial(logit), data=train)
  models$logit[[i]] <- m.logit
  p.logit <- predict(m.logit, type = "response", newdata = train.2)
  r <- confusionMatrix(round(p.logit, 0), train.2$Survived)
  results[i,] <- c("Logit", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.logit)
  
  #################
  # Random Forest
  #################
  m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                             Fare + Embarked + family.no + any.lived, data = train.1)
  models$forest[[i]] <- m.forest
  p.forest <- predict(m.forest, newdata = train.2)
  r <- confusionMatrix(p.forest, train.2$Survived)
  results[iterations+i,] <- c("Random Forest", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.forest)
  
  #################
  # Conditional Inference Forest
  ##################
  m.cforest <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                         Fare + Embarked + family.no + any.lived, data = train)
  p.cforest <-  predict(m.cforest, newdata = train)
  r <- confusionMatrix(p.cforest, train$Survived) 
  
  ##################
  # Boosted Trees
  ##################
  m.boost <-ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                  Fare + Embarked + family.no, data=train, verbose=TRUE,na.action=na.rpart)
  models$boost[[i]] <- m.boost
  p.boost <-predict(m.boost, newdata=train, type="vector")
  r <- confusionMatrix(p.boost, train$Survived)
  results[iterations*2+i,] <- c("Boosted Trees", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.boost)
   
#   ##################
#   # Neural Networks
#   ##################
#   m.nn <- neuralnet(formula, hidden = 6, data = train.1.n, threshold = 0.05)
#   p.nn <-ifelse(compute(m.nn,train.2.n[,2:8])$net.result>=0.5, 1, 0)
#   a.nn <- sum(p.nn==train.2.n[,1])/length(p.nn)
#   a.nn 
#   confusionMatrix(p.nn, train.2.n[,1])
  
  ##################
  # Naive Bayes 
  ##################
  m.nb <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Fare + Embarked + family.no, data = train)
  models$nb[[i]] <- m.nb
  p.nb<-predict(m.nb,newdata=train.2)
  r <- confusionMatrix(p.nb, train.2$Survived)
  results[iterations*3+i,] <- c("Naive Bayes", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.nb)
  
  ##################
  # Ensemble Model (Random Forest)
  ##################
  pred <- data.frame(Survival=train.2$Survived,
                     logit=round(p.logit, 0), 
                     bt=p.boost, 
                     rf=p.forest,
                     nb=p.nb)
  m.ensemble <- randomForest(as.factor(Survival) ~ logit + bt + rf + nb, data = pred)
  models$ensemble[[i]] <- m.ensemble
  p.ensemble <- predict(m.ensemble, newdata = pred)
  r <- confusionMatrix(p.ensemble, pred$Survival)
  results[iterations*4+i,] <- c("Ensemble", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, pred, m.ensemble, p.ensemble, p.logit, p.boost, p.forest, p.nb)

}

rm(i, formula)

results$Accuracy <- round(as.numeric(results$Accuracy),4)
results$Sensitivity <- round(as.numeric(results$Sensitivity),4)
results$Specificity <- round(as.numeric(results$Specificity),4)
results

results.avg <- ddply(results, ~Model, summarise, avg = round(mean(Accuracy), 5))
results.avg[order(results.avg$avg, decreasing = TRUE),]

#################
# Make the final preditions
#################

test$logit <- round(predict(models$logit[[1]], type = "response", newdata = test),0)
test$rf <- as.numeric(predict(models$forest[[1]], newdata = test))-1
test$bt <- as.numeric(predict(models$boost[[1]], newdata = test))-1
test$nb <- as.numeric(predict(models$nb[[1]], newdata = test))-1

test$Survived <- predict(models$ensemble[[1]], 
                         newdata = test[c("logit", "rf", "bt","nb")])

test$Survived <- round(predict(m.logit, type = "response", newdata = test),0) # Trying a single logit model
test$Survived <- round(predict(m.logit, type = "response", newdata = test),0) # Trying a single cf model

submission <- test[c("PassengerId", "Survived")]
write.csv(submission, "predictions.csv", row.names = FALSE)

tuneRF <- train(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                  Fare + Embarked + family.no, method = "rf", train, 
                  trControl = trainControl(method = "cv", number = 10),
                  metric = "Accuracy")
names(tuneRF)
m.forest <- tuneRF$finalModel
test$Survived <- predict(tuneRF$finalModel, newdata = test) # Trying a tuned RF

pred <- data.frame(Survival=train$Survived,
                   logit=round(p.logit, 0), 
                   bt=as.numeric(p.boost)-1, 
                   rf=as.numeric(p.forest)-1)
test$Survived <- ifelse(rowSums(test[c("logit", "bt", "rf")])>1, 1, 0) # TRying a simple average
