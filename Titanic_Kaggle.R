# Kaggle Competition
rm(list = ls()); gc()
set.seed(98143)
library(caret, rpart, plyr)
library(randomForest, ada, neuralnet)
library(e1071, missForest)

# TO-DO: 
# Save models within the for loop
# Add spouse variable

setwd("C:/Users/josdavis/Documents/GitHub/Titanic-Data-Comp")
test <- read.csv("test.csv", header = TRUE)
train <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Want to determine the number of family members
# Combine data sets 
test$dat <- "test"
train$dat <- "train"
combined <- rbind(train[,-2], test)

# Get last name
combined$last.name <- strsplit(as.character(combined$Name), ",")
combined$last.name <- as.factor(unlist(combined$last.name)[seq(1, 2618, 2)]) #take every other element

# Count the number of names by last name
combined <- ddply(combined, c("last.name"), function(x)cbind(x, family.no = length(unique(x$Name)) - 1))

# Split the data up again into training and test sets
test <- combined[combined$dat == "test",]
train.o <- merge(combined[combined$dat == "train",], train[,c("Survived", "Name")],
                by = "Name", all.x = TRUE, all.y = TRUE)


# Get Titles (for age imputation mainly)
getTitle <- function(data) {
  title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
  title.comma.end <- title.dot.start + attr(title.dot.start, "match.length")-1
  data$Title <- substr(data$Name, title.dot.start+2, title.comma.end-1)
  return (data$Title)
}   
train.o$Title <- as.factor(getTitle(train.o))
rm(getTitle)

# Impute missing values
train <- missForest(train.o[c("Survived", "Pclass", "Sex", "Age",
                                  "SibSp", "Fare", "Embarked", "family.no", "Title")], verbose = FALSE)$ximp

# Create a numeric version as well
# train.n <- train[c("Survived", "Pclass", "Sex", "Age",
#                             "SibSp", "Fare", "Embarked", "family.no")]
# train.n$Sex <- ifelse(train$Sex == "male", 1, 0)
# train.n$Embarked <- as.numeric(train$Embarked)
# train.n <- as.matrix(train.n)

formula <- as.formula("Survived~Pclass+Sex+Age+SibSp+Embarked+Fare+family.no")

# Create an empty matrix for the modeling results
iterations = 3
results = data.frame(matrix(0, 5 * iterations, 4, dimnames = 
                              list(NULL, c("Model", "Accuracy", "Sensitivity", "Specificity"))))


models <- vector(mode="list", length=iterations)
models <- list(logit = models, 
               forest = models, 
               boost = models,
               nb = models,
               ensemble = models)


for(i in 1:iterations){

  #################
  # Split up into train/test sets
  #################
  idx <- createDataPartition(train.o[,3], times = 1, p = 0.75, list = FALSE)
  train.1 <- train[idx,]; train.2 <- train[-idx,]
  #train.1.n <- train.n[idx,]; train.2.n <- train.n[-idx,]
  rm(idx)
  
  #################
  # Logit
  #################
  m.logit <- glm(Survived~Pclass+Sex+Age+SibSp+Fare+family.no, family = binomial(logit), data = train.1)
  models$logit[[i]] <- m.logit
  p.logit <- predict(m.logit, type = "response", newdata = train.2)
  r <- confusionMatrix(round(p.logit, 0), train.2$Survived)
  results[i,] <- c("Logit", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.logit)
  
  #################
  # Random Forest
  ##################
  m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                             Fare + Embarked + family.no, data = train.1)
  models$forest[[i]] <- m.forest
  p.forest <- predict(m.forest, newdata = train.2)
  r <- confusionMatrix(p.forest, train.2$Survived)
  results[iterations+i,] <- c("Random Forest", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.forest)
  
  ##################
  # Boosted Trees
  ##################
  m.boost <-ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                  Fare + Embarked + family.no, data=train.1, verbose=TRUE,na.action=na.rpart)
  models$boost[[i]] <- m.boost
  p.boost <-predict(m.boost, newdata=train.2, type="vector")
  r <- confusionMatrix(p.boost, train.2$Survived)
  results[iterations*2+i,] <- c("Boosted Trees", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.boost)
#   
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
  m.nb <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Fare + Embarked + family.no, data = train.1)
  models$nb[[i]] <- m.nb
  p.nb<-predict(m.nb,train.2)
  r <- confusionMatrix(p.nb, train.2$Survived)
  results[iterations*3+i,] <- c("Naive Bayes", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.nb)
  
  ##################
  # Ensemble Model 
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

ddply(results, ~Model, summarise, avg = round(mean(Accuracy), 5))

##################
# Who did I get right and wrong? 
##################
# train.2.o[p.forest == 1 & train.2.o$Survived == 0,][c("Name" ,"Age", "Sex", "Pclass", "family.no")]