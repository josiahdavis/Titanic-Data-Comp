# Kaggle Competition
rm(list = ls()); gc()
set.seed(98143)
library(caret, rpart, plyr)
library(randomForest, ada, neuralnet)
library(e1071, missForest)

# TO-DO: Enhance family number algorithm

setwd("C:/Users/josdavis/Documents/GitHub/Titanic-Data-Comp")
test <- read.csv("test.csv", header = TRUE)
train.o <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Add Number of Family Members
train.o$last.name <- strsplit(as.character(train.o$Name), ",")
train.o$last.name <- as.factor(unlist(train.o$last.name)[seq(1, 1782, 2)]) #take every other element

# How many additional family members on board
train.o <- ddply(train.o, c("last.name"), function(x)cbind(x, family.no = length(unique(x$Name)) - 1))

# Get Titles
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
                                  "SibSp", "Fare", "Embarked", "family.no", "Title")], verbose = TRUE)$ximp

# Create a numeric version as well
# train.n <- train[c("Survived", "Pclass", "Sex", "Age",
#                             "SibSp", "Fare", "Embarked", "family.no")]
# train.n$Sex <- ifelse(train$Sex == "male", 1, 0)
# train.n$Embarked <- as.numeric(train$Embarked)
# train.n <- as.matrix(train.n)

formula <- as.formula("Survived~Pclass+Sex+Age+SibSp+Fare+Embarked+family.no")



# Create an empty matrix for the modeling results
iterations = 5
results = data.frame(matrix(0, 5 * iterations, 4, dimnames = 
                              list(NULL, c("Model", "Accuracy", "Sensitivity", "Specificity"))))

for(i in 1:iterations){

  #################
  # Split up into train/test sets
  #################
  idx <- createDataPartition(train.o[,3], times = 1, p = 0.60, list = FALSE)
  train.1 <- train[idx,]; train.2 <- train[-idx,]
  #train.1.n <- train.n[idx,]; train.2.n <- train.n[-idx,]
  rm(idx)
  
  #################
  # Logit
  #################
  m.logit <- glm(formula, family = binomial(logit), data = train.1)
  p.logit <- predict(m.logit, type = "response", newdata = train.2)
  r <- confusionMatrix(round(p.logit, 0), train.2$Survived)
  results[i,] <- c("Logit", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.logit)
  
  #################
  # Random Forest
  ##################
  m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                             Fare + Embarked + family.no, data = train.1)
  p.forest <- predict(m.forest, newdata = train.2)
  r <- confusionMatrix(p.forest, train.2$Survived)
  results[iterations+i,] <- c("Random Forest", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, m.forest)
  
  ##################
  # Boosted Trees
  ##################
  m.boost <-ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                  Fare + Embarked + family.no, data=train.1, verbose=TRUE,na.action=na.rpart)
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
  p.ensemble <- predict(m.ensemble, newdata = pred)
  r <- confusionMatrix(p.ensemble, pred$Survival)
  results[iterations*4+i,] <- c("Ensemble", r$overall[1], r$byClass[1], r$byClass[2])
  rm(r, pred, m.ensemble, p.ensemble, p.logit, p.boost, p.forest, p.nb)
  
}

rm(i, formula)

results$Accuracy <- as.numeric(results$Accuracy)
results$Sensitivity <- as.numeric(results$Sensitivity)
results$Specificity <- as.numeric(results$Specificity)

ddply(results, ~Model, summarise, avg = round(mean(Accuracy), 5))

##################
# Who did I get right and wrong? 
##################
# train.2.o[p.forest == 1 & train.2.o$Survived == 0,][c("Name" ,"Age", "Sex", "Pclass", "family.no")]