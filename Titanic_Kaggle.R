# Kaggle Competition
rm(list = ls()); gc()
set.seed(98143)
library(caret)
library(rpart)
library(plyr)
library(randomForest)
library(ada)
library(neuralnet)
library(e1071)


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

# Impute missing values
library(missForest)
train <- missForest(train.o[c("Survived", "Pclass", "Sex", "Age",
                                  "SibSp", "Fare", "Embarked", "family.no", "Title")], verbose = TRUE)$ximp

# Create a numeric version as well
train.n <- train[c("Survived", "Pclass", "Sex", "Age",
                            "SibSp", "Fare", "Embarked", "family.no")]
train.n$Sex <- ifelse(train$Sex == "male", 1, 0)
train.n$Embarked <- as.numeric(train$Embarked)
train.n <- as.matrix(train.n)
str(train.n)

#################
# Split up into train/test sets
#################
idx <- createDataPartition(train.o[,3], times = 1, p = 0.60, list = FALSE)
train.1.o <- train.o[idx,]; train.2.o <- train.o[-idx,]
train.1 <- train[idx,]; train.2 <- train[-idx,]
train.1.n <- train.n[idx,]; train.2.n <- train.n[-idx,]
formula <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + 
                      Fare + Embarked + family.no")

#################
# Decision Tree
#################
m.tree <- rpart(formula, data = train.1)
p.tree <- predict(m.tree, newdata = train.2)
a.tree <- sum(round(p.tree,0) == train.2$Survived) / length(train.2$Survived)
a.tree

confusionMatrix(round(p.tree, 0), train.2$Survived)

#################
# Random Forest
##################
m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                           Fare + Embarked + family.no, data = train.1)
p.forest <- predict(m.forest, newdata = train.2)
a.forest <- sum(p.forest == train.2$Survived) / length(train.2$Survived)
a.forest
confusionMatrix(p.forest, train.2$Survived)

##################
# Boosted Trees
##################
m.boost <-ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                Fare + Embarked + family.no, data=train.1, verbose=TRUE,na.action=na.rpart)
p.boost <-predict(m.boost, newdata=train.2, type="vector")
a.boost <- sum(p.boost==train.2$Survived)/length(p.boost)
confusionMatrix(p.boost, train.2$Survived)

##################
# Neural Networks
##################
m.nn <- neuralnet(formula, hidden = 6, data = train.1.n, threshold = 0.05)
p.nn <-ifelse(compute(m.nn,train.2.n[,2:8])$net.result>=0.5, 1, 0)
a.nn <- sum(p.nn==train.2.n[,1])/length(p.nn)
a.nn # Got 0.8366197183 with seed of 98143
confusionMatrix(p.nn, train.2.n[,1])

##################
# Naive Bayes 
##################
m.nb <- naiveBayes(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                     Fare + Embarked + family.no, data = train.1)
p.nb<-predict(m.nb,train.2)
a.nb <- sum(p.nb==train.2$Survived)/length(p.nb)
a.nb
confusionMatrix(p.nb, train.2$Survived)

##################
# Ensemble Model 
##################
pred <- data.frame(Survival=train.2$Survived,dt=round(p.tree,0), bt=p.boost, rf=p.forest, nn=p.nn, nb=p.nb)

m.ensemble <- randomForest(as.factor(Survival) ~ dt + bt + rf + nn + nb, data = pred)
p.ensemble <- predict(m.ensemble, newdata = pred)
a.ensemble <- sum(p.ensemble == pred$Survival) / length(pred$Survival)
a.ensemble


##################
# Who did I get right and wrong? 
##################
train.2.o[p.forest == 1 & train.2.o$Survived == 0,][c("Name" ,"Age", "Sex", "Pclass", "family.no")]

