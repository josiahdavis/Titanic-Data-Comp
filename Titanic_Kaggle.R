# Kaggle Competition
rm(list = ls()); gc()
set.seed(98143)
library(caret)
library(rpart)

setwd("C:/Users/josdavis/Documents/GitHub/Titanic-Data-Comp")
test <- read.csv("test.csv", header = TRUE)
train <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Add Number of Family Members
train$last.name <- strsplit(as.character(train$Name), ",")
train$last.name <- as.factor(unlist(train$last.name)[seq(1, 1782, 2)]) #take every other element

# How many additional family members on board
library(plyr)
train <- ddply(train, c("last.name"), function(x)cbind(x, family.no = length(unique(x$Name)) - 1))

# Get Titles
getTitle <- function(data) {
  title.dot.start <- regexpr("\\,[A-Z ]{1,20}\\.", data$Name, TRUE)
  title.comma.end <- title.dot.start + attr(title.dot.start, "match.length")-1
  data$Title <- substr(data$Name, title.dot.start+2, title.comma.end-1)
  return (data$Title)
}   

train$Title <- as.factor(getTitle(train))

# Impute missing values
library(missForest)
train <- missForest(train[c("Survived", "Pclass", "Sex", "Age",
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

idx <- createDataPartition(train[,3], times = 1, p = 0.60, list = FALSE)
train.1 <- train[idx,]
train.2 <- train[-idx,]

formula <- as.formula("Survived ~ Pclass + Sex + Age.Fill + SibSp + 
                      Fare + Embarked + family.no")


#################
# Decision Tree
#################
m.tree <- rpart(formula, data = train.1)
p.tree <- predict(m.tree, newdata = train.2)
a.tree <- sum(round(p.tree, 0) == train.2$Survived) / length(train.2$Survived)
a.tree # I got 0.8146067

confusionMatrix(round(p.tree, 0), train.2$Survived)

#################
# Random Forest
##################

library(randomForest)
m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age.Fill + SibSp + 
                           Fare + Embarked + family.no, data = train.1)
p.forest <- predict(m.forest, newdata = train.2)
a.forest <- sum(p.forest == train.2$Survived) / length(train.2$Survived)
a.forest

confusionMatrix(p.forest, train.2$Survived)

# People I got wrong that actually lived
summary(train.2[p.forest == 0 & train.2.i$Survived == 1,][c("Age", "Sex", "Pclass")])

# People I got right that actually lived
summary(train.2[p.forest == 1 & train.2.i$Survived == 1,][c("Age", "Sex", "Pclass")])

# People I got wrong that actually died
summary(train.2[p.forest == 1 & train.2.i$Survived == 0,][c("Age", "Sex", "Pclass")])

# People I got right that actually died
summary(train.2[p.forest == 0 & train.2.i$Survived == 0,][c("Age", "Sex", "Pclass")])

# I got 0.8651685 (seed = 413487)
# I got 0.8595506 (seed = 25)
# I got 0.8398876 (seed = 543)
# I got 0.8455056 (seed = 15142)
# I got 0.8426966 (seed = 98143)

##################
# Boosted Trees
##################

library(ada)
m.boost <-ada(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                Fare + Embarked + family.no, data=train.1, verbose=TRUE,na.action=na.rpart)
p.boost <-predict(m.boost, newdata=train.2, type="vector")

a.boost <- sum(p.boost==train.2$Survived)/length(p.boost)
confusionMatrix(p.boost, train.2$Survived)

# People I got wrong that actually lived
summary(train.2[p.boost == 0 & train.2$Survived == 1,][c("Age", "Sex", "Pclass")])
# People I got right that actually lived
summary(train.2[p.boost == 1 & train.2$Survived == 1,][c("Age", "Sex", "Pclass")])
# People I got wrong that actually died
summary(train.2[p.boost == 1 & train.2$Survived == 0,][c("Age", "Sex", "Pclass")])
# People I got right that actually died
summary(train.2[p.boost == 0 & train.2$Survived == 0,][c("Age", "Sex", "Pclass")])

##################
# Neural Networks
##################
library(neuralnet)
m.nn <- neuralnet(Survived ~ Pclass + Sex + Age + SibSp + 
                    Fare + family.no, 
                  hidden = 6, data = train.1.n, threshold = 0.05)

# Perform out of sample predictions
p.nn <-ifelse(compute(m.nn,train.2.n)$net.result>=0.5, 1, 0)

# % Correctly predicted assessment
neuralnetworks.accuracy <- sum(neuralnetworks.prediction==y.pred)/length(neuralnetworks.prediction)
