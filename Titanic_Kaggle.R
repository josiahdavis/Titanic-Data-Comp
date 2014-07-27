# Kaggle Competition
rm(list = ls()); gc()
set.seed(413487)
library(caret)
library(rpart)

setwd("C:/Users/josdavis/Documents/GitHub/Titanic-Data-Comp")
test <- read.csv("test.csv", header = TRUE)
train <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Their last name
train$last.name <- strsplit(as.character(train$Name), ",")
train$last.name <- as.factor(unlist(train$last.name)[seq(1, 1782, 2)]) #take every other element

# How many additional family members on board
library(plyr)
train <- ddply(train, c("last.name"), function(x)cbind(x, family.no = length(unique(x$Name)) - 1)) 




fm <- c()
rm(i, j)

train$spouse <- 0
# Whether the person has a spouse on board
# Loop through each passanger, check for relatives, opposite gender, age

for (i in levels(train$Name)){
  if(train[train$Name == i,]$family.no > 0 & train[train$Name == i,]$Age > 15 & !is.na(train[train$Name == i,]$Age)){
    
    # Loop through all family members
    # Create a dataframe of adult family members
    fm <- (train[train$last.name == train[train$Name == i,]$last.name & train[train$Name == i,]$Age > 15, ])
    fm$Name <- droplevels(fm$Name)
    
    for (j in levels(fm$Name)){
      # print(paste(j, ", ", train[train$Name == j,]$Name))
      train$spouse <-  ifelse(any(fm[fm$Name != j,]$Sex != fm[fm$Name == j,]$Sex), 1, 0)
    }
  }
  fm <- c()
}


summary(as.factor(train$married))
train[train$married == 1,]

#################
# Split up into train/test sets
#################

idx <- createDataPartition(train[,3], times = 1, p = 0.60, list = FALSE)
train.1 <- train[idx,]
train.2 <- train[-idx,]

formula <- as.formula("Survived ~ Pclass + Sex + Age + SibSp + 
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
# Impute missing values
library(missForest)
train.1.i <- missForest(train.1[c("Survived", "Pclass", "Sex", "Age",
                                  "SibSp", "Fare", "Embarked", "family.no")], verbose = TRUE)$ximp
train.2.i <- missForest(train.2[c("Survived", "Pclass", "Sex", "Age",
                                  "SibSp", "Fare", "Embarked", "family.no")], verbose = TRUE)$ximp

library(randomForest)
m.forest <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + 
                           Fare + Embarked + family.no, data = train.1.i)
p.forest <- predict(m.forest, newdata = train.2.i)
a.forest <- sum(p.forest == train.2.i$Survived) / length(train.2.i$Survived)
a.forest

# People I got wrong that actually lived
summary(train.2[p.forest == 0 & train.2.i$Survived == 1,][c("Age", "Sex", "Pclass")])

# People I got right that actually lived
summary(train.2[p.forest == 1 & train.2.i$Survived == 1,][c("Age", "Sex", "Pclass")])

# People I got wrong that actually died
summary(train.2[p.forest == 1 & train.2.i$Survived == 0,][c("Age", "Sex", "Pclass")])

# People I got right that actually died
summary(train.2[p.forest == 0 & train.2.i$Survived == 0,][c("Age", "Sex", "Pclass")])




confusionMatrix(p.forest, train.2.i$Survived)

# I got 0.8651685 (seed = 413487)
# I got 0.8595506 (seed = 25)
# I got 0.8398876 (seed = 543)
# I got 0.8455056 (seed = 15142)
# I got 0.8426966 (seed = 98143)