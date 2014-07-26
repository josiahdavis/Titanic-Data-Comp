# Kaggle Competition
rm(list = ls()); gc()
set.seed(413487)
library(caret)
library(rpart)

setwd("C:/Users/josdavis/Documents/Personal/Data Sets/Titanic - Kaggle")
test <- read.csv("test.csv", header = TRUE)
train <- read.csv("train.csv", header = TRUE)

##################
# Create Variables
##################

# Their last name
train$last.name <- strsplit(as.character(train$Name), ",")
train$last.name <- unlist(train$last.name)[seq(1, 1782, 2)] #take every other element

# How many additional family members on board
library(plyr)
family.m <- ddply(train, c("last.name"), function(x)length(unique(x$Name)) - 1) 
# Change this to adult family members
names(family.m) <- c("last.name", "family.no")
train <- merge(train, family.m, by.x = "last.name", by.y = "last.name",
              all.x = TRUE, all.y = TRUE)

# People that are possibly married are people with
# 1 - More family members than siblings
# 2 - Above the age of 18
# 3 - Opposite genders
# This would incorrectly label older families, but a good start

# Whether the person has a spouse on board
# Loop through each passanger, check for relatives, opposite gender, age

for (i in levels(train$Name)){
  if(train[train$Name == i,]$family.no > 0 & 
       train[train$Name == i,]$Age > 16 & 
       !is.na(train[train$Name == i,]$Age)){
    
  }
}

y <- 0 
fn2 <- function (N) 
{
  i=1
  while(i <= N) {
    y <- i*i
    print(y)
    i <- i + 1
    
  }
}
fn2(10)

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


confusionMatrix(p.forest, train.2.i$Survived)

# I got 0.8651685 (seed = 413487)
# I got 0.8595506 (seed = 25)
# I got 0.8398876 (seed = 543)
# I got 0.8455056 (seed = 15142)
# I got 0.8426966 (seed = 98143)

# Fill in NA values
# Create spouse-on-board variable
# Create ensemble model
# Create neural network model