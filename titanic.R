# Load and check data
# Load packages
library("caret")

setwd("./Desktop/DataScientist/KaggleCompetition/Titanic/")
train <- read.csv('train.csv')
test  <- read.csv('test.csv')

# Create the training and testing data set
inTrain <- createDataPartition(train$Survived, p=.75, list = FALSE)
training <- train[inTrain,]
testing <- training[-inTrain,]

# Some suspicious variables with high correlation
cor(training$Fare, training$Pclass)

# Is ticket number and passenger ID relevant for their survival??

# Feature enginneering
# Grab title from passenger names
full$Title <- gsub('(.*, )|(\\..*)', '', full$Name)

# Show title counts by sex
table(full$Sex, full$Title)