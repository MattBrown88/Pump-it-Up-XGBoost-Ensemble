#Matt Brown                email: matthew.brown.iowa@gmail.com
#Location: Washington, DC
#Website: www.DrivenData.org
#Competition: Pump it Up: Data Mining the Water Table
#Solution is 9th out of 2017 participants

#Software Tools: XGBoost package in R

#Brief Model Description: Ensemble of 11 XGBoost models with equal weight to each solution

#Feature Selection
#The original data set contained 40 variables. I reduced it down to 26 variables by removing variables
#that were similar/duplicates of other variables.

#After removing duplicates, I used xgb.importance function to remove variables that did not 
#improve the model. I used trial and error to remove and then test to see how the model performed.
#I also modified some variables to reduce the effects of missing data. Specific modifications and
#explanation is shown in the 'Feature Selection' section of the code.

#Load required libraries
library(xgboost)
library(Matrix)
library(MatrixModels)
library(data.table)


#Read in test dataset
test<-read.csv("C:/Users/580010/Desktop/Kaggle/DrivenData -Pump It Up/test.csv")

#Create new column in test 
test$status_group <- 0

#Read in train and label datasets
train<-read.csv("C:/Users/580010/Desktop/Kaggle/DrivenData -Pump It Up/train.csv")
label<-read.csv("C:/Users/580010/Desktop/Kaggle/DrivenData -Pump It Up/TrainingLabels.csv")

#Subset label so that it only contains the label (target variable)
label <- subset(label, select = status_group )

#Combine the train and the label data sets
train<-cbind(train,label)

#Create new status_group column so test and train and same number of columns. 
#Required when building the model
train$status_group<-0

#Designate columns as train and test
train$tst <- 0
test$tst <- 1

#Combine train and test into one dataset
data<- rbind(train,test)

#######Feature Engineering###########
#Changed date_recorded to the Date variable type
data$date_recorded<-as.Date(data$date_recorded)

#Set region_code and district_code as a factors
data$region_code<-factor(data$region_code)
data$district_code<-factor(data$district_code)

#Modified construction year so that it starts at 0 and counts up 
#E.g. 1960=0, 1961=1, etc.
min_year<-1960
data$construction_year<-data$construction_year-min_year

#Set missing construction_year values to to the median of the construction year values
data$construction_year[data$construction_year<0]= median(data$construction_year[data$construction_year>0])

#Set missing gps_height values to the median of gps_height
data$gps_height[data$gps_height==0]=median(data$gps_height[data$gps_height>0])

#data$amount_tsh[data$amount_tsh==0]<-median(data$amount_tsh[data$amount_tsh>0])

#######Feature Selection#######
#Removed duplicate/similar and variables which did not improve the model by setting the 

#Not sure what num_private is, and it didn't improve the model.
data$num_private<-NULL

#Removed because only one unique value
data$recorded_by<-NULL

#Removed because there were too many unique values
data$wpt_name<-NULL

#Removed because both are similar to extraction_type_class
data$extraction_type_group<-NULL
data$extraction_type<-NULL

#Removed because similar to payment
data$payment_type<-NULL

#Removed because similar to quality_group.
data$water_quality<-NULL

#data$basin<-NULL
data$scheme_management<-NULL

#Removed because all are location variables. Long and lat should be sufficient to account for
#location.
data$district_code<-NULL
data$region<-NULL
data$region_code<-NULL
data$subvillage<-NULL
data$ward<- NULL

#Removed because similar to waterpoint_type
data$waterpoint_type_group<-NULL

#Removed because duplicate of quantity
data$quantity_group<-NULL

#Removed because too many unique values. Another option is to group the installers to reduce the
#number of unique values
data$installer<-NULL

#Separate data into train and test set
data_train <- data[data$tst==0,]
data_test <- data[data$tst==1,]

#Create test set that doesn't contain the ID column. I did this because the test and train
#datsets need to have the same number of columns when making predictions.
data_test.noID<-subset(data_test, select = -id)

#Remove the id and status group columns from the train dataset. I don't want these columns
#to affect the the model
data_train<-subset(data_train, select = c(-id,-status_group))

#Convert data frames to numeric matrices. Xgboost requires user to enter data as a numeric matrix
data_test.noID <- as.matrix(as.data.frame(lapply(data_test.noID, as.numeric)))
data_train <- as.matrix(as.data.frame(lapply(data_train, as.numeric)))
label<-as.numeric(label$status_group)

#Create a xgb.DMatrix which is the best format to use to create an xgboost model
train.DMatrix <- xgb.DMatrix(data = data_train,label = label, missing = NA)

#For loop to run model 11 time with different random seeds. Using an ensemble technique such as this
#improved the model performance

#Set i=2 because the first column is for the id variable
i=2

#Create data frame to hold the 11 solutions developed by the model
solution.table<-data.frame(id=data_test[,"id"])
for (i in 2:12){
  #Set seed so that the results are reproducible
  set.seed(i)

#Cross validation to determine the number of iterations to run the model.
#I tested this model with a variety of parameters to find the most accurate model
xgb.tab = xgb.cv(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                 nrounds = 500, nfold = 4, early.stop.round = 10, num_class = 4, maximize = FALSE,
                 evaluation = "merror", eta = .2, max_depth = 12, colsample_bytree = .4)

#Create variable that identifies the optimal number of iterations for the model
min.error.idx = which.min(xgb.tab[, test.merror.mean])

#Create model using the same parameters used in xgb.cv
model <- xgboost(data = train.DMatrix, objective = "multi:softmax", booster = "gbtree",
                 eval_metric = "merror", nrounds = min.error.idx, 
                 num_class = 4,eta = .2, max_depth = 14, colsample_bytree = .4)

#Predict. Used the data_test.noID because it contained the same number of columns as the train.DMatrix
#used to build the model.
predict <- predict(model,data_test.noID)

#Modify prediction labels to match submission format
predict[predict==1]<-"functional"
predict[predict==2]<-"functional needs repair"
predict[predict==3]<-"non functional"

#View prediction
table(predict)

#Add the solution to column i of the solutions data frame. This creates a data frame with a column for
#each prediction set. Each prediction is a vote for that prediction. Next I will count the number of votes
#for each prediction as use the element with the most votes as my final solution.
solution.table[,i]<-predict
}

#Count the number of votes for each solution for each row
solution.table.count<-apply(solution.table,MARGIN=1,table)

#Create a vector to hold the final solution
predict.combined<-vector()

x=1
#Finds the element that has the most votes for each prediction row
for (x in 1:nrow(data_test)){
  predict.combined[x]<-names(which.max(solution.table.count[[x]]))}

#View the number of predictions for each classification
table(predict.combined)

#Create solution data frame
solution<- data.frame(id=data_test[,"id"], status_group=predict.combined)

#View the first five rows of the solution to ensure that it follows submission format rules
head(solution)

#Create csv submission file
write.csv(solution, file = "Water_solution - xgboost 45.csv", row.names = FALSE)

#Calculate the importance of each variable to the model.
#Used this function to remove variables from the model variables which don't contribute to the model.
importance <- xgb.importance(feature_names = colnames(data_train), model =model)
importance
xgb.plot.importance(importance_matrix = importance)

#score .8247
