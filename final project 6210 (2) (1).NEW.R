data.df <- read.csv("Telco-Customer-Churn.csv", header = TRUE) #load data
dim <- dim(data.df) #dim of the data frame
dim
head(data.df) #first six rows
View(data.df) #show all the data


#different subsets of the data
data.df[1:10,20]
data.df[1:10, ]
data.df[5000,1:19]
data.df[754,c(1:2,6:8,17:21)]
data.df[ ,20]
class(data.df$Churn) #Churn is character
data.df$TotalCharges[1:10]
length(data.df$TotalCharges)
sum(is.na(data.df$TotalCharges))
mean(data.df$TotalCharges,na.rm = TRUE)
summary(data.df$TotalCharges)


# Install packages
install.packages(c("forecast",
                   "ggplot2",
                   "gplots",
                   "reshape",
                   "GGally",
                   "MASS"))

# Library packages
library(forecast)
library(ggplot2)
library(gplots)
library(reshape)
library(GGally)
library(MASS)


#Bar plot relationship between SeniorCitizen and Churn
ggplot(data.df, aes(x = as.factor(SeniorCitizen),fill= Churn),stat="identity")+
  geom_bar() +
  labs(title = "Relationship between SeniorCitizen and Churn",
       x = "SeniorCitizen",
       fill = "Churn Status")

# side by side boxplot
par(mfcol=c(1,2))
boxplot(data.df$tenure ~ data.df$Churn,xlab = "Churn",ylab="Tenure")
boxplot(data.df$MonthlyCharges ~ data.df$Churn,xlab="Churn",ylab="MonthlyCharges")


#Jitter Plot of Monthly Charges by Contract and Churn
ggplot(data.df, aes(x = Contract, y = MonthlyCharges, color = Churn)) +
  geom_jitter(width = 0.3, alpha = 0.7) +
  labs(title = "Jitter Plot of Monthly Charges by Contract and Churn",
       x = "Contract",
       y = "Monthly Charges",
       color = "Churn Status") +
  theme_minimal()


# Create bar plot (Partner and Churn)
ggplot(data.df, aes(x = as.factor(Partner), fill= Churn)) +
  geom_bar() +
  labs(title = "Bar Plot of Partner with Churn",
       x = "Partner",
       y = "Churn") +
  theme_minimal()


#

ggplot(data.df, aes(x = InternetService,y=MonthlyCharges, fill = Churn)) +
  geom_boxplot() +
  labs(title = "Box Plot of InternetService and MonthlyCharges",
       x = "InternetService",
       y = "MonthlyCharges")


ggplot(data.df, aes(x = InternetService, fill = StreamingMovies)) +
  geom_bar(position = "dodge") +
  labs(title = "Bar Plot of InternetService and StreamingMovies",
       x = "InternetService",
       y = "Count",
       fill = "StreamingMovies")



#
ggplot(data.df,aes(x=PaymentMethod,fill=Churn),stat="identity",position="Dodge")+
  geom_bar()+
    labs(title = "Bar Plot of PaymentMethod and Churn",
         x="PaymentMethod",
         y="Churn")+
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#Stacked barplot
ggplot(data.df, aes(x = PaymentMethod, fill = Contract)) +
  geom_bar(position = "stack") +
  labs(title = "Stacked Bar Plot of PaymentMethod and Contract with Churn",
       x = "PaymentMethod",
       y = "Count",
       fill = "Contract") +
  facet_wrap(~ Churn) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))


# Create plot of TotalCharges and Churn
boxplot(TotalCharges ~ Churn, data = data.df, col = c("blue", "red"), main = "Boxplot for Total Charges")


# Data Preprocessing
# Boxplot to visualize outliers in Total Charges
boxplot(TotalCharges ~ Churn, data = data.df, col = c("blue", "red"), main = "Boxplot for Total Charges")

summary(data.df$TotalCharges)

# Subset the dataframe to show rows where TotalCharges is NA
na_rows <- data.df[is.na(data.df$TotalCharges), ]

# Print the subsetted dataframe
print(na_rows)



# Calculate IQR for TotalCharges column
Q1 <- quantile(data.df$TotalCharges, 0.25,na.rm=TRUE)
Q3 <- quantile(data.df$TotalCharges, 0.75,na.rm =TRUE)
IQR <- Q3 - Q1

# Define outliers
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

# Remove outliers from TotalCharges column
outliers<- data.df[data.df$TotalCharges >= lower_bound & data.df$TotalCharges <= upper_bound, ]

#print outliers
print(outliers)





#Data Reduction and dimension reduction
# Convert "Churn" to numeric
data.df$Churn <- ifelse(data.df$Churn == "Yes", 1, 0)
data.df$Churn

# Remove rows with missing values
data.df <- na.omit(data.df)


# List of numeric variables
pca_components <- c("TotalCharges", "MonthlyCharges", "tenure", "Churn","SeniorCitizen")

pca_result <- prcomp(data.df[pca_components], scale. = TRUE)


pca_result

biplot(pca_result)
biplot(pca_result,choices = c(1,3))



#Variance explained by each pc
pr.var <- pca_result$sdev^2
pve <- pr.var/sum(pr.var)
pve

#plot variance explained for each pc
plot(pve,xlab = "Principal Component",
     ylab="Proportion of Variance Explained",
     ylim = c(0,1),type = "b")

# plot cumulative proportion of variance explaines
plot(cumsum(pve),xlab = "Principal Component",
     ylab="Cumulative Proportion Variance Explaines",
     ylim=c(0,1),type = "b")

#Data Partition
k <- 10  # Number of folds
n <- nrow(data.df)  # Number of observations
subset_size <- floor(n / k)  # Size of each fold
remaining <- n %% k  # Number of remaining observations
# Create an empty list to store the folds
folds <- vector("list", k)
# Shuffle the indices of the dataset
indices <- sample(n)
# Partition the dataset into k folds
for (i in 1:k) {
  start_index <- (i - 1) * subset_size + 1
  end_index <- min(i * subset_size, n)
  if (i <= remaining) {
    end_index <- end_index + 1
  }
  fold_indices <- indices[start_index:end_index]
  folds[[i]] <- data.df[fold_indices, ]
}
# Perform cross-validation
for (i in 1:k){
# Combine all folds except the current one for training
train_data <- do.call(rbind, folds[-i])
# Use the current fold for validation
validation_data <- folds[[i]]
# Train your model using 'train_data' and evaluate it using 'validation_data'
# For demonstration, you can print the number of observations in each fold
num_observations <- nrow(validation_data)
print(paste("Number of observations in fold", i, "=", num_observations))
}



#Task 5
install.packages(c("rpart", "rpart.plot", "caret", "e1071","gains","randomForest"))

# Load necessary libraries
library(caret)
library(rpart)
library(rpart.plot)
library(gains)
library(e1071)


# Encoding categorical variables
data.df$gender <- as.factor(data.df$gender)
data.df$Churn<- factor(data.df$Churn,levels=c(0,1),labels("No","Yes"))


# Normalizing numerical data
data.df$tenure <- scale(data.df$tenure)

set.seed(123)  # for reproducibility
indexes <- createDataPartition(data.df$Churn, p=0.8, list=FALSE)
train_data <- data.df[indexes, ]
test_data <- data.df[-indexes, ]

model_logit <- glm(Churn ~ ., data=train_data, family=binomial)
logit.reg <- glm(Personal.Loan ~ ., data = train.df, family = "binomial")
options(scipen = 999)
summary(model_logit)
model_rf <- train(Churn ~ ., data=train_data, method="rf",
                  trControl=trainControl(method="cv", number=10))
# Predictions
predictions_logit <- predict(model_logit, test_data, type="response")
predictions_rf <- predict(model_rf, test_data)

# Convert probabilities to binary outcome for logistic regression
predictions_logit <- ifelse(predictions_logit > 0.5, "Yes", "No")

# Evaluation
confusionMatrix(predictions_logit, test_data$Churn)
confusionMatrix(predictions_rf, test_data$Churn)

# ROC for Logistic Regression
roc_logit <- roc(response=test_data$Churn, predictor=as.numeric(predictions_logit))

# ROC for Random Forest
roc_rf <- roc(response=test_data$Churn, predictor=as.numeric(predictions_rf$predictions))

# Plot
plot(roc_logit, col="red")
plot(roc_rf, add=TRUE, col="blue")
legend("bottomright", legend=c("Logistic Regression", "Random Forest"),
       col=c("red", "blue"), lwd=2)

# Extract the necessary columns
data_sub = data[['tenure', 'Churn']]

install.packages("dplyr")

# Load the dplyr package
library(dplyr)

# Select only 'Churn' and 'Tenure'
data <- data %>% select(Churn, tenure)

# Select only 'Churn' and 'Tenure' without using dplyr's pipe operator
data <- select(data, Churn, tenure)

# Build a decision tree model
tree_model <- rpart(Churn ~ tenure, data = train_set, method = "class")


# Verify the columns are correctly included
str(data)


# Attempt to rebuild the decision tree model
tree_model <- rpart(Churn ~ tenure, data = train_set, method = "class")

# Build a decision tree model
tree_model <- rpart(Churn ~ tenure, data = train_set, method = "class")


# Plot the decision tree model
rpart.plot(tree_model, main="Decision Tree for Churn Prediction based on tenure", extra=102)

# Predict using decision tree
tree_pred <- predict(tree_model, test_set, type = "class")
tab_tree <- table(test_set$Churn, tree_pred)
accuracy_tree <- sum(diag(tab_tree)) / sum(tab_tree)

# Print accuracy
print(paste("Accuracy of Decision Tree:", accuracy_tree))



#TASK_6
# Load necessary libraries
library(caret)
library(pROC)
library(e1071)  # Add this line to load the e1071 library

# Read the data
data.df <- read.csv("Telco-Customer-Churn.csv", header = TRUE)

# Convert target variable to factor
data.df$Churn <- as.factor(data.df$Churn)

# Partition the data into training and validation sets
index <- createDataPartition(data.df$Churn, p = 0.7, list = FALSE)
training_set <- data.df[index, ]
validation_set <- data.df[-index, ]

# Build a classification model using Naive Bayes
nb_model <- naiveBayes(Churn ~ ., data = training_set)

# Apply the model to training and validation sets
training_pred <- predict(nb_model, training_set)
validation_pred <- predict(nb_model, validation_set)

# Define a function to calculate evaluation metrics
calculate_metrics <- function(actual, predicted) {
  
  # Confusion Matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Accuracy
  accuracy <- cm$overall['Accuracy']
  
  # Sensitivity (True Positive Rate)
  sensitivity <- cm$byClass['Sensitivity']
  
  # Specificity (True Negative Rate)
  specificity <- cm$byClass['Specificity']
  
  # Precision (Positive Predictive Value)
  precision <- cm$byClass['Pos Pred Value']
  
  # FDR (False Discovery Rate)
  fdr <- 1 - precision
  
  # FOR (False Omission Rate)
  f_or <- 1 - cm$byClass['Neg Pred Value']
  
  # Convert predicted values to numeric
  predicted_numeric <- as.numeric(predicted)
  
  # ROC Curve and AUC
  roc_obj <- roc(as.numeric(actual), predicted_numeric)
  auc <- auc(roc_obj)
  
  # Calculate Lift
  pos_rate <- sum(actual == "Yes") / length(actual)
  lift <- cm$byClass['Sensitivity'] / pos_rate
  
  # Return metrics
  return(list(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, 
              Precision = precision, FDR = fdr, FOR = f_or, AUC = auc, Lift = lift))
}

# Apply the function to training and validation data
training_metrics <- calculate_metrics(training_set$Churn, training_pred)
validation_metrics <- calculate_metrics(validation_set$Churn, validation_pred)

# Display the metrics
print("Training Metrics:")
print(training_metrics)

print("Validation Metrics:")
print(validation_metrics)











