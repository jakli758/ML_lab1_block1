setwd("C:/Users/jakob/OneDrive - Linköpings universitet/Uni/Semester_1/ML/Labs/")

#### 1 ####

data <- read.csv("optdigits.csv", header = FALSE)

n <- dim(data)[1]
set.seed(12345)
id <- sample(1:n, floor(n*0.4))
train <- data[id,]
id1 <- setdiff(1:n, id)
set.seed(12345)
id2 <- sample(id1, floor(n*0.3))
valid <- data[id2,]
id3 <- setdiff(id1,id2)
test <- data[id3,]

train
#### 2 ####
library(kknn)
# model that is trained on train dataset and predicts on train dataset
model_train <- kknn(as.factor(V65)~., train = train, test = train, k=30, kernel="rectangular")
train_predictions <- fitted(model_train)

# model that is trained on train dataset and predicts on train dataset
model_test <- kknn(as.factor(V65)~., train = train, test = test, k=30, kernel="rectangular")
test_predictions <- fitted(model_test)

# confusion matrices
conf_matrix_train <- table(predicted_values=train_predictions, true_values=train$V65)
cat("Train data predictions: \n")
conf_matrix_train

conf_matrix_test <- table(predicted_values=test_predictions, true_values=test$V65)
cat("Test data predictions: \n")
conf_matrix_test


# missclassification error
missclassification_error <- function(confusion_matrix){
  # return 1 - (sum of correct classifications / total classifications) 
  return(1-sum(diag(confusion_matrix)) / sum(confusion_matrix))
}
missclassification_error_train <- missclassification_error(conf_matrix_train)
missclassification_error_test <- missclassification_error(conf_matrix_test)

cat("Missclassification Error on train data:", missclassification_error_train, "\n")
cat("Missclassification Error on test data:", missclassification_error_test, "\n")

#TODO: Comment on the quality of predictions for different digits and on the overall prediction quality.


#### 3 ####
display_img <- function(df, index){
  # helper function to rotate the images
  rotate <- function(x) t(apply(x, 2, rev)) # https://stackoverflow.com/questions/16496210/rotate-a-matrix-in-r-by-90-degrees-clockwise
  img <- rotate(rotate(rotate(matrix(unlist(df[index,1:64]), nrow=8,ncol=8)))) # use unlist to avoid "'x' must be numeric matrix error"
  heatmap(img, Colv = NA, Rowv = NA)
}

display_img(train, 1511)

# add column with probability for correct class to df
correct_class_probs <- sapply(1:nrow(train), function(i) {
  model_train$prob[i, as.character(train$V65[i])]
})
train$correct_prob <- correct_class_probs

# add column with predicted class 
train$predicted_value <- fitted(model_train)

# get indices for class 8 in train set
i_eight <- which(train$V65 == 8)

# create df with only eights and sort df to get the images with highest/lowest probs
train_eight <- train[i_eight,]
train_eight_ordered <- train_eight[order(train_eight$correct_prob),]

# visualize two easiest to classify
tail(train_eight_ordered,2)
display_img(data, 981)
display_img(data, 3447)

# visualize three hardest to classify
head(train_eight_ordered,3)
display_img(data, 1793)
display_img(data, 869)
display_img(data, 3591)

# TODO: comment on whether these cases seem to be hard or easy to recognize visually


#### 4 ####

k_values <- 1:30
train_errors <- numeric(length=length(k_values))
val_errors <- numeric(length=length(k_values))

for (K in k_values){
  train_model <- kknn(as.factor(V65)~., train = train, test = train, k=K, kernel="rectangular")
  
  train_predictions <- fitted(train_model)
  train_conf_matrix <- table(predicted_values=train_predictions, true_values=train$V65)
  train_error <- missclassification_error(train_conf_matrix)
  train_errors[K] <- train_error
  
  val_model <- kknn(as.factor(V65)~., train = train, test = valid, k=K, kernel="rectangular")
  
  val_predictions <- fitted(val_model)
  val_conf_matrix <- table(predicted_values=val_predictions, true_values=valid$V65)
  val_error <- missclassification_error(val_conf_matrix)
  val_errors[K] <- val_error
}
train_errors
train_conf_matrix
val_errors
df_errors <- data.frame(K = k_values, train_error = train_errors, val_error=val_errors)

library(ggplot2)
ggplot(df_errors, aes(x = K)) +
  geom_line(aes(y = train_error, color = "Training Error")) +
  geom_line(aes(y = val_error, color = "Validation Error")) +
  labs(
    title = "Training and Validation Errors vs K",
    x = "Number of Neighbors (K)",
    y = "Misclassification Error"
  ) +
  scale_color_manual(
    name = "Error Type",
    values = c("Training Error" = "blue", "Validation Error" = "red")
  )

# TODO: Report the optimal 𝐾𝐾 according to this plot. Finally, estimate the test error
# for the model having the optimal K, compare it with the training and validation
# errors and make necessary conclusions about the model quality.



#### 5 ####

cross_entropy_error <- function(y_true, y_hat){
  loss <- -sum(y_true * log(y_hat + 1e-15)) / length(y_hat)
  return(loss)
}


k_values <- 1:150
train_errors <- numeric(length=length(k_values))
val_errors <- numeric(length=length(k_values))
#-sum(y_true * log(y_hat + 1e-15)) / nrow(y)

for (K in k_values){
  train_model <- kknn(as.factor(V65)~., train = train, test = train, k=K, kernel="rectangular")

  train_probs <- train_model$prob
  train_error <- cross_entropy_error(train$V65, train_probs)
  train_errors[K] <- train_error
  
  val_model <- kknn(as.factor(V65)~., train = train, test = valid, k=K, kernel="rectangular")
  
  val_probs <- val_model$prob
  val_error <- cross_entropy_error(valid$V65, val_probs)
  val_errors[K] <- val_error
}
val_errors
df_errors <- data.frame(K = k_values, train_error = train_errors, val_error=val_errors)

ggplot(df_errors, aes(x = K)) +
  geom_line(aes(y = train_error, color = "Training Error")) +
  geom_line(aes(y = val_error, color = "Validation Error")) +
  labs(
    title = "Training and Validation Errors vs K",
    x = "Number of Neighbors (K)",
    y = "Misclassification Error"
  ) +
  scale_color_manual(
    name = "Error Type",
    values = c("Training Error" = "blue", "Validation Error" = "red")
  )

# TODO What is the optimal 𝐾𝐾 value here? Assuming that response has
#multinomial distribution, why might the cross-entropy be a more suitable choice
#of the error function than the misclassification error for this problem?