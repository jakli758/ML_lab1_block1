## Assignment 2. Linear regression and ridge regression

### 1. diabetes preparation

```{r}
parkinsons <- read.csv("parkinsons.csv")
```


We split the diabetes into a training and a test set (60/40) 
```{r}
library(caret)

set.seed(123)
train_index <- creatediabetesPartition(parkinsons$subject, p = 0.6, list = FALSE)

train_set <- parkinsons[train_index, ]
test_set <- parkinsons[-train_index, ]
```

The target variable is `motor_UPDRS`, and the column `subject.` is excluded from the predictors but retained for grouping purposes.

```{r}
# Exclude non-predictor columns
predictor_columns <- setdiff(names(train_set), c("motor_UPDRS", "subject."))

# Scale the predictors in the training set
scaled_train <- train_set
scaled_train[predictor_columns] <- scale(train_set[predictor_columns])

# Save scaling parameters from training set
scaling_params <- attributes(scale(train_set[predictor_columns]))

# Scale the predictors in the test set using training set parameters
scaled_test <- test_set
scaled_test[predictor_columns] <- scale(
  test_set[predictor_columns],
  center = scaling_params$`scaled:center`,
  scale = scaling_params$`scaled:scale`
)

scaled_train <- scaled_train[, c(predictor_columns, "motor_UPDRS", "subject.")]
scaled_test <- scaled_test[, c(predictor_columns, "motor_UPDRS", "subject.")]
```


### 2. Linear Regression Model


```{r}
#Regression model on the training diabetes and removing subject from the model
linear_model <- lm(motor_UPDRS ~ . - subject., diabetes = scaled_train)


# Training and test MSE
train_predictions_train <- predict(linear_model, scaled_train)
train_mse <- mean((scaled_train$motor_UPDRS - train_predictions_train)^2)

test_predictions_test <- predict(linear_model, scaled_test)
test_mse <- mean((scaled_test$motor_UPDRS - test_predictions_test)^2)

train_mse
test_mse

```



The MSE for the train and test set are close in value, that means the model adapts well to other sets than the training

```{r}
summary(linear_model)
```

The variables that contribute significantly are the ones with a p-value <0.05 : age,sex,total_UPDRS, jitter, jitter.Abs,Shimmer.APQ5,Shimmer.APQ11,RPDE,PPE.

### 4. Implementing functions

```{r}
# Log-likelihood function
logLikelihood <- function(theta, sigma, X, T) {
  n <- length(T) # Number of observations
  residuals <- T - X %*% theta # Calculate residuals
  - (n / 2) * log(2 * pi) - n * log(sigma) - (1 / (2 * sigma^2)) * sum(residuals^2)
}
```

```{r}
#Ridge function
ridge <- function(theta, sigma, X, T, lambda) {
  logLikelihood <- logLikelihood(theta, sigma, X, T)
  ridge_penalty <- lambda * sum(theta[-1]^2)
  return(-logLikelihood + ridge_penalty)
}

```

```{r}
#RidgeOpt function
ridgeOpt <- function(lambda, X, T, initial_theta, initial_sigma) {
  # Define the objective function to minimize
  objective_function <- function(params) {
    theta <- params[-length(params)] # Extract theta
    log_sigma <- params[length(params)] # Optimize log(sigma)
    sigma <- exp(log_sigma) 
    
    ridge(theta, sigma, X, T, lambda)
  }
  
  # Combine initial guesses for theta and log(sigma) into a single vector
  initial_log_sigma <- log(initial_sigma)
  initial_params <- c(initial_theta, initial_log_sigma)
  
  # Minimize the objective function
  opt <- optim(
    par = initial_params,
    fn = objective_function,
    method = "BFGS",
    control = list(fnscale = 1) # Default direction for minimization
  )
  
  # Extract optimized theta and sigma
  optimized_theta <- opt$par[-length(opt$par)] 
  optimized_sigma <- exp(opt$par[length(opt$par)]) 
  
  
  return(list(
    optimized_theta = optimized_theta,
    optimized_sigma = optimized_sigma,
    value = opt$value, # Final objective value
    convergence = opt$convergence # Convergence status
  ))
}
```

```{r}
#DF function
#Compute degrees of freedom for Ridge regression
df <- function(lambda, X) {
  # Compute the Ridge hat matrix
  n <- ncol(X) # Number of predictors
  I <- diag(n) # Identity matrix of size n x n
  H <- X %*% solve(t(X) %*% X + lambda * I) %*% t(X) # Hat matrix
  
  # Return the trace of the hat matrix
  return(sum(diag(H)))
}
```


### 4. Computing optimal theta for different lambdas

```{r}
# Define required variables
lambdas <- c(1, 100, 1000) # Regularization parameters
T <- scaled_train$motor_UPDRS # Target variable
X <- as.matrix(cbind(1, scaled_train[, predictor_columns])) # Design matrix with intercept
initial_theta <- rep(0, ncol(X)) # Initialize theta as zeros
initial_sigma <- 1 # Initial guess for sigma


# Placeholder for results
results <- lapply(lambdas, function(lambda) {
  # Optimize parameters for the current lambda
  opt_result <- ridgeOpt(lambda, X, T, initial_theta, initial_sigma)
  
  # Extract optimized parameters
  optimized_theta <- opt_result$optimized_theta
  optimized_sigma <- opt_result$optimized_sigma
  
  # Compute predictions
  train_predictions <- X %*% optimized_theta
  test_X <- as.matrix(cbind(1, scaled_test[, predictor_columns])) # Test matrix with intercept
  test_predictions <- test_X %*% optimized_theta
  
  # Compute MSE for training and test sets
  train_mse <- mean((T - train_predictions)^2)
  test_mse <- mean((scaled_test$motor_UPDRS - test_predictions)^2)
  
  # Compute degrees of freedom
  degrees_of_freedom <- df(lambda, X)
  
  # Return results
  list(
    lambda = lambda,
    optimized_theta = optimized_theta,
    optimized_sigma = optimized_sigma,
    train_mse = train_mse,
    test_mse = test_mse,
    degrees_of_freedom = degrees_of_freedom
  )
})

# Results
for (res in results) {
  cat("\nLambda:", res$lambda, "\n")
  cat("Optimized Sigma:", res$optimized_sigma, "\n")
  cat("Training MSE:", res$train_mse, "\n")
  cat("Test MSE:", res$test_mse, "\n")
  cat("Degrees of Freedom:", res$degrees_of_freedom, "\n")
}
```

#Lambda is the regularization parameter that controls the impact of the ridge penalty. Lambda = 1 is the most appropriate for the ridge regression model because it has the lowest MSE, 6.32 vs 27.80 and 61.78. That means generalization performance to new diabetes is better. The degrees of freedom are the balance between complexity and regularization. A model with low DF may be too simple to explain the target variable with all variables used in the model. High DF may be too complex to fit other diabetes sets than the one used for training. The DF for lambda = 1 are 18.8, so relatively close to the DF of lambda = 100 (14.7). With the MSE and the DF for each lambda we can say that lambda = 1 is the most appropriate in our case.