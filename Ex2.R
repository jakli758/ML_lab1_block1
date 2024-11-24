parkinsons <- read.csv("C:/Users/victo/OneDrive/Bureau/A1_SML/Machine Learning/Labs/Lab1/data/parkinsons.csv")
print(summary(parkinsons[parkinsons$subject==1,]))
print(sum(parkinsons$subject == 1))
      
#library(caret)

set.seed(123)
train_index <- createDataPartition(parkinsons$subject, p = 0.6, list = FALSE)

train_set <- parkinsons[train_index, ]
test_set <- parkinsons[-train_index, ]

nrow(train_set)
nrow(test_set)


# Exclude non-predictor columns
predictor_columns <- setdiff(names(train_set), c("motor_UPDRS", "subject."))

# Scale the predictors in the training set
scaled_train <- train_set
scaled_train[predictor_columns] <- scale(train_set[predictor_columns])

# Save scaling parameters (mean and sd) from training set
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

# Check results
summary(scaled_train)
summary(scaled_test)



# Fit a linear regression model using training data
linear_model <- lm(motor_UPDRS ~ ., data = scaled_train)

# Display summary of the model to check variable significance
summary(linear_model)

#Variables that contribute significantly, 
#the ones with a p-value<0.05 (subject doesnt count) : age,sex,total_UPDRS, jitter, 
#jitter.Abs(keep only one of the two),Shimmer.APQ5,RPDE,PPE


# Calculate training MSE
train_predictions <- predict(linear_model, scaled_train)
train_mse <- mean((scaled_train$motor_UPDRS - train_predictions)^2)
cat("Training MSE:", train_mse, "\n")

# Calculate test MSE
test_predictions <- predict(linear_model, scaled_test)
test_mse <- mean((scaled_test$motor_UPDRS - test_predictions)^2)
cat("Test MSE:", test_mse, "\n")

#The MSE for test and train are close in values, wich indicate that the model fits well for the prediction with the test set

log_likelihood <- function(theta, sigma, X, y) {
  n <- length(y)
  y_pred <- X %*% theta
  ll <- -n / 2 * log(2 * pi) - n / 2 * log(sigma^2) - sum((y - y_pred)^2) / (2 * sigma^2)
  return(ll)
}

# Utilisation avec les coefficients du modèle ajusté
theta <- coef(linear_model)
X <- as.matrix(cbind(1, scaled_train[, predictor_columns]))
y <- scaled_train$motor_UPDRS

# Recalculer sigma à partir des résidus
residuals <- y - X %*% theta
sigma <- sd(residuals)
cat("Estimated sigma:", sigma, "\n")

# Calculer la log-vraisemblance
ll <- log_likelihood(theta, sigma, X, y)
cat("Log-likelihood:", ll, "\n")


##Ridge

ridge_log_likelihood <- function(theta, sigma, X, y, lambda) {
  # theta: Vector of coefficients
  # sigma: Dispersion (standard deviation of residuals)
  # X: Matrix of predictors
  # y: Vector of target values
  # lambda: Ridge penalty parameter
  
  # Compute the negative log-likelihood
  n <- length(y)
  y_pred <- X %*% theta
  neg_log_likelihood <- n / 2 * log(2 * pi) + n / 2 * log(sigma^2) + sum((y - y_pred)^2) / (2 * sigma^2)
  
  # Add Ridge penalty term (lambda * ||theta||^2)
  ridge_penalty <- lambda * sum(theta[-1]^2) # Exclude the intercept (if present) from penalty
  
  # Return penalized negative log-likelihood
  return(neg_log_likelihood + ridge_penalty)
}

# Define lambda (penalty parameter)
lambda <- 1  # Example value

# Compute Ridge penalized log-likelihood
ridge_ll <- ridge_log_likelihood(theta, sigma, X, y, lambda)
cat("Ridge Penalized Negative Log-Likelihood:", ridge_ll, "\n")

ridge_opt <- function(lambda, X, y) {
  # lambda: Ridge penalty parameter
  # X: Predictor matrix (scaled, with intercept if needed)
  # y: Target vector
  
  # Objective function: Penalized negative log-likelihood
  objective_function <- function(params) {
    # Extract theta and sigma from params
    p <- ncol(X)
    theta <- params[1:p]       # First p elements are theta
    sigma <- params[p + 1]     # Last element is sigma
    
    # Penalized negative log-likelihood
    if (sigma <= 0) return(Inf)  # Ensure sigma is positive
    ridge_ll <- ridge_log_likelihood(theta, sigma, X, y, lambda)
    return(ridge_ll)
  }
  
  # Initialize parameters (theta as 0, sigma as 1)
  init_params <- c(rep(0, ncol(X)), 1)  # Initialize theta and sigma
  
  # Optimize using BFGS
  result <- optim(
    par = init_params,
    fn = objective_function,
    method = "BFGS",
    control = list(maxit = 1000)  # Increase max iterations if needed
  )
  
  # Extract optimized values
  optimized_theta <- result$par[1:ncol(X)]
  optimized_sigma <- result$par[ncol(X) + 1]
  
  return(list(theta = optimized_theta, sigma = optimized_sigma, value = result$value, convergence = result$convergence))
}

# Prepare the data
X <- as.matrix(cbind(1, scaled_train[, predictor_columns]))  # Add intercept
y <- scaled_train$motor_UPDRS

# Set a lambda value
lambda <- 1


# Run RidgeOpt
ridge_result <- ridge_opt(lambda, X, y)

# Print results
cat("Optimized coefficients (theta):\n", ridge_result$theta, "\n")
cat("Optimized sigma:", ridge_result$sigma, "\n")
cat("Objective function value:", ridge_result$value, "\n")
cat("Convergence status (0=success):", ridge_result$convergence, "\n")

ridge_df <- function(lambda, X) {
  # lambda: Ridge penalty parameter
  # X: Predictor matrix (scaled, with intercept if needed)
  
  # Identity matrix of the size of the number of predictors
  I <- diag(ncol(X))
  
  # Compute the hat matrix for Ridge regression
  H <- X %*% solve(t(X) %*% X + lambda * I) %*% t(X)
  
  # Compute the degrees of freedom as the trace of the hat matrix
  df <- sum(diag(H))
  
  return(df)
}

# Prepare the data
X <- as.matrix(cbind(1, scaled_train[, predictor_columns]))  # Add intercept if needed

# Test for a specific lambda
lambda <- 10
df <- ridge_df(lambda, X)
cat("Degrees of Freedom for lambda =", lambda, ":", df, "\n")

# Explore a range of lambda values
lambdas <- seq(0.1, 100, length.out = 10)
df_values <- sapply(lambdas, function(l) ridge_df(l, X))

# Plot DF vs Lambda
plot(lambdas, df_values, type = "b", main = "Degrees of Freedom vs Lambda",
     xlab = "Lambda", ylab = "Degrees of Freedom")


