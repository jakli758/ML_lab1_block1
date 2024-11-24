data <- read.csv("C:/Users/victo/OneDrive/Bureau/A1_SML/Machine Learning/Labs/Lab1/data/pima-indians-diabetes.csv", header=FALSE)

colnames(data) <- c("n_preg", "plasma", "pressure","thickness","insulin","body_mass","diab_fun","age","diabetes")

# 1. Scatter plot

gradient <- colorRampPalette(c("blue", "red"))
colors <- gradient(100)  

# Normalize diab_fun to match the color indices
scaled_diab_fun <- as.numeric(cut(data$diab_fun, breaks = 100))

# Plot with the color gradient
plot(data$plasma, data$age, main="Plasma glucose concentration on age",
     xlab="Plasma", ylab="Age", pch=16, col=colors)
legend("topright", 
       legend = c("Low diabetes level", "High diabetes level"),
       col = c("blue", "red"), 
       pch = 16, 
       title = "Diabetes Level")
#Is diabetes easy to classify with these two variables? 
#it is not easy to classify diabetes using only plasma glucose concentration and age. 
#While the scatter plot shows some general trends, such as higher plasma glucose concentrations being associated with diabetes (red points), the overlap between the two classes (red and blue points) across the age and plasma glucose dimensions suggests that a simple logistic regression model might struggle to achieve high accuracy. 
#This is because the decision boundary between the two groups would not clearly separate the classes, leading to significant misclassification errors. Additional features or transformations might be necessary to improve classification performance.

#2 train logic regression model

data$diabetes <- as.factor(data$diabetes)  

# Train logistic regression model
logistic_model <- glm(diabetes ~ plasma + age, data = data, family = binomial)

# Make predictions (probabilities)
predicted_prob <- predict(logistic_model, type = "response")

# Classify based on threshold r = 0.5
predicted_class <- ifelse(predicted_prob >= 0.5, 1, 0)

# Add predictions to the data for inspection
data$predicted_prob <- predicted_prob
data$predicted_class <- predicted_class

# Print the first few rows of the predictions
head(data[, c("plasma", "age", "diabetes", "predicted_prob", "predicted_class")])


# Report the probabilistic equation of the estimated model
model_summary <- summary(logistic_model)
intercept <- coef(model_summary)[1]
plasma_coef <- coef(model_summary)[2]
age_coef <- coef(model_summary)[3]

prob_eq <- paste0("Probability(diabetes=1) = 1 / (1 + exp(-(", round(intercept, 4), 
                  " + ", round(plasma_coef, 4), " * plasma + ", 
                  round(age_coef, 4), " * age)))")
cat("Probabilistic Equation:\n", prob_eq, "\n\n")

# Compute training misclassification error
misclassification_error <- mean(data$diabetes != data$predicted_class)
cat("Training Misclassification Error:", round(misclassification_error, 4), "\n\n")

# Scatter plot with predicted values as colors
plot(data$plasma, data$age, 
     col = ifelse(data$predicted_class == 1, "red", "blue"), 
     pch = 16, 
     main = "Plasma glucose concentration on Age (Predicted Diabetes Classes)",
     xlab = "Plasma glucose concentration", 
     ylab = "Age")
legend("topright", legend = c("No Diabetes (0)", "Diabetes (1)"), 
       col = c("blue", "red"), pch = 16)

# The logistic regression model provides reasonable separation between the 
# two classes (diabetes and no diabetes) using plasma glucose concentration 
# and age. The scatter plot shows that higher glucose levels are strongly 
# associated with diabetes (red points), while lower levels are associated 
# with no diabetes (blue points). 
# 
# However, there is notable overlap between the classes in regions with 
# intermediate glucose levels and across different age groups, which may 
# contribute to the observed misclassification error of X%.
# 
# This suggests that while the model captures general trends, it struggles 
# to classify borderline cases accurately. Incorporating additional predictors 
# or exploring non-linear models may improve classification performance.


# Decision boundary equation: plasma as a function of age
# 0 = intercept + plasma_coef * plasma + age_coef * age
# plasma = -(intercept + age_coef * age) / plasma_coef
decision_boundary <- function(age) {
  -(intercept + age_coef * age) / plasma_coef
}


# Add decision boundary
age_range <- seq(min(data$age), max(data$age), length.out = 100)
lines(decision_boundary(age_range), age_range, col = "black", lwd = 2, lty = 2)

# Output decision boundary equation
cat("Decision Boundary Equation:\n")
cat("Plasma =", round(-intercept / plasma_coef, 4), "+", 
    round(-age_coef / plasma_coef, 4), "* Age\n\n")

# The decision boundary derived from the logistic regression model effectively 
# separates the two classes (diabetes and no diabetes) in the provided dataset. 
# As shown in the scatter plot, the majority of data points on each side of the 
# boundary correspond to the correct class. There is minimal overlap, particularly 
# in regions with intermediate plasma glucose levels, which indicates that the 
# linear decision boundary is well-suited for this dataset.
#
# While the model captures the general trend of the data, slight misclassifications 
# may still occur for borderline cases. Overall, the linear model provides a good 
# fit for this classification task.

#Part 4
#With r=0.2 and r=0.8

# r=0.2
predicted_prob_02 <- predict(logistic_model, type = "response")


predicted_class_02 <- ifelse(predicted_prob_02 >= 0.2, 1, 0)

data$predicted_prob_02 <- predicted_prob_02
data$predicted_class_02 <- predicted_class_02

head(data[, c("plasma", "age", "diabetes", "predicted_prob_02", "predicted_class_02")])

plot(data$plasma, data$age, 
     col = ifelse(data$predicted_class_02 == 1, "red", "blue"), 
     pch = 16, 
     main = "Plasma glucose concentration on Age with Decision Boundary",
     xlab = "Plasma glucose concentration", 
     ylab = "Age")
legend("topright", legend = c("No Diabetes (0)", "Diabetes (1)"), 
       col = c("blue", "red"), pch = 16)

# r=0.8
predicted_prob_08 <- predict(logistic_model, type = "response")

# Classify based on threshold r = 0.8
predicted_class_08 <- ifelse(predicted_prob_08 >= 0.8, 1, 0)

# Add predictions to the data for inspection
data$predicted_prob_08 <- predicted_prob_08
data$predicted_class_08 <- predicted_class_08

# Print the first few rows of the predictions
head(data[, c("plasma", "age", "diabetes", "predicted_prob_08", "predicted_class_08")])

# Scatter plot with predicted classes
plot(data$plasma, data$age, 
     col = ifelse(data$predicted_class_08 == 1, "red", "blue"), 
     pch = 16, 
     main = "Plasma glucose concentration on Age with Decision Boundary",
     xlab = "Plasma glucose concentration", 
     ylab = "Age")
legend("topright", legend = c("No Diabetes (0)", "Diabetes (1)"), 
       col = c("blue", "red"), pch = 16)


#With a threshold of r=0.2, the model predicts more cases as diabetes because the threshold for classification is lower. 
#This increases the number of false positives, blue points incorrectly classified as red
#The model becomes more sensitive, capturing more potential diabetes cases but at the expense of increased misclassification of non-diabetic individuals.
#Lowering the threshold reduces specificity but increases sensitivity.

#With a threshold of r=0.8, the model predicts fewer cases as diabetes because the threshold for classification is higher. This results in fewer false positives but increases the likelihood of false negatives (red points incorrectly classified as blue).
#The model becomes more specific, capturing fewer diabetes cases but at the expense of potentially missing some true positives (diabetes cases).
#Raising the threshold increases specificity but reduces sensitivity.


#5

# Create new features based on basis expansion
data$z1 <- data$plasma^4
data$z2 <- data$plasma^3 * data$age
data$z3 <- data$plasma^2 * data$age^2
data$z4 <- data$plasma * data$age^3
data$z5 <- data$age^4

# Train logistic regression model with expanded features
expanded_model <- glm(diabetes ~ plasma + age + z1 + z2 + z3 + z4 + z5, data = data, family = binomial)

# Predict probabilities and classify
data$expanded_predicted_prob <- predict(expanded_model, type = "response")
data$expanded_predicted_class <- ifelse(data$expanded_predicted_prob >= 0.5, 1, 0)

# Compute training misclassification error
expanded_misclassification_error <- mean(data$diabetes != data$expanded_predicted_class)
expanded_misclassification_error

plot(data$plasma, data$age, 
     col = ifelse(data$expanded_predicted_class == 1, "red", "blue"), 
     pch = 16, 
     main = "Plasma glucose concentration on Age (Expanded Model Classes)",
     xlab = "Plasma glucose concentration", 
     ylab = "Age")
legend("topright", legend = c("No Diabetes (0)", "Diabetes (1)"), 
       col = c("blue", "red"), pch = 16)



