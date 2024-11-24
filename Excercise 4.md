**Question:** Why can it be important to consider various probability thresholds in the
classification problems, according to the book?

**Answer:** The book states that a classifcation threshold of 0.5 on average leads to the lowest missclassification error, 
however depending on the application scenario, this is not always the most important metric to optimize. In some cases, especially those were the target variable is heavily imbalanced
it can make sense to lower or increase the classification threshold. Especially in medical settings where a patients life can depend on a diagnosis being made, it is more desirable 
to have a high detection rate of the disease, even if we have to sacrifice the misclassification error to some extent. As the consequence of a patient's disease being undetected far outweighs
a potential false positive. (p. 50)

**Question:** What ways of collecting correct values of the target variable for the
supervised learning problems are mentioned in the book?

**Answer:** In some scenarios collecting data on the target variable is straight forward, as all that is required is to record the joint occurences of the predictive variables $x$ and the target variable $y$ 
occuring in the real world. For example measuring the height and weight of a person and then using the height as a predictor for a person's weight. However, in scenarios in which the target variable
is not as obvious, it might be required to have domain experts label the individual data points such as ECG scans that are indicative of cardiovascular disease, or images of skin irregularities that could possibly consitute 
skincancer. The process of manually labelling data, especially if it requires highly skilled domain experts, such as cardiologists or dermatologists, is incredibly time and resource intensive and makes the aggregation
of a large data set increasingly more difficult. (example of ECG scans mentioned on p. 4)

**Question:** How can one express the cost function of the linear regression in the matrix
form, according to the book? 

**Answer:** The forumula for the cost function can be found on page 41 figure 3.11. 


