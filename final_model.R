#Let us load the required libraries
library(tidyverse)
library(caret)

#Let us set the seed for reproducibility and acquire the data.
set.seed(0) #Ensures reproducibility
link <- "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
banknotes <- read_csv(link,col_names=c('x1','x2','x3','x4','y'))

#Let us convert the data into a format amenable for prediction in caret::train
banknotes_factor <- banknotes %>% mutate(y = as.factor(y))

#Let us set the final parameters that worked well.
params_final <- trainControl(method = 'boot',
                         number = 1000)

#Let us run the basic model in order to choose the best parameters dynamically.
svm_model_final_select <- train(y~.,data=banknotes_factor,method='svmRadial',
                     preProcess = c('center','scale'),
                     trainParams = params_final,
                     tuneLength = 10)

#Let us print the results so that we can confirm them
svm_model_final_select$results

#Let us allocate the final parameters accordingly
first_C_index <- which.max(svm_model_final_select$results$Accuracy)
svm_model_final_select$results[first_C_index+(0:1),] 
best_values <- svm_model_final_select$results[first_C_index+1,1:2]
rownames(best_values) <- NULL

#Let us confirm these values
best_values

#Let us train the final model
svm_model_final <- train(y~.,data=banknotes_factor,method='svmRadial',
                         preProcess = c('center','scale'),
                         trainParams = params_final,
                         tuneGrid = best_values)

#Let us confirm its results
svm_model_final$results

#Let us obtain a final yhat and confirm this with a confusion matrix 
final_yhat <- predict(svm_model_final,banknotes_factor)
final_cmat <- confusionMatrix(final_yhat,banknotes_factor$y,
                              positive='1')
final_cmat