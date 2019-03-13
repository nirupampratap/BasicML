library(datasets)

data(iris)

reqData <- c()
finalOutput <- c()

# Define all combinations
paramCombs = list('Sepal.Length','Sepal.Width','Petal.Length','Petal.Width',
         c('Sepal.Length','Sepal.Width'),c('Sepal.Length','Petal.Length'),
         c('Sepal.Length','Petal.Width'),c('Sepal.Width','Petal.Length'),
         c('Petal.Length','Petal.Width'),c('Sepal.Width','Petal.Width'),
         c('Sepal.Length','Sepal.Width','Petal.Length'),
         c('Sepal.Length','Sepal.Width','Petal.Width'),
         c('Sepal.Length','Petal.Length','Petal.Width'),
         c('Sepal.Width','Petal.Length','Petal.Width'),
         c('Sepal.Length','Sepal.Width','Petal.Length','Petal.Width'))

# Define some variables
maxAccuracy = 0
maxParamUsed = 0
KForMaxAcc = 0

# Loop over the combinations
for(selComb in paramCombs)
{
  
#>>> Section 1 - K Identification <<<
  
  # Select the data
  reqData = iris[, c(selComb)]
  
  # Define two dataframes that we shall use for collecting and processing output
  output <- c()
  diffOut <- c()
  
  #Loop over k from 2:12, run Kmeans and collect the output
  for(i in 2:12)
  {
    irisModel <- kmeans(reqData, centers=i, nstart=25)
    
    output <- rbind(output,c(i, irisModel$tot.withinss))
  }
  
  # Set the colnames for the output vector
  colnames(output) <- c("K", "TotalDistSq")
  
  # Initialize a few variables that we shall use
  selK = 0
  prevDist = -1
  selectionDone = 0
  pctAtElbow = 0.75
  
  for(i in 1:length(output[,"TotalDistSq"]))
  {
    if(prevDist == -1)
    {
      prevDist = as.numeric(output[i, "TotalDistSq"])
      diffOut[i] = 0
    }
    else
    {
      diffOut = rbind(diffOut, c(prevDist - as.numeric(output[i, "TotalDistSq"])))
      
      if(pctAtElbow * diffOut[i-1] < diffOut[i] && selectionDone == 0 && diffOut[i-1] != 0)
      {
        selK = as.numeric(output[i-1,"K"])
        selectionDone = 1
      }
      
      prevDist = as.numeric(output[i, "TotalDistSq"])
    }
  }
  
#>>> Section 2 - Accuracy Measurement <<<
  
  # Create a copy of the dataframe for processing down below
  irisCopy <- iris
  
  # reqData has already been set to the right columns. No not resetting it again
  
  # Create the model. set K to selected K
  irisModel <- kmeans(reqData, centers=selK, nstart=25)
  
  # Attach the fitted values to the dataframe copy
  irisCopy <- cbind(irisCopy, as.numeric(irisModel$cluster))
  
  #Change the colname of the fitted values
  colnames(irisCopy)[6] = "fittedValues"
  
  label1 = "setosa"
  label2 = "versicolor"
  label3 = "virginica"
  
  #Set the column for predicted type
  irisCopy = cbind(irisCopy, "dummyVal")
  colnames(irisCopy)[7] = "PredictedType"
  
  #Mention that the last column is of type character/string. 
  irisCopy$PredictedType = as.character(irisCopy$PredictedType)
  
  # Identify each of the groups and then decide which one is the correct label
  for(i in 1:selK)
  {
    # Initialize the counts
    l1Count = 0
    l2Count = 0
    l3Count = 0
    
    #Select the set to process. filter by the rows
    selSet = irisCopy[irisCopy$fittedValues == i, ]
    
    #For this set find the major type (i.e. from the given classification)  
    l1Count = nrow(selSet[selSet$Species == label1,])
    l2Count = nrow(selSet[selSet$Species == label2,])
    l3Count = nrow(selSet[selSet$Species == label3,])
    
    if(l1Count >= l2Count)
    {
      #Label1 is to be applied to this group
      if(l1Count >= l3Count)
      {
        irisCopy[irisCopy$fittedValues == i, 7] = label1
      }
      else
      {
        irisCopy[irisCopy$fittedValues == i, 7] = label3
      }
    }
    else
    {
      # L2 is to be applied to this group
      if(l2Count >= l3Count)
      {
        irisCopy[irisCopy$fittedValues == i, 7] = label2
      }
      else # L3 is to be applied to this group
      {
        irisCopy[irisCopy$fittedValues == i, 7] = label3
      }
    }
  }
  
  matchedVal = nrow(irisCopy[irisCopy$Species == irisCopy$PredictedType,])
  totalRows = length(irisCopy$Sepal.Length)
  
  accuracy = round(matchedVal/totalRows * 100,2)
  
  # Check which one has maximum accuracy
  if(maxAccuracy <= accuracy)
  {
    maxAccuracy = accuracy
    maxParamUsed = selComb
    KForMaxAcc = selK
  }
  
  # Collect the final output and ensure that you concatenate all selected columns into a single string
  finalOutput = rbind(finalOutput,c(paste(selComb,collapse="-"), selK, accuracy))
}

# Set the column names for final output
colnames(finalOutput) = c("Parameters", "Selected K", "Accuracy (%)")

#>>> Section 3 - Save the final output to a file

WorkingDir = "./"
setwd(WorkingDir)

# Save to a file
write.csv(finalOutput, "MSA2018_Grp10_HW2_Q4_2_output.csv")

# Print Max accuracy
sprintf("Max Accuracy Parameters: %s, K: %s, Accuracy: %s ", paste(maxParamUsed, collapse="-"),  KForMaxAcc, maxAccuracy)
