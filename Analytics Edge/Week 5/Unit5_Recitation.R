# Unit 5 - Recitation


# Video 2

# Load the dataset

emails = read.csv("energy_bids.csv", stringsAsFactors=FALSE)

str(emails)

# Look at emails

emails$email[1]
emails$responsive[1]
 
emails$email[2]
emails$responsive[2]

# Responsive emails

table(emails$responsive)

#the responsive variable in the emails data frame is made in a similar manner as amazon mechanical turk

# Video 3


# Load tm package

library(tm)


# Create corpus

corpus = Corpus(VectorSource(emails$email))

corpus[[1]]


# Pre-process data
corpus = tm_map(corpus, tolower)

# IMPORTANT NOTE: If you are using the latest version of the tm package, you will need to run the following line before continuing (it converts corpus to a Plain Text Document). This is a recent change having to do with the tolower function that occurred after this video was recorded.
corpus = tm_map(corpus, PlainTextDocument)


corpus = tm_map(corpus, removePunctuation)

corpus = tm_map(corpus, removeWords, stopwords("english"))

corpus = tm_map(corpus, stemDocument)

# Look at first email
corpus[[1]]



# Video 4

# Create matrix

dtm = DocumentTermMatrix(corpus)
dtm     #99% sparse

#the dtm tells the no of columns and the number of rows in the model. also the dtm returns the sparsity of the
#data matrix.

#now sparsity leads to the model overfitting the data, so i need to remove the infrequent words and hence decrea
#the sparsity of the matrix.

# Remove sparse terms
dtm = removeSparseTerms(dtm, 0.97)
dtm    # after removing the non sparse term the matrix is 92% sparse

# Create data frame
labeledTerms = as.data.frame(as.matrix(dtm))

#this creates a data frame where each column is the word and the value in the column is the number of times the
#word appeared in the email

# Add in the outcome variable
labeledTerms$responsive = emails$responsive

str(labeledTerms)



# Video 5


# Split the data

library(caTools)

set.seed(144)

spl = sample.split(labeledTerms$responsive, 0.7)

train = subset(labeledTerms, spl == TRUE)
test = subset(labeledTerms, spl == FALSE)

# Build a CART model

library(rpart)
library(rpart.plot)

emailCART = rpart(responsive~., data=train, method="class")

prp(emailCART)



# Video 6

# Make predictions on the test set

pred = predict(emailCART, newdata=test)
pred[1:10,]
pred.prob = pred[,2]

# Compute accuracy

table(test$responsive, pred.prob >= 0.5)

(195+25)/(195+25+17+20)

# Baseline model accuracy

table(test$responsive)
215/(215+42)


#now in this application domain higher cost is involved for the false negative than to the false positive hence 
#i have to change the threshold. hence look at the roc curve


# Video 7

# ROC curve


# 1. create a prediction object, the syntax is same as the table Fn.     prediction(test$y,pred$y)
# 2. create a performance object from the pred objefct in the first step
# 3. plot(perfROCR,colorize=TRUE)    plot the perf object.

library(ROCR)

predROCR = prediction(pred.prob, test$responsive)

perfROCR = performance(predROCR, "tpr", "fpr")

plot(perfROCR, colorize=TRUE)

# Compute AUC

performance(predROCR, "auc")@y.values     #extract "auc" from the performance object and get @y.values as the auc



