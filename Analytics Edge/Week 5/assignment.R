# Unit 5 - Twitter


# VIDEO 5

# Read in the data

tweets = read.csv("tweets.csv", stringsAsFactors=FALSE)

str(tweets)


# Create dependent variable

tweets$Negative = as.factor(tweets$Avg <= -1)

table(tweets$Negative)


# Install new packages

install.packages("tm")
library(tm)
install.packages("SnowballC")
library(SnowballC)


# Create corpus
 
corpus = Corpus(VectorSource(tweets$Tweet))

# Look at corpus
corpus

corpus[[1]]


# Convert to lower-case

corpus = tm_map(corpus, tolower)

#to lower is a generic function that converts the corpus in the lower case letter. the tm_map() function applies
#works as apply() function i pass the vector and 

corpus[[1]]

# IMPORTANT NOTE: If you are using the latest version of the tm package, you will need to run the following line before continuing (it converts corpus to a Plain Text Document). This is a recent change having to do with the tolower function that occurred after this video was recorded.

corpus = tm_map(corpus, PlainTextDocument)


# Remove punctuation

corpus = tm_map(corpus, removePunctuation)

corpus[[1]]

# Look at stop words 
stopwords("english")[1:10]

#list of stop words for the english language.

# Remove stopwords and apple

corpus = tm_map(corpus, removeWords, c("apple", stopwords("english")))

corpus[[1]]

# Stem document 

corpus = tm_map(corpus, stemDocument)

corpus[[1]]




# Video 6

# Create matrix

frequencies = DocumentTermMatrix(corpus)

#the doc term matrix creates a word docuemtn matrix from the text provided. 
#each row of the frequencies matrix is a tweet and each column is one of the stemmed word in the tweet.

frequencies

# Look at matrix 

inspect(frequencies[1000:1005,505:515])

# Check for sparsity

findFreqTerms(frequencies, lowfreq=20)

#finds the frequent term using findFreqTerms() this finds the frequent terms in the frequencies matrix.

#really sparse matrix is bad so there will be a lot of terms that wont be useful in the prediction, so i need to
#remove the sparse terms from the matrix. 

# Remove sparse terms

sparse = removeSparseTerms(frequencies, 0.995)

#0.95 arg in the removeSparseTerms means that only keep words that occur in 5% or more tweets.0.98  2% or more

sparse

# Convert to a data frame

tweetsSparse = as.data.frame(as.matrix(sparse))  #convert the sparse matrix in a data frame

# Make all variable names R-friendly

colnames(tweetsSparse) = make.names(colnames(tweetsSparse))  #add colnames to the sparse matrix

# Add dependent variable

tweetsSparse$Negative = tweets$Negative

# Split the data

library(caTools)

set.seed(123)

split = sample.split(tweetsSparse$Negative, SplitRatio = 0.7)

trainSparse = subset(tweetsSparse, split==TRUE)
testSparse = subset(tweetsSparse, split==FALSE)



#once i have the data in the relevant data frame then i can use the general ML models for processing the data
#i can use the tree model and the logit regression model to build a text classification system

#also text class is no different from other machine learning tasks.


# Video 7

# Build a CART model

library(rpart)
library(rpart.plot)

tweetCART = rpart(Negative ~ ., data=trainSparse, method="class")

prp(tweetCART)

# Evaluate the performance of the model
predictCART = predict(tweetCART, newdata=testSparse, type="class")

table(testSparse$Negative, predictCART)

# Compute accuracy

(294+18)/(294+6+37+18)

# Baseline accuracy 

table(testSparse$Negative)

300/(300+55)

#now i am using the random forest model


# Random forest model

library(randomForest)
set.seed(123)

tweetRF = randomForest(Negative ~ ., data=trainSparse)

# Make predictions:
predictRF = predict(tweetRF, newdata=testSparse)

table(testSparse$Negative, predictRF)

# Accuracy:
(293+21)/(293+7+34+21)

