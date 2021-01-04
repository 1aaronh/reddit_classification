# Reddit Content Classification Model & Comparison of Vectorization Methods
---
# Problem Statement:
---
Will a model improve its predictions of text data based on whether it is fitted on simple Count Vetcors or Term Frequency-Inverse Document Frequency (TF-IDF) vectors?
-------------
# Data & Background:
---
The text data for analysis comes from the [gaming](https://www.reddit.com/r/gaming/) and [movies](https://www.reddit.com/r/movies/) subreddits. The data was retrieved using [PRAW API](https://praw.readthedocs.io/en/latest/). The two subreddits were chosen because they were intrepreted to be distinct but with enough overlap to challenge a classification model as they are both categories in entertainment.

TF-IDF's identification of the relative importance of a word (term frequency) between different documents will be considered against the measurement of simpler word frequencies that are measured with Scikit-learn's Countvectorizer.

# Methodology & Preprocessing:
---
Separate PRAW objects were created for each subreddit respectively. These objects were parsed into dictionaries then Pandas DataFrames. It was determined that both the title and body of a document would be relevant for disinguishing a subreddit category and these features were combined. The features 'title' and 'body' in each DataFrame were concatenated to engineer a single feature 'alltext'. The target feature 'subreddit' was engineered for each DataFrame. The target features were populated with the names 'gaming' and 'movies'. 

After feature engineering, the two dataframes were concatenated vertically with Pandas. The data for the subreddit category was binarized as:
    - 'gaming':1
    - 'movies':0

# Vectorization & EDA:
---
Stop words were removed and the default token pattern was selected to remove punctuation when using both TF-IDF and CountVectorizer. The max_features count for EDA purposes was chosen at 2000. EDA was performed from the results of the TF-IDF vectors with different N-gram ranges. Below is an example of an N-gram range of 2:

![](https://github.com/1aaronh/reddit_classification/blob/master/images/tf_2gram.png)
Many of the vectors are generic and can be found in any subreddit. Titles of specific games or movies do stand out though.

# Modeling:
---
A basic Logistic Regression model was selected for each method of vectorizing. Both models performed well with the model trained on TF-IDF vectors slightly exceeding the one trained on simpler count vectors. Train Test Split was used in advance of modeling. Scikit-learn's Pipeline and GridSearch methods were used to find the optimal combinations of hyperparameters.

### Model Results & Metrics:
---
- CountVectorizer
The model fit with CountVectorizer delivered a test accuracy of 91.6%. Below are this models metrics:

![](https://github.com/1aaronh/reddit_classification/blob/master/images/cvec_confusionmatrix.png)
The model did very well at distinguishing between the subreddits.

![](https://github.com/1aaronh/reddit_classification/blob/master/images/cvec_roc.png)
The AUC score is almost perfect. The model likely benefitted from the fact that the classes were already reasonably balanced in the original data.

- TF-IDF
This model printed test accuracy of 93.5%. We can verify the slight outperformance by comparing the same metrics:

![](https://github.com/1aaronh/reddit_classification/blob/master/images/tfid_confusionmatrix.png)
Interesting observation that TF-IDF slightly underperforms the CountVectorizer model on the gaming category. Future tests would incorporate more data to see if this pattern holds.

![](https://github.com/1aaronh/reddit_classification/blob/master/images/tfid_roc.png)
Slight outperformance on ROC Curve when we consider overall classification between classes.

# Conclusions:
---
We have almost the same quality of model output regardless of the chosen method of vectorization. The model fitted with TF-IDF tends to perfomr slightly better but this should continue to be tested, perhaps with different subreddit categories altogether. The generally high testing accuracy scores for both models gives confidence that strong combinations of hyperameters were discovered by GridSaerch.

# Next Steps:
---
More nuanced selection of hyperparameters for the vectorizers can be done such as selecting different regular Expressions for token patterns. Doing this may discover more exact nuance among the words. 

The prinary next step would be additional API calls to gather more data from the subreddits. This can offer more robust confidence in the output of the models and the reliability of the metrics. Last, we can see if models fit in this fashion would perform as well when we use text data from entirely different subreddits.
