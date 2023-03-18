# Topic Modeling

This code implements topic modeling on a set of text files in a folder using natural language processing techniques. The code loads the text files into a pandas dataframe, preprocesses the text by expanding abbreviations, expanding contractions, removing apostrophes, removing links, removing extra symbols, and removing accented characters. It then splits the text into sentences and performs topic modeling using a count vectorizer and K-means clustering. The resulting clusters are plotted, and their top keywords are displayed.

The code also loads text data from three different sources and performs several NLP tasks on them, such as sentence segmentation, word tokenization, and n-gram extraction (bigrams and trigrams). It generates word clouds to visualize the most frequently occurring bigrams and trigrams.

Moreover, it performs vectorization on the text data using Bag of Words (BOW) and Term Frequency-Inverse Document Frequency (TF-IDF) techniques. The code performs clustering using the K-means algorithm to group similar sentences into clusters. The optimal number of clusters is found by using the elbow method on the Sum of Squared Errors (SSE) metric.

One-hot encoding: The code uses the OneHotEncoder from scikit-learn to perform one-hot encoding on three different dataframes: mllpi_sent_df, trans_sent_df, and hth_sent_df. One-hot encoding is a technique for representing categorical data as binary vectors.

Word embedding: The code uses the Word2Vec model from gensim to perform word embedding on three different lists of text: mllpi_lst, hth_lst, and trans_lst. Word embedding is a technique for representing words as vectors in a high-dimensional space, where semantically similar words are close to each other.

Summarization: The code uses three different techniques for text summarization: Latent Semantic Analysis (LSA), Non-Negative Matrix Factorization (NMF), and Latent Dirichlet Allocation (LDA). These techniques are used to identify the most important topics in a text corpus and summarize them into a smaller number of topics.

Overall, the code appears to be performing some basic preprocessing and analysis of text data using a variety of techniques.

##Mallet Implementation of LDA
This code block is using the Mallet implementation of LDA for topic modeling. It starts by defining the path to the Mallet executable file. Then, a function named compute_coherence_values is defined, which takes in the dictionary, corpus, texts, limit, start, and step as input parameters. It iterates through the range of num_topics starting from start to limit with a step size of step. For each num_topics, it computes an LDA model using the Mallet implementation and saves the model and its coherence score. Finally, it returns two lists, model_list and coherence_values.

After defining the compute_coherence_values function, it is used to calculate the coherence scores for different numbers of topics. Then, the coherence scores are plotted against the number of topics, and the number of topics with the highest coherence scores are printed.

The next part of the code defines a function named format_topics_sentences. This function takes in the ldamodel, corpus, and texts as input parameters. It iterates through each document in the corpus and extracts the dominant topic, percentage contribution, and keywords for each document. It then returns a Pandas DataFrame containing the dominant topic, percentage contribution, keywords, and original text for each document.

After defining the format_topics_sentences function, it is used to create a DataFrame named df_topic_sents_keywords, which contains the dominant topic, percentage contribution, keywords, and original text for each document.

To run the code, first ensure that all necessary libraries are installed, including pandas, numpy, nltk, gensim, sklearn, matplotlib, and seaborn. Then, clone the repository containing the code and navigate to the appropriate directory.

Next, make sure the text files you wish to analyze are stored in a folder and update the path in the code accordingly. You may also want to adjust the parameters for the various NLP techniques used in the code, such as the number of clusters for K-means clustering and the number of topics for LDA.

Once the code is executed, it will output several plots and dataframes summarizing the results of the analysis. These can be used to gain insights into the key topics and themes present in the text data.

It is worth noting that this code is just one example of how to perform topic modeling and other NLP techniques in Python. There are many other libraries and approaches available, and the best approach may depend on the specific requirements of your project. However, this code provides a good starting point for those looking to get started with text analysis in Python.
