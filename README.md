MACHINE LEARNING HUMAN ART
Old Art Style Images — EDA & Artist Classification
A machine learning project built around a dataset of old art style illustrations paired with text captions and descriptions. The goal was to explore the data, understand its structure, and build a model capable of predicting the artist behind each illustration based on text features.
Why this project?
I came across this dataset on Kaggle and thought it was a genuinely interesting challenge. Most image classification projects use modern photography or clean benchmark datasets — this one is messier and more human. The captions are written in natural language, the artist names are inconsistent, and a big chunk of the data is labeled "Unknown". That made it a good exercise in data cleaning and in figuring out what you can actually learn from imperfect data.

You can find the dataset here: https://www.kaggle.com/datasets/adarsh2626/old-art-style-images-with-captions-dataset

What I did
1. Data cleaning
The dataset had a few issues that needed sorting out before doing anything useful:

artist_name had entries written as "Unknown" (not null, just the string)
description had 104 empty rows
Some artists appeared only once or twice — not enough to train on

After cleaning I filtered down to artists with at least 10 works, which gave around 2049 images. Then I narrowed it further to the top 15 artists to keep class balance reasonable.
2. ML Pipeline
The model takes the combined caption + description text for each image and tries to predict the artist. Three classifiers were tested inside a scikit-learn pipeline:
ModelNotesLogistic RegressionStrong baseline, interpretableLinear SVMBest performer on text tasksNaive BayesFast, solid for sparse TF-IDF matrices
Feature extraction used TF-IDF with unigrams and bigrams (500 features, sublinear TF scaling).
Results were evaluated with 5-fold cross-validation and a held-out test set. The best model was selected by test accuracy and a full classification report and confusion matrix were generated.
What could be improved
This project is intentionally kept simple — scikit-learn only, no deep learning. That said, there's a lot of room to push further:

Image features: Right now the model only uses text. Adding CNN-based image embeddings (ResNet, EfficientNet) would likely improve accuracy significantly since each artist has a recognizable visual style
Better text embeddings: Replacing TF-IDF with Sentence-BERT or similar would capture semantic meaning rather than just word frequency
Data augmentation: With only 10-50 images per artist, augmentation techniques could help
More data cleaning: Some "Unknown" entries might be identifiable through the URL column
