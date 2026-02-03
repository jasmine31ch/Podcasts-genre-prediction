# Podcasts-genre-prediction
## Description
The application predicts a podcast episode’s genre using short audio snippets. It uses models to analyze both the acoustic features of short audio snippets and the text obtained from ASR transcription. This addresses the challenge of manually labeling podcast genres by providing a faster, more consistent, and more scalable automated solution.
## User and Benefits
The intended users of this system include podcast listeners as well as podcast platforms that curate and recommend content. Individual listeners who are browsing for new shows can benefit from a genre prediction model because it helps them explore unfamiliar podcasts that match their interests even when they have limited prior knowledge about those creators or channels. Podcast platforms and developers can also use this model as part of a recommendation pipeline, as many podcast episodes lack consistent or accurate metadata. This application could assist with searching recommendations, maintain more accurate metadata for the platforms, and help new creators become discoverable.
## Dataset
https://github.com/jasmine31ch/Podcasts-genre-prediction
## Platform
Google Colab - A100
## ML Aprroach
### 1. Data Collection:
We constructed our dataset using Spotify’s genre-specific Top 30 rankings. For each of the six target genres, we selected five channels by prioritizing shows labeled exclusively with that genre, and when necessary, those listing the genre as their primary category. From these, three channels per genre were assigned to the training set, while one channel each was reserved for validation and testing. 
For the training dataset (3 shows), each show has 6 episodes. (180 snippets)
For the test/validation dataset (1 show each dataset), each has 4 episodes. (40 snippets)
Each snippet has 40s. The snippet is selected randomly from the episodes and they don’t overlap with each other.
### 2. Data sheet:
![image](https://github.com/jasmine31ch/Podcasts-genre-prediction/blob/master/img/Data%20sheet.png)

### 3. Data Processing:
Collect podcast using RSS: We collected podcast audio from open-source sources by retrieving each show’s RSS feed. After identifying podcast names on ListenNotes (https://www.listennotes.com/#), we used the platform to locate their RSS URLs. We then wrote scripts to automatically parse the RSS feeds, extract audio file links, and download the corresponding episodes for dataset construction.
Divide data into training, test and validation dataset: Train : test : val = 9 : 2 : 2 (copied to data/split)
Divide data into random, non-overlap snippets and reorder: stored in data/snippets
ASR: Translate the audio snippets to text and store training, test and validation dataset into csv files separately.

### 4. Evaluation:
For the hyperparameters of models, we use validation dataset to find the better hyperparameters. For the test outcomes, we evaluate our approach using classification accuracy as the primary metric, reporting overall accuracy on the validation and test sets. In addition, we compute a classification report (precision, recall, F1-score for each genre) and analyze a confusion matrix to understand which genres are most frequently misclassified.

## Model details
### 1. Baseline Model
### 1a. Description: 
We implemented a classical machine learning baseline using TF-IDF bag-of-words features and a multinomial Logistic Regression classifier. This baseline provides a reference point for understanding task difficulty and measuring improvements from deep learning models.
    
### 1b. Preprocessing: 
We first encoded the genre labels using LabelEncoder to obtain integer class IDs. For text, we applied a standard TF-IDF pipeline implemented with scikit-learn’s TfidfVectorizer. It included (1) Filled any missing transcripts (NaN) with empty strings and cast them to str. (2) Lowercase all text. (3) Used unigrams and bigrams. (4) Limited the vocabulary to max_features=10000.
    
### 1c. Hyperparameters: 
### TF-IDF Vectorizer:

top 10,000 most informative terms (max_features=10,000)

both unigrams and bigrams to capture context

terms must appear in at least 3 documents to reduce noise (min_df=3)

apply sublinear term frequency scaling (sublinear_tf=True)

normalize to lowercase (lowercase=True)

### Logistic Regression:

L2 regularization

max_iter = 300 to ensure convergence on the high-dimensional TF-IDF features

class_weight=None and n_jobs=-1 to use all available CPU cores

We treated this as a standard but solid baseline model and did not run an extensive hyperparameter sweep for Logistic Regression, since its primary role is to provide a simple reference rather than be the strongest possible classifier.

### 2. Text Model
### 2a. Description: 
We fine-tuned a pretrained DistilBERT-base-uncased model for podcast genre classification using only podcast transcripts. DistilBERT is a smaller, faster variant of BERT that retains ~97% of BERT's language understanding while being 40% smaller and 60% faster. Crucially, we implemented automated hyperparameter optimization using Optuna to find the best training configuration.
### 2b. Preprocessing: 
Converted transcripts to strings for tokenization; Used DistilBERT's pretrained tokenizer; Transformed genre labels into numerical indices with bidirectional mappings.
### 2c. The base TrainingArguments provide default values that Optuna then overrides per trial. Our search space is:
learning_rate: log-uniformly in [1e-6, 5e-5]

num_train_epochs: {3, 4, 5}

per_device_train_batch_size: categorical in {8, 16}

weight_decay: continuous in [0.0, 0.1]

We run 20 trials, each trial:

Instantiates a fresh DistilBERT classification model via model_init

Fine-tunes it on the training set

Evaluates on the validation set using our compute_metrics function (accuracy and macro F1)

The best trial (stored in best_trial) provides the chosen learning_rate, num_train_epochs, per_device_train_batch_size, and weight_decay, which we then plug into a new TrainingArguments object for the final training run.

This text-only DistilBERT model serves as our main neural text model and is used as a key comparison point alongside the TF-IDF baseline and the multimodal Audio & Text model.

### 3. Audio & Text Model
### 3a. Description: 
We use a multimodal architecture that combines audio and text features. For audio, we use a pretrained Wav2Vec2-base (facebook/wav2vec2-base-960h) encoder, and for text we use a pretrained DistilBERT-base-uncased encoder.

### 3b. For the pretrained model: 
We experimented with unfreezing different numbers of layers in both Wav2Vec2 and DistilBERT. Models performed better when more layers were unfrozen, indicating that deeper fine-tuning helps the model adapt to podcast-specific acoustic and linguistic patterns.

### 3c. Hyperparameters: 
We trained the model using AdamW with a learning rate of 2e-5, and applied a smaller LR (0.1×) to pretrained encoder layers. We used a batch size of 8, CosineAnnealingLR for scheduling, and early stopping for stability. Audio snippets were fixed at 5 seconds to only get the audio features, text sequences were capped at 256 tokens, and the classifier included dropout (0.5) to reduce overfitting.

### 3d. For the model: 
We also experimented with larger audio and text encoders, but due to the limited dataset size and GPU constraints, they did not yield noticeably better performance.



