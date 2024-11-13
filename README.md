# News-Category-Prediction


## 1. Introduction  
In today’s fast-paced digital era, categorizing and analyzing news articles is crucial to managing the massive flow of information. This project tackles these challenges by implementing a **News Category Prediction Model**, complemented by a **Sentiment Analysis Module** and a **News Recommendation System** to enhance the user experience. Using the MIND (Microsoft News Dataset), this project aims to create models capable of classifying news articles, assessing their sentiment, and delivering personalized recommendations.

## 2. Data Source  
Dataset: [Microsoft News Dataset (MIND)](https://msnews.github.io/)  
The MIND dataset, provided by Microsoft Research, includes titles, abstracts, and categories for thousands of news articles across 16 categories, such as **sports**, **finance**, **health**, **lifestyle**, and more. The training set contains 81,222 articles, and the test set includes 20,305 articles.

## 3. Data Cleaning  
Our analysis focuses on news titles and abstracts to improve classification accuracy. To prepare the data, we applied normalization techniques, including converting to lowercase, removing punctuation, eliminating stopwords, and filtering out single letters and numbers.

## 4. Methods  
### 1) Bag of Words & TF-IDF  
Using **Term Frequency-Inverse Document Frequency (TF-IDF)**, we built a feature matrix to capture meaningful distinctions in textual data. Scikit-learn’s `CountVectorizer()` and `TfidfVectorizer()` helped to convert text data into vectors, optimizing it for machine learning models.

### 2) Random Forest  
We used the **RandomForestClassifier** from `sklearn.ensemble` to build an ensemble model that combines multiple decision trees, each trained on a random subset of data, to enhance the model's predictive power through a voting mechanism.

### 3) Logistic Regression  
Our **Logistic Regression** model initially demonstrated modest accuracy. After cross-validation using `cross_val_predict`, the model’s performance improved significantly, proving its capability to identify patterns in news categorization.

### 4) Neural Networks  
Using **Keras**, we implemented a neural network with a dense layer followed by a Softmax layer to generate probabilities for each category. This architecture provided insights into confidence levels for each class in our multi-class classification problem.

### 5) Support Vector Machines  
Through hyperparameter tuning, our **Support Vector Machine (SVM)** model demonstrated significant improvements in accuracy by handling complex decision boundaries effectively.

### 6) News Recommendation System  
Using a **content-based approach** and cosine similarity on TF-IDF vectors, our recommendation system suggests news articles based on similarity to user-viewed articles, enhancing relevance.

### 7) Sentiment Analysis  
Leveraging **VADER** sentiment intensity analyzer, we categorized articles as positive, neutral, or negative. This insight into public opinion deepens our understanding of news content sentiment.

## 5. Evaluation and Analysis  
- **Logistic Regression**: Achieved 73% accuracy after adding TF-IDF and cross-validation.
- **SVM**: Initial accuracy of 70%, improved to 72.46% after parameter tuning.
- **Random Forest**: Demonstrated 64% accuracy on this dataset.
- **Neural Network**: Reached 72.42% accuracy on the test set.

Some categories, such as 3, 11, and 13, showed high precision and recall, while others (e.g., 6, 12) had lower scores, likely due to limited data representation.

## 6. Related Work  
Previous studies on news categorization used methods like Naive Bayes, Logistic Regression, and SVM with varying success. Deep learning approaches, such as CNNs and RNNs, were also tested to capture both spatial and temporal patterns in text data. This project draws on these techniques to improve predictive accuracy for news categorization.

## 7. Discussion and Conclusion  
Our results highlight that **SVM, Neural Networks, and Logistic Regression** provided the best accuracy, around 73%, on the MIND dataset. The dataset’s skewed distribution towards certain categories, such as sports, posed challenges for balanced categorization. Future work could involve refining algorithms to handle larger datasets, incorporating subcategories, and expanding beyond English news sources. Using advanced models like transformers may further enhance model interpretability and accuracy, especially with balanced category distribution.

## 8. Future Work  
- **Enhance Data Balance**: Obtain a more balanced dataset across categories for fair representation.
- **Explore Transformer Models**: Investigate transformer-based models for improved performance.
- **Expand Language Support**: Add support for multilingual news categorization.
- **Implement Real-Time Recommendation**: Develop capabilities for streaming data for real-time recommendations.
- **Incorporate User Feedback**: Tailor the model to adapt to individual user preferences for better personalization.

## 9. References  
1. [Kaur, S., et al., 2016](https://www.caeaccess.org/archives/volume5/number1/kaur-2016-cae-652224.pdf)  
2. [Stanford CS229 Project Report, 2018](https://cs229.stanford.edu/proj2018/report/183.pdf)  
3. [IOSR Journal of Computer Engineering](https://www.iosrjournals.org/iosr-jce/papers/Vol18-issue1/Version-3/D018132226.pdf)  
4. [IEEE Xplore Article](https://ieeexplore.ieee.org/document/9725409)  
5. [Keras Documentation](https://keras.io/)
