# Sentiment Analysis of Movie Reviews

This project aims to perform sentiment analysis on movie reviews using Python, NumPy, pandas, scikit-learn, and the IMDb dataset from Kaggle. The IMDb dataset contains labeled movie reviews indicating whether each review is positive or negative.

## Installation

Before running the project, make sure you have Python installed on your system. You can install the required libraries using pip:

```bash
pip install numpy pandas scikit-learn
```

You'll also need to download the IMDb dataset from Kaggle and place it in the project directory.

## Usage

1. **Data Preprocessing**: Load the IMDb dataset using pandas and preprocess the text data (e.g., tokenization, removing stopwords, etc.).

2. **Feature Extraction**: Convert the text data into numerical features using techniques like bag-of-words or TF-IDF.

3. **Model Training**: Split the dataset into training and testing sets. Train a sentiment analysis model using scikit-learn classifiers like Logistic Regression, Naive Bayes, or Support Vector Machine.

4. **Evaluation**: Evaluate the trained model's performance on the testing set using metrics like accuracy, precision, recall, and F1-score.

5. **Inference**: Use the trained model to predict the sentiment of new movie reviews.

## Project Structure

- `README.md`: Instructions and information about the project.
- `imdb_dataset.csv`: IMDb dataset containing movie reviews and their corresponding sentiment labels.
- `sentiment_analysis.py`: Python script containing the code for preprocessing, feature extraction, model training, evaluation, and inference.

## Running the Script

To run the sentiment analysis script:

```bash
python sentiment_analysis.py
```

Make sure you have the necessary permissions to read the IMDb dataset file and that it is located in the same directory as the script.

## Acknowledgments

- IMDb for providing the dataset.
- Kaggle for hosting the IMDb dataset.
- Developers of NumPy, pandas, and scikit-learn for their contributions to the Python ecosystem.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
