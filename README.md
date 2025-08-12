# BERT_Sentimental_analysis

# Amazon Review Sentiment Analysis with BERT

This project demonstrates how to perform sentiment analysis on Amazon reviews using a pre-trained multilingual BERT model from the Hugging Face Hub. The script downloads the "Amazon Reviews" dataset, predicts the sentiment (as a 1-5 star rating), and then evaluates the model's performance on a binary (positive/negative) classification task.

## Key Features

- **Pre-trained Model**: Utilizes the `nlptown/bert-base-multilingual-uncased-sentiment` model for sentiment prediction.
- **Kaggle Dataset**: Automatically downloads and decompresses the `bittlingmayer/amazonreviews` dataset using `kagglehub`.
- **Sentiment Prediction**: Predicts a 1 to 5-star rating for each review.
- **Model Evaluation**: Evaluates the model's accuracy on a binary (positive/negative) sentiment task, achieving **88.50% accuracy** on a sample of the test set.
- **Data Visualization**: Includes imports for `matplotlib` and `seaborn` for potential data visualization (e.g., confusion matrix).

## Project Structure

```
.
├── sentiment_analysis.ipynb      # The main Jupyter Notebook with all the code.
├── requirements.txt              # A list of Python packages required to run the project.
└── README.md                     # This file.
```

## Setup and Installation

Follow these steps to set up your local environment and run the project.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/rishu1947-ops.git
    cd rishu1947-ops
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # For Unix/macOS
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  Launch Jupyter Lab, Jupyter Notebook, or open the `sentiment_analysis.ipynb` file in an editor like VS Code that supports notebooks.
2.  Run the cells in the notebook sequentially.
3.  The `kagglehub` library will handle the download and caching of the Amazon Reviews dataset automatically. The first run might take a few minutes to download the model and dataset.
4.  The notebook will process a sample of 1,000 reviews from the test set, predict their sentiment, and print the final accuracy score.

## Evaluation and Results

The `nlptown` model is trained to predict a 1-5 star rating. However, the Amazon Reviews dataset provides binary labels: `__label__1` for negative and `__label__2` for positive.

To properly evaluate the model against the dataset, the following mapping was applied:
- **Model's Star Prediction** -> **Binary Label**
  - **1 or 2 stars** -> **1 (Negative)**
  - **4 or 5 stars** -> **2 (Positive)**
  - **3 stars** -> **3 (Neutral)** - These are excluded from the accuracy calculation.

On a sample of 1,000 test reviews, the model achieved the following performance:

- **Accuracy:** **88.50%**

This result demonstrates the model's strong ability to generalize, even though its original training objective (5-star rating) was different from the evaluation task's binary labels.

## Dependencies

The project relies on several key Python libraries:
- `torch`: For deep learning operations.
- `transformers`: For loading the pre-trained model from Hugging Face.
- `pandas`: For data manipulation and analysis.
- `kagglehub`: For easy dataset access from Kaggle.
- `scikit-learn`: For calculating evaluation metrics like accuracy.
- `matplotlib` & `seaborn`: For data visualization.

All dependencies are listed in the `requirements.txt` file.
