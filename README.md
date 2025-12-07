# AI-Powered Job Recommendation System

A machine learning-based job recommendation system that matches user profiles with relevant job listings using skill similarity and demographic filters.

## Project Overview

This project implements a hybrid recommendation approach combining content-based filtering (TF-IDF vectorization on skills) with demographic filtering (location and experience level). The system uses a Random Forest classifier to predict job industries and cosine similarity to match user skills with job requirements.

## Dataset

The dataset contains 50,000 job listings with the following attributes:

- Job Title
- Company
- Location
- Experience Level
- Salary
- Industry
- Required Skills

**Data Distribution:**

- 7 industries (Education, Finance, Healthcare, Manufacturing, Marketing, Retail, Software)
- 7 locations (Sydney, San Francisco, New York, Berlin, London, Bangalore, Toronto)
- 3 experience levels (Entry Level, Mid Level, Senior Level)

## Features

### 1. Industry Classification Model

- Algorithm: Random Forest Classifier (200 estimators)
- Features: Skills (TF-IDF), Experience Level (One-Hot), Location (One-Hot), Salary (MinMax scaled)
- Performance: 98.1% accuracy, 98.1% macro F1-score
- Cross-validation: 98.3% mean accuracy (5-fold stratified CV)

### 2. Job Recommendation Engine

- Skill matching using TF-IDF vectorization and cosine similarity
- Optional filters for location and experience level
- Returns top N most relevant jobs with similarity scores

## Technical Implementation

### Preprocessing Pipeline

```
ColumnTransformer:
- Skills: TfidfVectorizer
- Experience Level: OneHotEncoder
- Location: OneHotEncoder
- Salary: MinMaxScaler
```

### Model Architecture

```
Pipeline:
- Preprocessing (ColumnTransformer)
- Classifier (RandomForestClassifier)
```

## Installation

### Requirements

```
pandas
numpy
scikit-learn
```

### Setup

```bash
pip install pandas numpy scikit-learn
```

## Usage

### Basic Recommendation

```python
# Define user profile
user_profile = {
    'skills': 'Python, Machine Learning, Data Analysis',
    'preferred_location': 'New York',
    'experience_level': 'Mid Level'
}

# Get recommendations
recommendations = recommend_jobs(user_profile, top_n=5)
print_recommendations(recommendations)
```

### Function Parameters

**recommend_jobs(user_profile, top_n=5)**

- `user_profile` (dict): User preferences
  - `skills` (str): Comma-separated skills
  - `preferred_location` (str or None): Location filter
  - `experience_level` (str or None): Experience level filter
- `top_n` (int): Number of recommendations to return

**Returns:** List of tuples containing (job_index, similarity_score, job_details_dict)

## Model Performance

### Classification Metrics

| Metric            | Value   |
| ----------------- | ------- |
| Accuracy          | 98.11%  |
| F1-Score (macro)  | 98.11%  |
| Baseline Accuracy | 14.60%  |
| Improvement       | +83.51% |

### Cross-Validation Results

- Mean Accuracy: 98.3%
- Standard Deviation: 0.1%
- Consistent performance across all folds

### Per-Industry Performance

| Industry      | Precision | Recall | F1-Score |
| ------------- | --------- | ------ | -------- |
| Education     | 1.00      | 1.00   | 1.00     |
| Finance       | 0.94      | 0.93   | 0.93     |
| Healthcare    | 1.00      | 1.00   | 1.00     |
| Manufacturing | 1.00      | 1.00   | 1.00     |
| Marketing     | 1.00      | 1.00   | 1.00     |
| Retail        | 1.00      | 1.00   | 1.00     |
| Software      | 0.93      | 0.94   | 0.94     |

## Project Structure

```
AI-Powered Job Recommendations/
├── job_recommendation_system.ipynb    # Main notebook with implementation
├── job_recommendation_dataset.csv     # Dataset (50,000 records)
└── README.md                          # Project documentation
```

## Methodology

### 1. Data Loading and Exploration

- Load dataset using pandas
- Validate data integrity (no missing values)
- Analyze feature distributions

### 2. Feature Engineering

- TF-IDF vectorization for skill text
- One-hot encoding for categorical variables
- MinMax scaling for numerical features

### 3. Model Training

- Train/test split: 80/20 with stratification
- Random Forest classifier with 200 trees
- Pipeline integration for preprocessing

### 4. Model Evaluation

- Accuracy and F1-score calculation
- Confusion matrix analysis
- Cross-validation for robustness testing

### 5. Recommendation System

- TF-IDF-based skill matching
- Cosine similarity computation
- Demographic filtering

## Key Findings

1. **High Model Accuracy**: The Random Forest classifier achieves 98.1% accuracy, significantly outperforming the baseline (14.6%).

2. **Balanced Performance**: Macro F1-score matches accuracy, indicating consistent performance across all industry classes.

3. **Finance-Software Confusion**: Minor confusion between Finance and Software industries (189 misclassifications), likely due to overlapping skill requirements (Python, SQL, data analysis).

4. **Robust Predictions**: Cross-validation results show consistent performance (std = 0.1%), indicating the model generalizes well.

## Limitations

- Skill matching relies on exact or similar keyword matches
- No semantic understanding of skill relationships
- Limited to predefined location and experience level values
- Does not account for user preferences beyond skills, location, and experience

Kaggle Dataset:https://www.kaggle.com/datasets/samayashar/ai-powered-job-recommendations
