# Resume Screening Classifier (AI-11)

## Project Overview

This project develops an end-to-end **Resume Screening Classifier** designed for the 3MTT AI–Machine Learning track. The primary goal is to automate and streamline the initial resume screening process by intelligently classifying resumes into predefined categories and evaluating their fit against specific job descriptions. Built using Google Colab, this solution demonstrates a comprehensive machine learning pipeline, from robust data preprocessing and feature engineering to multi-model training, rigorous evaluation, and a practical skill-matching system.

## Features

-   **Data Loading & Cleaning**: Handles `CSV` datasets, standardizes column names, checks for and removes duplicate entries, and manages potential parsing errors.
-   **Enhanced Data Exploration**: Provides in-depth analysis of dataset shape, category distribution, resume length, and identifies top common words with visualizations.
-   **Robust Text Preprocessing**: A custom `preprocess_text` function performs:
    -   Lowercasing
    -   URL removal
    -   Email removal
    -   Phone number removal
    -   Number removal
    -   Punctuation removal
    -   Special character removal
    -   Stopword removal
    -   Lemmatization
    -   Extra whitespace removal
-   **Advanced Feature Engineering**: Utilizes `TfidfVectorizer` with optimized parameters (`max_features`, `ngram_range=(1,2)`, `min_df=5`, `max_df=0.7`) to transform raw text into numerical features, capturing both single words and common two-word phrases.
-   **Multi-Model Training**: Trains and evaluates various classification models:
    -   Logistic Regression
    -   Linear Support Vector Machine (Linear SVM)
    -   Multinomial Naive Bayes
    -   Random Forest Classifier
-   **Comprehensive Model Evaluation**: Each model is rigorously evaluated using:
    -   Accuracy, Precision, Recall, F1-score
    -   Detailed Classification Reports
    -   Visualized Confusion Matrices
-   **Automated Model Selection & Persistence**: Automatically identifies and saves the best-performing model (based on F1-score) and the fitted `TfidfVectorizer` using `joblib` for future inference.
-   **Intelligent Skill Matching**: Extracts skills from resumes and job descriptions using an expanded dictionary of professional technical and soft skills.
-   **Dynamic Fit Score Generation**: Calculates a holistic `fit_score` (0-100%) by combining:
    -   **70% Skill Match Percentage**: Quantifies how many required job skills are present in the resume.
    -   **30% Model Confidence**: Reflects the confidence of the classifier's category prediction.
-   **Fit Recommendation & Confidence Levels**: Provides a `Fit` or `Not Fit` recommendation based on the fit score and assigns a qualitative confidence level (High, Medium, Low) to the overall assessment.
-   **Modular and Documented Code**: The notebook is structured with clear Markdown explanations, comments, and modular functions, ensuring readability and maintainability.

## Project Architecture

The resume screening system follows a modular machine learning pipeline as depicted below:
Resume Dataset → Data Cleaning → Text Preprocessing → TF-IDF Vectorization → Model Training → Evaluation → Best Model Selection → Skill Matching → Fit Score Calculation → Recruiter Dashboard Integration.

## Installation

To set up the environment, run the following commands in your Colab notebook or local Python environment:

```bash
!pip install nltk scikit-learn pandas numpy matplotlib seaborn joblib
```

Then, download NLTK data:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## Usage

1.  **Upload Dataset**: Ensure your resume dataset is uploaded to your Colab environment or specified path. The dataset should contain at least two columns: one for `category` (or `Category`) and one for `resume_text` (or `Resume`, `Text`).
2.  **Run All Cells**: Execute all cells in the notebook sequentially. This will:
    -   Install dependencies and import libraries.
    -   Load and preprocess your dataset.
    -   Perform data exploration and visualization.
    -   Train and evaluate the classification models.
    -   Select and save the best model and TF-IDF vectorizer.
    -   Demonstrate the `generate_fit_score` function.

### Example Fit Score Generation:

Once the notebook has been run, you can use the `generate_fit_score` function with your `best_model` and `tfidf_vectorizer`:

```python
# Assumes 'best_model', 'tfidf_vectorizer', and 'best_model_name' are loaded from the notebook

sample_job_description = "We are looking for a Data Scientist with strong Python, SQL, and machine learning skills. Experience with deep learning frameworks like TensorFlow or PyTorch is a plus. Good communication and agile methodology experience required."

sample_resume = "Experienced data scientist with a proven track record in Python and SQL. Skilled in various machine learning algorithms and statistical modeling. Familiar with agile development. Strong communication skills."

fit_results = generate_fit_score(
    sample_resume,
    sample_job_description,
    tfidf_vectorizer,
    best_model,
    best_model_name
)

for key, value in fit_results.items():
    print(f"{key.replace('_', ' ').title()}: {value}")
```

## Dataset

The project utilizes a public resume classification dataset. It is expected to contain 'category' and 'resume_text' columns. The dataset undergoes pre-processing to remove duplicates and clean text content.

## Results & Model Performance

After training and evaluation, the models demonstrated the following performance:

| Model                     | Accuracy | Precision | Recall | F1-Score |
| :------------------------ | :------- | :-------- | :----- | :------- |
| Logistic Regression       | 0.8459   | 0.8444    | 0.8459 | **0.8448** |
| Linear SVM                | 0.8327   | 0.8318    | 0.8327 | 0.8319   |
| Multinomial Naive Bayes   | 0.8042   | 0.8059    | 0.8042 | 0.8037   |
| Random Forest             | 0.8042   | 0.8029    | 0.8042 | 0.8028   |

Based on the F1-Score, **Logistic Regression** was selected as the best-performing model. Detailed classification reports, confusion matrices, and a visual comparison of model performance are available within the notebook.

## Recruiter Dashboard

The frontend for the Recruiter Dashboard is envisioned to be built using **Dala Studio**. This dashboard will serve as an intuitive interface for recruiters to interact with the backend machine learning model developed in this Google Colab notebook, facilitating seamless resume uploads, fit score visualization, and candidate recommendations.

## Limitations, Assumptions, and Ethical Considerations

The project acknowledges several limitations, including dependency on training data quality, keyword-based skill extraction, and the potential for bias amplification if historical biases exist in the data. Assumptions include resume and job description clarity, and language consistency. Ethical considerations emphasize transparency, fairness, and data privacy. For a comprehensive discussion, please refer to Section 9 of the main notebook.

## Future Enhancements

Several improvements can be explored to further enhance this project, including:

1.  **Advanced NLP Models**: Incorporating models like Word2Vec, Sentence Transformers, or fine-tuning BERT/RoBERTa/GPT for deeper contextual understanding.
2.  **Sophisticated Skill Extraction**: Implementing Named Entity Recognition (NER) for more precise skill identification.
3.  **OCR Support**: Enabling processing of image-based or PDF resumes.
4.  **Resume Ranking and Similarity**: Developing features to rank resumes based on similarity to job descriptions.
5.  **Recruiter Dashboard (UI/UX)**: Building an interactive web application for recruiters.
6.  **API Deployment**: Exposing the model as a RESTful API for integration.
7.  **Skill Taxonomy and Ontology**: Integrating a comprehensive skill taxonomy for smarter matching.
8.  **Automated Feedback Loop**: Implementing continuous model improvement through recruiter feedback.
9.  **Multilingual Support**: Extending the model to other languages.
10. **Experience Level and Education Extraction**: Adding modules to extract detailed candidate profile information.

For a more detailed discussion on future improvements, please refer to Section 10 of the notebook.

## Contributing

Contributions are welcome! Please feel free to fork the repository, make improvements, and submit pull requests.

## License

This project is licensed under the [MIT License](LICENSE) - see the `LICENSE` file for details. (Replace `LICENSE` with actual license file if created)

## Acknowledgements

-   Special thanks to the 3MTT AI–Machine Learning track for inspiring this project.
-   Grateful for the open-source community and libraries (NLTK, scikit-learn, pandas, numpy, matplotlib, seaborn).
