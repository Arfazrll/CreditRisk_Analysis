# Credit Risk Analysis

## Overview
This project aims to perform **Credit Risk Analysis** on a dataset of loan applicants. The objective is to predict the likelihood of loan defaults using various machine learning models, including **ensemble models**, **LSTM neural networks**

The analysis leverages business metrics such as **approval rate**, **default capture rate**, **precision**, and **AUC** to evaluate the models. The goal is to identify which model is most effective in predicting loan defaults, optimizing for both predictive accuracy and business profitability.

## Project Structure

The project consists of the following stages:

1. **Data Preprocessing**:
   - Data cleaning and feature selection to handle missing values and imbalanced data.
   - Encoding categorical variables and scaling numerical features to prepare the data for model training.

2. **Model Training**:
   - Training several machine learning models, including:
     - **LSTM (Long Short-Term Memory networks)**
   - Models are evaluated using both traditional performance metrics and business-specific metrics.

3. **Model Evaluation**:
   - Models are assessed using **accuracy**, **precision**, **recall**, **F1 score**, and **AUC**.
   - Additional evaluations are performed to determine the **total cost** associated with false positives and false negatives.

4. **Optimization**:
   - Hyperparameters for each model are optimized using **GridSearchCV** and **cross-validation** to improve model performance.

5. **Risk Assessment and Classification**:
   - **Risk categories** are defined based on predicted probabilities.
   - A comprehensive **risk distribution analysis** is performed to classify loan applicants into risk groups (low, medium, high).

6. **ROI Analysis**:
   - A **Return on Investment (ROI)** analysis is performed to measure the financial impact of using the predictive models.

## Libraries and Tools

This project utilizes several libraries and tools:
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Matplotlib & Seaborn**: For data visualization (charts, confusion matrices, etc.).
- **Scikit-learn**: For training models, cross-validation, and evaluating metrics.
- **TensorFlow/Keras**: For training deep learning models (LSTM).
- **Imbalanced-learn (SMOTE)**: For addressing class imbalance.
- **SciPy**: For statistical tests and optimizations.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
````

The `requirements.txt` file contains all required libraries for running the project.

## Usage

1. **Load the Dataset**:
   The dataset is loaded at the start of the notebook. You can adjust the file path or ensure that the dataset is placed correctly.

2. **Data Preprocessing**:
   The dataset is cleaned and transformed for model training. Missing values are handled, categorical variables are encoded, and numerical features are scaled.

3. **Model Training**:
   Models are trained using multiple algorithms, including **LSTM** and **ensemble methods**. The **LSTM** model is used for sequential data, while ensemble models combine predictions from multiple classifiers.

4. **Model Evaluation**:
   Performance metrics such as **accuracy**, **precision**, **recall**, and **AUC** are computed to assess the effectiveness of each model. Additionally, business metrics like **default capture rate** and **approval rate** are considered.

5. **Risk Assessment**:
   Based on the model's predicted probabilities, loan applicants are classified into different risk segments, which are then analyzed for the overall risk distribution.

6. **ROI Calculation**:
   ROI is calculated based on the total cost savings from using predictive models compared to the baseline (approving all loans without prediction).

## Results and Conclusions

### Key Findings:

* **Model Performance**: The models perform differently depending on the evaluation metric used. For example, ensemble models tend to provide high AUC and better handling of imbalanced classes, while LSTM models perform well with sequential data.
* **Business Metrics**: Using these models results in a significant reduction in business costs associated with **false positives** (approving loans that lead to defaults) and **false negatives** (denying loans that could have been repaid).
* **Risk Segmentation**: The models provide useful insights into risk segmentation. Applicants classified into **high risk** categories tend to have a significantly higher default rate, while **low-risk** groups show lower chances of default.
* **ROI**: The **ensemble models** (such as **XGBoost** and **LightGBM**) result in the highest ROI, as they lead to cost savings by preventing defaults while allowing the bank to approve more loans safely.

### Business Implications:

* **Cost Savings**: The models help reduce financial losses by identifying applicants who are more likely to default. By optimizing the threshold for classification, the bank can approve more loans with less risk.
* **Improved Decision Making**: Using these predictive models allows for data-driven decisions on loan approvals, significantly improving profitability and reducing risks.
* **Risk Mitigation**: Identifying **high-risk** applicants early on can prevent financial losses. The models provide a strong basis for decision-makers to make informed choices about loan approval.

### Conclusion:

The project demonstrates that machine learning models, particularly ensemble methods and deep learning (LSTM), can significantly improve the accuracy of credit risk predictions. By integrating business-specific metrics such as **cost of false positives** and **false negatives**, this analysis provides a comprehensive evaluation of the models' financial impact. It concludes that adopting predictive models can help reduce risks, increase profitability, and make more informed decisions in the credit approval process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


### Explanation:
- **Overview**: Provides an introduction to the project and its goals.
- **Project Structure**: Details the various stages of the project, such as data preprocessing, model training, and evaluation.
- **Libraries and Tools**: Lists the essential libraries used in the project.
- **Installation**: Provides instructions on how to install the required dependencies.
- **Usage**: Explains how to load the dataset, train models, evaluate them, and analyze the results.
- **Results and Conclusions**: Summarizes key findings, business implications, and conclusions from the analysis.

