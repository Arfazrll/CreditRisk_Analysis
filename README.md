
# Credit Risk Prediction and Analysis using LSTM (Long Short-Term Memory)

## Overview

This project focuses on predicting **credit risk** using **LSTM (Long Short-Term Memory)**, a type of **deep learning** model. LSTM is a variant of **Recurrent Neural Networks (RNNs)** that is well-suited for learning from sequential data, where temporal dependencies exist—making it particularly effective for analyzing financial data over time.

The objective of this project is to evaluate the likelihood of loan defaults using the LSTM model. By capturing long-term dependencies and patterns in sequential data (such as credit history and financial behavior), the LSTM model offers powerful predictive capabilities for **credit risk** analysis.

The analysis leverages business-specific metrics, including **approval rate**, **default capture rate**, **precision**, and **AUC**, to assess the effectiveness of the model in predicting defaults while optimizing for **business profitability**.

## Project Structure

The project follows the stages outlined below, with a strong focus on LSTM-based modeling:

### 1. **Data Preprocessing**

- **Data Cleaning and Feature Selection**: 
  - Handling missing values and addressing any imbalanced data using techniques like **SMOTE (Synthetic Minority Over-sampling Technique)**.
  - Encoding categorical variables (e.g., employment status, loan type) and scaling numerical features for optimal LSTM training.
  
- **Reshaping for LSTM**:
  - Preparing the data for LSTM involves reshaping it into a 3D format `[samples, timesteps, features]`, which captures the sequential nature of the financial data.

### 2. **Model Training with LSTM**

The core of the project is the training of an **LSTM model**. LSTM is well-suited for capturing long-term dependencies in data, making it ideal for time-series and sequential data, such as credit scores and historical loan information.

- **Model Architecture**:
  - **Input Layer**: Accepts reshaped data.
  - **LSTM Layers**: Stacked LSTM layers capture complex sequential patterns.
  - **Dropout Layers**: Introduced to prevent overfitting by randomly setting a fraction of input units to zero during training.
  - **Dense Output Layer**: The output layer is a **sigmoid activation** to predict binary outcomes: high-risk or low-risk applicants.

- **Compilation and Training**:
  - The model is compiled with the **Adam optimizer** and the **binary cross-entropy** loss function for binary classification tasks.
  - The LSTM model is trained using **backpropagation through time** (BPTT) to adjust the weights based on the sequential nature of the data.

### 3. **Model Evaluation**

The performance of the LSTM model is evaluated using the following metrics:

- **Traditional Metrics**: 
  - **Accuracy**, **Precision**, **Recall**, **F1-score**, and **AUC** (Area Under the Curve).
  
- **Business Metrics**:
  - **Default Capture Rate**: Measures the proportion of actual defaults captured by the model.
  - **Approval Rate**: Measures how many loan applicants the model approves, balancing risk and profitability.

### 4. **Optimization**

- **Hyperparameter Tuning**:
  - **GridSearchCV** and **cross-validation** are used to optimize hyperparameters, such as the number of LSTM units, learning rate, and batch size, for better performance.

### 5. **Risk Assessment and Classification**

- Based on the model's predicted probabilities, applicants are classified into risk categories: **low**, **medium**, or **high**.
- **Risk Distribution Analysis**:
  - A comprehensive analysis is performed to understand the proportion of applicants in each risk category and evaluate the model's ability to distinguish high-risk applicants from low-risk ones.

### 6. **ROI Analysis**

- **Return on Investment (ROI)**:
  - An ROI analysis is performed to measure the financial impact of the LSTM model by comparing the total cost savings from avoiding defaults to the baseline scenario of approving all loans without prediction.

## Libraries and Tools

This project uses several key libraries for data preprocessing, deep learning model training, and evaluation:

- **Pandas**: For data manipulation and handling large datasets.
- **NumPy**: For numerical operations and array handling.
- **Matplotlib** & **Seaborn**: For data visualization (e.g., confusion matrices, risk distribution plots, ROC curves).
- **Scikit-learn**: For model evaluation, cross-validation, and computing performance metrics.
- **TensorFlow/Keras**: For building and training the **LSTM** deep learning model.
- **Imbalanced-learn (SMOTE)**: For handling class imbalance in the dataset.
- **SciPy**: For statistical tests and optimizations.

## Installation

To install the necessary dependencies, run the following command:

```bash
pip install -r requirements.txt
````

The `requirements.txt` file includes all required libraries for running the project.

## Usage

### 1. **Load the Dataset**

Ensure that the dataset is correctly placed in the directory and loaded into the project for processing.

### 2. **Data Preprocessing**

The dataset is cleaned, missing values are handled, categorical variables are encoded, and numerical features are scaled for optimal model performance.

### 3. **Model Training**

Train the **LSTM model** using the processed data. The model is designed to learn long-term dependencies in the financial data and predict the likelihood of loan defaults.

### 4. **Model Evaluation**

Evaluate the model's performance using **accuracy**, **precision**, **recall**, **F1-score**, and **AUC**. Business metrics such as **default capture rate** and **approval rate** are also computed to assess profitability.

### 5. **Risk Assessment**

Classify applicants into **low**, **medium**, or **high** risk categories based on predicted probabilities, and perform a detailed risk distribution analysis.

### 6. **ROI Calculation**

Calculate the ROI based on the cost savings from preventing loan defaults compared to a baseline model that approves all loans.

## Results and Conclusions

### Key Findings:

* **Model Performance**: The **LSTM model** excels in capturing long-term dependencies in the data, which is crucial for understanding financial behaviors over time.

  * **Business Metrics**: Using the LSTM model reduces business costs related to false positives and false negatives, leading to improved profitability.
  * **Risk Segmentation**: The model's ability to classify applicants into **high**, **medium**, or **low-risk** categories provides valuable insights into managing loan approval decisions.

### Business Implications:

* **Cost Savings**: By identifying high-risk applicants early, the LSTM model can prevent financial losses due to defaults.
* **Improved Decision Making**: Data-driven predictions help banks approve loans more confidently while minimizing risk.
* **Risk Mitigation**: Early identification of high-risk applicants leads to better risk management and informed decisions about loan approval.

## Conclusion

This project demonstrates that **LSTM** deep learning models can significantly enhance **credit risk prediction** by effectively handling sequential data. By incorporating business-specific metrics like **default capture rate** and **approval rate**, this project highlights the model's financial impact. **LSTM** models are well-suited for time-series financial data, offering a valuable tool for improving loan approval decisions and reducing defaults in the banking sector.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.




