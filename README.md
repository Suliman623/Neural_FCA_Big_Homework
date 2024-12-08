# Neural_FCA_Big_Homework: Dataset Analysis and Performance Evaluation
**Overview**

This project explores the integration of Neural Networks with Formal Concept Analysis (FCA) to enhance interpretability in machine learning. We compare Neural FCA with standard ML models (e.g., Decision Trees, Random Forests, kNN, and XGBoost) across three datasets:

1. US Electricity
2. World Population
3. DS Salaries

**Objective**

The aim is to evaluate Neural FCA's interpretability and performance against standard ML models in terms of accuracy and F1 score while identifying trade-offs in execution speed and result quality.

**Features**

**Datasets**: Preprocessed data with categorical features binarized and numerical features divided into intervals.

**Models Evaluated**

1. Neural FCA for interpretability.

2. Standard ML models for performance benchmarking.

**Metrics**: Accuracy and F1 score calculated using 5-fold cross-validation.


**Results**


**US Electricity Dataset:**

Neural FCA achieved an accuracy of 91.8% and an F1 score of 94.7%.

Best-performing ML model (Random Forest): Accuracy 93.4%, F1 score 95.7%.

**World Population Dataset:**

Neural FCA achieved an accuracy of 70.9% and an F1 score of 71.1%.

Best-performing ML model (Random Forest): Accuracy 77.3%, F1 score 78.7%.

**DS Salaries Dataset:**

Neural FCA struggled with an accuracy of 25.6% and an F1 score of 25.4%.

Best-performing ML model (Random Forest): Accuracy 58.6%, F1 score 61.2%.


**Challenges**

1. Sofia Algorithm: Replaced "Cbo" for faster execution but impacted result quality.
2. Complexity in DS Salaries Dataset: Wide numerical ranges posed challenges for Neural FCA.

**Setup and Installation**

Clone the repository:

git clone https://github.com/your-username/neural-fca.git
cd neural-fca

**Install dependencies:**

pip install -r requirements.txt

**Run the main script:**

python main.py

**How to Use**

Place your datasets in the datasets/ folder.
Modify main.py to specify the target dataset and features.
Run the script to process data, train models, and evaluate results.


License
This project is licensed under the MIT License.

