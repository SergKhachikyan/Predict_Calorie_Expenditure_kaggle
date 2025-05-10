# 🏃‍♂️ Predict Calorie Expenditure from Activity Data 🏃‍♀️

This project predicts calorie expenditure based on activity data using machine learning models. The dataset is sourced from Kaggle and contains information about various human activities along with their corresponding energy expenditure. The goal of this project is to develop a model that accurately predicts calorie expenditure based on activity data.

## 🚀 Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/SergKhachikyan/Predict_Calorie_Expenditure_kaggle.git
    ```

2. **Navigate to the project directory:**
    ```bash
    cd Predict_Calorie_Expenditure_kaggle
    ```

3. **Install the dependencies:**
    Make sure Python 3.x is installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

4. **Launch the project:**
    Open the Jupyter Notebook interface to explore and run the notebooks:
    ```bash
    jupyter notebook
    ```

## 🔧 Technologies

This project uses the following technologies:
- **Python** 🐍: Programming language.
- **Pandas** 📊: Data manipulation and analysis.
- **NumPy** 🔢: Numerical computations.
- **Scikit-learn** 🔬: Machine learning library used for model building.
- **Matplotlib & Seaborn** 📈: Data visualization libraries.
- **Jupyter Notebook** 📓: Interactive environment to run and explore the code.

## 📝 How to Use

1. **Prepare the dataset:**
    - Download the dataset from Kaggle: [Calorie Expenditure Dataset](https://www.kaggle.com/datasets/).
    - Place your dataset inside the `data/` folder.

2. **Train the model:**
    - Open Jupyter Notebook:
      ```bash
      jupyter notebook
      ```
    - Open the relevant notebook (e.g., `calorie_expenditure_model.ipynb`) and run the cells sequentially to preprocess the data, build the model, and start training.

3. **Make predictions:**
    After training the model, you can use the inference script to predict calorie expenditure based on new activity data:
    ```bash
    python src/inference.py --input_data path/to/your/activity_data.csv
    ```

## 💡 Features

- **Predict Calorie Expenditure** 🔥: Predict the number of calories burned based on activity data such as heart rate, duration, and other features.
- **Data Preprocessing** 🔄: Clean and preprocess raw data before feeding it to the machine learning model.
- **Model Evaluation** 📊: Evaluate the model’s performance using metrics like mean squared error (MSE) and R-squared.
- **Visualization** 🌈: Visualize training performance, model predictions, and error distributions.

## 🧠 Model Architecture

- **Input Layer**: Takes in various activity features such as duration, heart rate, age, and weight.
- **Regression Model**: A regression model (like Random Forest or Linear Regression) predicts the calorie expenditure.
- **Output Layer**: Outputs the predicted calorie expenditure for a given activity.

## 🏆 Model Performance

- **Loss Function**: Mean Squared Error (MSE), suitable for regression tasks.
- **Metrics**: Model performance evaluated by R-squared (R²) and Mean Squared Error (MSE).

## 📊 Visualizations

- **Training Curves**: Visualize loss during training epochs.
- **Model Predictions**: Compare predicted calorie expenditure with actual values.
- **Feature Importance**: Visualize the importance of different features in the prediction.

![calorie](https://github.com/user-attachments/assets/7287505e-87ba-42c6-b9d2-860165158602)


---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the project, open issues, or submit pull requests.
