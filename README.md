# House Price Prediction

This project aims to predict house prices based on their square footage, number of bedrooms, and number of bathrooms using a linear regression model. The following steps are included:

1. Data Loading
2. Exploratory Data Analysis (EDA)
3. Feature Extraction
4. Model Training
5. Predictions
6. Performance Metrics Evaluation


## Data Files

- `train.csv`: Training dataset
- `test.csv`: Test dataset
- `sample_submission.csv`: Sample submission file


## Steps to Run the Notebook

1. **Import Libraries**: Import necessary libraries such as pandas, numpy, matplotlib, seaborn, sklearn.

2. **Load the Data**: Load the train and test datasets.

    ```python
    train = pd.read_csv('/mnt/data/train.csv')
    test = pd.read_csv('/mnt/data/test.csv')
    ```

3. **Data Exploration**: Perform exploratory data analysis to understand the data distribution and relationships between features.

    - Display the first few rows of the dataset.
    - Summary statistics.
    - Check for missing values.
    - Visualize the distribution of the target variable (SalePrice).
    - Plot pairwise relationships between essential features and the target variable.
    - Display the correlation matrix.
    - Scatter plots with regression lines.
    - Box plots and violin plots to check for outliers and distribution.

4. **Feature Selection**: Select the essential features for the model.

    ```python
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    X = train[features]
    y = train['SalePrice']
    ```

5. **Split the Data**: Split the data into training and testing sets.

    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ```

6. **Model Training**: Train a linear regression model on the training data.

    ```python
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

7. **Model Evaluation**: Evaluate the model on the testing data using mean squared error, root mean squared error, and R^2 score.

    ```python
    from sklearn.metrics import mean_squared_error, r2_score
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    ```

8. **Predictions**: Make predictions on the test dataset and save the results in a submission file.

    ```python
    X_submission = test[features]
    submission_predictions = model.predict(X_submission)
    submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': submission_predictions})
    submission.to_csv('/mnt/data/submission.csv', index=False)
    ```

## Visualizations

- Distribution of Sale Prices
- Pairwise Relationships
- Correlation Matrix
- Scatter Plots with Regression Lines
- Box Plots and Violin Plots

## Performance Metrics

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R^2 Score

## Submission

The submission file `submission.csv` contains the predicted house prices for the test dataset.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- sklearn

To install the required libraries, use:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
