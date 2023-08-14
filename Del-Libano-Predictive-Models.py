import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.markdown("asdasdasd")

def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    # Convert "documentDate" column to datetime format
    data["documentDate"] = pd.to_datetime(data["documentDate"])

    # Create a DataFrame with all dates within the desired range
    date_range = pd.date_range(start=data["documentDate"].min(), end=data["documentDate"].max(), freq="D")
    date_range_df = pd.DataFrame({"documentDate": date_range})

    # Merge date_range_df with data based on "documentDate"
    salesbyday = date_range_df.merge(data, on="documentDate", how="left")

    # Find the index of the first occurrence of "2021-06-28"
    start_index = salesbyday[salesbyday['documentDate'] == '2021-06-28'].index[0]

    # Fill missing values with 0
    salesbyday.fillna(0, inplace=True)

    # Add a new column for the days of the week
    # salesbyday["dayOfWeek"] = salesbyday["documentDate"].dt.day_name()

    # Remove the first 37 observations and start from "2021-06-28"
    salesbyday = salesbyday.iloc[start_index:].reset_index(drop=True)

    # Drop rows where transactionType is "Returned"
    salesbyday = salesbyday[(salesbyday['transactionType'] != 'Returned')]

    # Select rows where the transactionType column is equal to “Discounted” and sets the corresponding value in the quantity column to zero
    salesbyday.loc[salesbyday['transactionType'] == 'Discounted', 'quantity'] = 0

    # Set 'branchesCombined' column as 'customerName'
    salesbyday['branchesCombined'] = salesbyday['customerName']

    # Update 'branchesCombined' column for rows containing '-'
    salesbyday.loc[salesbyday['branchesCombined'].notna() & salesbyday['branchesCombined'].str.contains('-', na=False), 'branchesCombined'] = salesbyday.loc[salesbyday['branchesCombined'].notna(), 'branchesCombined'].str.split('-').str[0]

    # Create a copy of salesbyday DataFrame
    new_data = salesbyday.copy()

    # Extract the weight information from the itemName column
    new_data['weight'] = new_data['itemName'].str.extract(r'(\d+)(KG|TON)', expand=False)[0].astype(float)

        
    # Assign the weight 0.008 only if itemName is "35522 Tagliatel.Ricci 10x400G"
    new_data.loc[new_data['itemName'] == '35522 Tagliatel.Ricci 10x400G', 'weight'] = 0.008


    # Convert the weight to tons and calculate quantityInTons
    new_data['quantityInTons'] = new_data.apply(lambda row: row['quantity'] * row['weight'] / (1000 if 'KG' in str(row['itemName']) else 1) if pd.notnull(row['itemName']) else np.nan, axis=1)


    # Drop the weight column as it is no longer needed
    new_data.drop('weight', axis=1, inplace=True)


    # Convert the documentDate column to datetime format
    new_data['documentDate'] = pd.to_datetime(new_data['documentDate'])

    # Extract day, month, and year features
    new_data['day'] = new_data['documentDate'].dt.day
    new_data['month'] = new_data['documentDate'].dt.month
    new_data['year'] = new_data['documentDate'].dt.year

    # Specify the desired column order
    # column_order = ['documentDate', 'day', 'dayOfWeek', 'month', 'year', 'documentName', 'transactionType', 'customerName', 'branchesCombined', 'customerType', 'disctrict', 'city', 'itemName', 'itemType', 'quantity', 'quantityInTons', 'value']
    column_order = ['documentDate', 'day', 'month', 'year', 'documentName', 'transactionType', 'customerName', 'branchesCombined', 'customerType', 'disctrict', 'city', 'itemName', 'itemType', 'quantity', 'quantityInTons', 'value']
    new_data = new_data[column_order]

    # Define the item types
    item_types = ['3559 Vermicelli', '3555 Fusilli', '3554 Penne', '3556 Rigate', '3550 Pasta Integrale', '3551 Spaghetti', '3553 Lasagna', '35522 Tagliatelle Ricci', '3552 Tagliatelle']
    '''
    new_data.loc[new_data.value < 50, "customer_size"] = "small"
    new_data.loc[(new_data.value > 50) & (new_data.value < 500) , "customer_size"] = "medium"
    new_data.loc[new_data.value > 500, "customer_size"] = "big"
    '''
    # Create dummy variables for the itemType column
    itemType_dummies = pd.get_dummies(new_data['itemType'].astype("str"), prefix='itemType')

    # Concatenate the dummy variables with the original data
    new_data = pd.concat([new_data, itemType_dummies], axis=1)

    # Define a function to multiply quantity in tons with each item type
    def multiply_quantity_with_item_type(row):
        for item in item_types:
            col_name = f'itemType_{item}'
            row[col_name] *= row['quantityInTons']
        return row

    # Apply the function to the dataframe
    new_data = new_data.apply(multiply_quantity_with_item_type, axis=1)

    # Specify the columns to use
    # cols_to_use = ['documentDate', 'day', 'dayOfWeek', 'month', 'year', 'documentName', 'transactionType', 'customerName', 'branchesCombined', 'customerType', 'disctrict', 'city', 'itemName', 'itemType', 'quantity', 'quantityInTons', 'value', 'itemType_0', 'itemType_3551 Spaghetti', 'itemType_3552 Tagliatelle', 'itemType_35522 Tagliatelle Ricci', 'itemType_3553 Lasagna', 'itemType_3554 Penne', 'itemType_3555 Fusilli', 'itemType_3556 Rigate']
    cols_to_use = ['documentDate', 'day', 'month', 'year', 'documentName', 'transactionType', 'customerName', 'branchesCombined', 'customerType', 'disctrict', 'city', 'itemName', 'itemType', 'quantity', 'quantityInTons', 'value', 'itemType_0', 'itemType_3551 Spaghetti', 'itemType_3552 Tagliatelle', 'itemType_35522 Tagliatelle Ricci', 'itemType_3553 Lasagna', 'itemType_3554 Penne', 'itemType_3555 Fusilli', 'itemType_3556 Rigate']

    # Select the desired columns from the dataframe
    new_data = new_data[cols_to_use]

    # Aggregate by 'documentDate'
    agg_dict = {col: 'first' for col in new_data.columns if new_data[col].dtype == 'object'}
    agg_dict.update({col: 'sum' for col in new_data.columns if new_data[col].dtype != 'object'})
    agg_dict.pop('day', None)
    agg_dict.pop('month', None)
    agg_dict.pop('year', None)
    agg_dict.pop('documentDate', None)

    # Group by 'documentDate' and aggregate the columns
    new_data = new_data.groupby('documentDate').agg(agg_dict).reset_index()

    # Recalculate 'day', 'month', and 'year' based on 'documentDate'
    new_data['day'] = new_data['documentDate'].dt.day
    new_data['month'] = new_data['documentDate'].dt.month
    new_data['year'] = new_data['documentDate'].dt.year

    # Define the columns to drop from the final DataFrame
    columns_to_drop = [ 'documentName', 'transactionType', 'customerName', 'branchesCombined', 'customerType', 'disctrict', 'city', 'itemName', 'itemType', 'quantity', 'itemType_0']

    # Drop the specified columns from the DataFrame
    trimmed_new_data = new_data.drop(columns_to_drop, axis=1)

    return trimmed_new_data


def getSeason(data):
    data['season'] = "null"
    # Winter
    data.loc[data.month.isin([12,1,2]), 'season'] = 1
    # Spring
    data.loc[data.month.isin([3,4,5]), 'season'] = 2
    # Summer
    data.loc[data.month.isin([6,7,8]), 'season'] = 3
    # Fall
    data.loc[data.month.isin([9,10,11]), 'season'] = 4
    
    return data

def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
    data = getSeason(data)
    holidays = pd.DataFrame({
    'Holiday/Fasting Period': ['New Year\'s Day', 'Armenian Orthodox Christmas', 'St Maroun\'s Day', 'Good Friday', 'Easter Sunday', 'Orthodox Easter Sunday', 'Ramadan', 'Eid al-Fitr', 'Labour Day', 'Martyrs\' Day', 'Resistance and Liberation Day', 'Eid al-Adha', 'Assumption of Mary', 'Islamic New Year', 'Ashura', 'All Saints\' Day', 'Independence Day', 'Christmas Day', 'Lent'],
    'Start Date': ['01-01', '01-06', '02-09', '04-15', '04-17', '04-24', '04-02', '05-02', '05-01', '05-06', '05-25', '07-09', '08-15', '08-08', '08-18', '11-01', '11-22', '12-25','03-02'],
    'End Date': ['01-01','01-06','02-09','04-15','04-17','04-24','05-01','05-03','05-01','05-06','05-25','07-12','08-15','08-09','08-19','11-01','11-22','12-25','04-16'],
    'Fixed/Variable': ['Fixed','Fixed','Fixed','Variable','Variable','Variable','Variable','Variable','Fixed','Fixed','Fixed','Variable','Fixed','Variable','Variable','Fixed','Fixed','Fixed','-Variable']
    })

    holidays['Start Date'] = pd.to_datetime(holidays['Start Date'], format='%m-%d')
    holidays['End Date'] = pd.to_datetime(holidays['End Date'], format='%m-%d')
    data['documentDate'] = pd.to_datetime(data['documentDate'], format='%Y-%m-%d')

    data['holiday'] = data['documentDate'].apply(lambda x: any((x.month == row['Start Date'].month) & (x.day >= row['Start Date'].day) & (x.day <= row['End Date'].day) for _, row in holidays.iterrows())).astype(int)

    return data

def load_data():
    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        st.write(data.head())
        return data


data = load_data()
if data is not None:
    transformed_data = data_preprocessing(data)
    feature_engineered_data = feature_engineering(transformed_data)
    st.write(feature_engineered_data)



def run_xgboost(feature_engineered_data):
    # Best hyperparameters obtained from Optuna but with gamma =0
    best_hyperparameters = {
        'max_depth': 4,
        'n_estimators': 555,
        'learning_rate': 0.040389286469709164,
        'min_child_weight': 7,
        'subsample': 0.19305186666172608,
        'colsample_bytree': 0.5868453113051293,
        'gamma': 0 # controls the trade-off between model complexity and fitting the training data
    }

    X = feature_engineered_data[['documentDate', 'day', 'month', 'year', 'season', 'holiday']]
    X.set_index('documentDate', inplace=True)
    X.index = pd.to_datetime(X.index)
    X['season'] = X['season'].astype('int')

    items = ['itemType_3551 Spaghetti',
             'itemType_3552 Tagliatelle', 'itemType_35522 Tagliatelle Ricci',
             'itemType_3553 Lasagna', 'itemType_3554 Penne', 'itemType_3555 Fusilli',
             'itemType_3556 Rigate']

    for item in items:
        print(item)
        y = feature_engineered_data[['documentDate', item]]
        y.set_index('documentDate', inplace=True)
        y.index = pd.to_datetime(y.index)

        X_train, X_test = X[:-14], X[-14:]
        y_train, y_test = y[:-14], y[-14:]

        # Initialize the XGBRegressor with the best hyperparameters
        model = XGBRegressor(objective='reg:squarederror', **best_hyperparameters)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_train = np.where(y_pred_train < 0, 0, y_pred_train)
        y_pred_test = model.predict(X_test)
        y_pred_test = np.where(y_pred_test < 0, 0, y_pred_test)

        # Calculate MAE, MSE, RMSE for train set and test set
        mae_train, mae_test = mean_absolute_error(y_train, y_pred_train), mean_absolute_error(y_test, y_pred_test)
        mse_train, mse_test = mean_squared_error(y_train, y_pred_train), mean_squared_error(y_test, y_pred_test)
        rmse_train, rmse_test = np.sqrt(mse_train), np.sqrt(mse_test)

        print(f"Train set metrics: (MAE: {mae_train} | MSE: {mse_train} | RMSE: {rmse_train})")
        print(f"Test set metrics: (MAE: {mae_test} | MSE: {mse_test} | RMSE: {rmse_test})")

        # Plotting predictions vs actuals for the test set
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test.index, y_test.values, label='Actuals')
        ax.plot(y_test.index, y_pred_test, color='r', label='Predicted')
        ax.legend()
        ax.set_title(f'Actual vs Predicted for Test set: {item}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Production')

if data is not None:
    run_xgboost(feature_engineered_data)

