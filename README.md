# 📈 Enhancing Financial Decision-Making: Predicting Daily Disbursed Amounts with XGBoost, Random Forest, Gradient Boosting

## 🎯 Objective

The primary purpose of this forecasting model is to provide valuable insights and predictions regarding future disbursement outcomes within the microfinance industry. By leveraging historical data and advanced analytical techniques, this model aims to assist decision-makers and stakeholders in making informed strategic decisions. The model's objective is to offer accurate and timely forecasts.

## 📊 Data for Model

The foundation of this forecasting model rests upon a robust dataset acquired from the company's data source "Mohasseel." The dataset encompasses a comprehensive daily record spanning from April 2019 to August 2023. This expansive temporal scope enables the model to capture diverse patterns that have emerged within the data.

### Data Structure

- **Original Data Columns:**
  - Daily Date
  - Disbursed Amount
  
- **Created Columns:**
  - Ramadan Season (Binary Feature)
  - Public Holidays (Binary Feature)
  - Weekends (Binary Feature)

## 📝 Notebook Structure

1. **Data Exploration:**
   - Analyzing the dataset to understand its structure and characteristics.

2. **Features Creation:**
   - Developing new features to improve model performance.

3. **Time Series Forecasting:**
   - Applying different models (XGBoost, Random Forest, Gradient Boosting) to forecast disbursement outcomes.


### 📅 Unveiling the Magic of Date Index Features

In our quest for precision in disbursement predictions, we wield date index features as our secret weapons. These features empower our XGBoost model to decipher and harmonize with the enchanting rhythms of seasonality hidden within our data.


🌟 **Discovering Seasonal Patterns**: Date index features, including 'month', 'quarter', 'year', 'dayofmonth', 'dayofweek', 'dayofyear', and 'weekofyear', light our path like celestial constellations. They reveal the captivating patterns that ebb and flow within our dataset. Understanding these temporal dances is paramount because disbursements often sway with the changing seasons.


🎯 **Enhancing Precision**: By infusing date index features into our XGBoost model, we equip it with precision-enhancing lenses. These special lenses allow our model to focus sharply on the intricate dance of time and make disbursement predictions with remarkable accuracy.


💡 **Guiding Informed Decisions**: In the dynamic world of finance, each decision carries weight. With our XGBoost model, fueled by date index features, we navigate the financial cosmos with confidence. We harness insights from the past and present to make decisions that gleam with wisdom.


With these date index features as our guiding stars, we not only simplify the complexity of disbursement predictions but also illuminate our path toward more informed and strategic decision-making.
