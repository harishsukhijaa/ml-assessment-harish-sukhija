Part B: Business Case Analysis - Promotion Effectiveness

Name: Harish Sukhija
Student ID: bitsom_ba_2511153


B1. Problem Formulation

(a) Machine Learning Formulation

Target Variable: items_sold (Continuous numerical value).
Input Features: promotion_type, location_type, store_size, competition_density, month, is_weekend, is_festival, monthly_footfall.
Problem Type: Supervised Machine Learning - specifically, Regression.
Justification: We are trying to predict a continuous numerical output (how many items will be sold) rather than classifying data into discrete categories. Once the regression model predicts the expected items_sold for every possible promotion at a specific store, we can simply recommend the promotion that yields the highest predicted value.

(b) Items Sold vs. Sales Revenue

Using Items Sold (Sales Volume) is much more reliable than Sales Revenue for evaluating promotion effectiveness. Some promotions, like Flat Discount or BOGO, naturally reduce the revenue per item. If a store sells 100 items at a 50% discount, the revenue might look lower than a normal day, but the promotion itself was highly effective at driving volume and clearing inventory. 
Broader Principle: This illustrates the principle of Metric Alignment. Your target variable must isolate the exact behavior you are trying to optimize without being skewed by confounding variables like price reductions dictated by the promo itself.

(c) Alternative Modeling Strategy

A single global model might fail because a Flat Discount might work wonders in a rural, price-sensitive location, but fail in an urban, premium location.
Alternative Strategy: First, use an unsupervised learning algorithm like K-Means to segment the 50 stores into 3 or 4 clusters based on demographics, location type, and store size. Then, train a separate Regression model for each cluster. This strikes a balance: it captures local regional behaviors better than a single global model, but still has enough data per model to learn effectively.


B2. Data and EDA Strategy

(a) Table Joins and Data Grain

Joining Strategy: Start with the Calendar Table as the base to ensure no missing dates. Left join the Store Attributes. Then, left join the Promotions table matching on store_id and month. Finally, aggregate the Transactions table to the daily level and join it.
Final Grain: One row = One Store, per Day (Store-Day level).
Aggregations Before Modeling: Sum of daily items sold, sum of daily revenue, calculation of daily footfall. 

(b) Exploratory Data Analysis (EDA)

1. Bar Chart (Average Items Sold by Promotion Type): To see which promotion generally performs best globally.
2. Grouped Boxplot (Items Sold by Location Type & Promotion): To identify interactions. For example, does BOGO work better in Rural vs Urban?
3. Time Series Line Plot (Sales Volume over Time): To identify seasonality and trends and see if we need to extract festival features.
4. Correlation Heatmap (Numeric Features): To check how competition density and store size correlate with sales.

(c) Handling Imbalance (80% No Promotion)

This is not a traditional class imbalance problem because it is a regression task, but the model will be biased toward baseline sales.
Solution: Treat "No Promotion" as a standard categorical value in the promotion_type feature. This allows the model to learn the strict baseline. We can also use Sample Weighting during model training, giving slightly higher weights to the 20% of rows where promotions were active.


B3. Model Evaluation and Deployment

(a) Train-Test Split and Metrics

Split Strategy: A random split is inappropriate because this is temporal data; randomly splitting would cause data leakage (predicting the past using future trends). I would use a Time-Series Split: Train on Years 1 and 2, and Test on Year 3.
Metrics: I would use MAE (Mean Absolute Error) because it is easy to interpret in business terms (e.g., "Our model's prediction is off by an average of 15 items per day"). I would also look at RMSE if miscalculating inventory by a large margin is very costly.

(b) Investigating Different Recommendations

I would use SHAP values or look at tree-based Feature Interactions.
Communication: I would explain to the marketing team that in December, the festival features are highly active. The model learned that during festive seasons, customers prefer long-term value, heavily boosting the weight of Loyalty Points. In March, which is a slower month, the model relies on the Flat Discount feature weight to artificially stimulate footfall.

(c) End-to-End Deployment

Saving: Export the trained Pipeline including scalers, encoders, and the model as a pickle file.
Prediction Pipeline: Build a Python script that runs on the 25th of every month. It pulls the upcoming month's calendar data and store attributes, simulates all 5 promotion types for each store, and outputs the promotion that yields the highest predicted items sold.
Monitoring: Implement Data Drift monitoring to check if incoming footfall data looks radically different from training data. If the MAE exceeds a predefined business threshold, it triggers an alert for the data science team to retrain the model.