# Pump-it-Up-XGBoost-Ensemble
Ensemble of XGBoost models to build a solution for DrivenData.com's Pump it Up: Data Mining the Water Table competion

Matt Brown                

email: matthew.brown.iowa@gmail.com

Location: Washington, DC

Website: www.DrivenData.org

Competition: Pump it Up: Data Mining the Water Table

Solution is 9th out of 2017 participants

Software Tools: XGBoost package in R

Brief Model Description: Ensemble of 11 XGBoost models with equal weight to each solution

Feature Selection
The original data set contained 40 variables. I reduced it down to 26 variables by removing variables
that were similar/duplicates of other variables.

Feature Engineering
For the construction_year and gps_height I used the median of them to replace the 0 values.

I first built a model using all of the available variables. I then removed variables that were duplicates and tested the model to understand how it changed the performance. I also used the xgb.importance function to understand the variable’s influence on the model.

I used an ensemble of XGBoost 11 XGBoost models only updating the random seed for each iteration. This turned out to be more accurate than a single XGBoost model with a large number of iterations and a low eta (learning rate).
