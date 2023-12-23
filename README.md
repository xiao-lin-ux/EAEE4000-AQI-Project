# EAEE4000-AQI-Project
In this project, we aim to predict the Air Quality Index (AQI) Based on Individual Air Pollutant AQI and conduct causal inference analysis.  

Fine-tuned NN and XGBoost algorithms are implemented as prediction models, with four different datapreprocessing strategies. After hyperparameter tuning, both models achieved low validation losses and high R2 values and can make accurate predictions.  

The Lignum package is used to conduct casual relationship modeling, and DAGs, including the casual relationship graph, the casual prediction graph, and the feature importance graph, are generated. The feature with the greatest causal influence on the predictions is also analyzed.  

### Data source: https://www.kaggle.com/datasets/hasibalmuzdadid/global-air-pollution-dataset/data
