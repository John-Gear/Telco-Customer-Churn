# формируем список числовых признаков
num_features = [
    'tenure',
    'MonthlyCharges',
    'TotalCharges',
    'SeniorCitizen'
]

# формируем список бинарных признаков
cat_binary_features = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'PaperlessBilling	'
]

# формируем список многозначных признаков
cat_multiclass_features = [
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingMovies',
    'StreamingTV',
    'Contract',
    'PaymentMethod'
]
