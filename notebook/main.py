import pandas as pd
df = pd.read_csv("data/European_Bank.csv")
print(df.head())
print(df.shape)
print(df.columns)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nDuplicate Rows:")
print(df.duplicated().sum())
print("\nData Types:")
print(df.dtypes)
print("\nChurn Distribution:")
print(df['Exited'].value_counts())
print("\nChurn Percentage:")
print(df['Exited'].value_counts(normalize=True) * 100)
print("\nActive vs Inactive Customer Churn:")
engagement_churn = pd.crosstab(
    df['IsActiveMember'],
    df['Exited'],
    margins=True
)
print(engagement_churn)
print("\nActive vs Inactive Churn Percentage:")
engagement_percentage = pd.crosstab(
    df['IsActiveMember'],
    df['Exited'],
    normalize='index'
) * 100
print(engagement_percentage)
df['CustomerProfile'] = 'Other'
df.loc[
    (df['IsActiveMember'] == 1) &
    (df['NumOfProducts'] >= 2),
    'CustomerProfile'
] = 'Active Engaged'
df.loc[
    (df['IsActiveMember'] == 0) &
    (df['NumOfProducts'] <= 1),
    'CustomerProfile'
] = 'Inactive Disengaged'
df.loc[
    (df['IsActiveMember'] == 1) &
    (df['NumOfProducts'] == 1),
    'CustomerProfile'
] = 'Active Low Product'
high_balance = df['Balance'].median()
df.loc[
    (df['IsActiveMember'] == 0) &
    (df['Balance'] > high_balance),
    'CustomerProfile'
] = 'Inactive High Balance'
print("\nCustomer Profile Distribution:")
print(df['CustomerProfile'].value_counts())
print("\nChurn by Number of Products:")
product_churn = pd.crosstab(
    df['NumOfProducts'],
    df['Exited']
)
print(product_churn)
print("\nChurn Percentage by Number of Products:")
product_churn_percentage = pd.crosstab(
    df['NumOfProducts'],
    df['Exited'],
    normalize='index'
) * 100
print(product_churn_percentage)
df['ProductCategory'] = df['NumOfProducts'].apply(
    lambda x: 'Single Product' if x == 1 else 'Multi Product'
)
print("\nProduct Category Distribution:")
print(df['ProductCategory'].value_counts())
print("\nAverage Balance by Activity:")

print(
    df.groupby('IsActiveMember')['Balance'].mean()
)
print("\nAverage Salary by Activity:")

print(
    df.groupby('IsActiveMember')['EstimatedSalary'].mean()
)
high_balance = df['Balance'].quantile(0.75)

at_risk_premium = df[
    (df['Balance'] > high_balance) &
    (df['IsActiveMember'] == 0)
]

print("\nAt Risk Premium Customers:")
print(at_risk_premium.shape[0])
salary_balance_mismatch = df[
    (df['EstimatedSalary'] > df['EstimatedSalary'].median()) &
    (df['Balance'] < df['Balance'].median())
]
print("\nSalary Balance Mismatch Customers:")
print(salary_balance_mismatch.shape[0])
active_churn = df[df['IsActiveMember'] == 1]['Exited'].mean() * 100
inactive_churn = df[df['IsActiveMember'] == 0]['Exited'].mean() * 100
print("\nEngagement Retention Ratio")
print(f"Active Customer Churn Rate : {active_churn:.2f}%")
print(f"Inactive Customer Churn Rate : {inactive_churn:.2f}%")
product_depth = df.groupby('NumOfProducts')['Exited'].mean() * 100
print("\nProduct Depth Index")
print(product_depth)
credit_card_score = pd.crosstab(
    df['HasCrCard'],
    df['Exited'],
    normalize='index'
) * 100
print("\nCredit Card Stickiness Score")
print(credit_card_score)
df['RelationshipStrengthIndex'] = (
    df['IsActiveMember'] +
    df['NumOfProducts'] +
    (df['Balance'] > df['Balance'].median()).astype(int)
)
print("\nRelationship Strength Index")
print(df['RelationshipStrengthIndex'].describe())
df.to_csv(
    "data/customer_engagement_analysis.csv",
    index=False
)
print("\nFile Exported Successfully")