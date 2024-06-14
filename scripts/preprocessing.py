import pandas as pd
from sklearn.preprocessing import StandardScaler

ranges = {
    "childhood": (0, 12),
    "Teenage": (13, 19),
    "young_adulthood": (20, 35),
    "middle_age": (36, 65),
    "eldery": (66, float("inf")),
}


def age_to_category(age):
    for category, (start, end) in ranges.items():
        if start <= age <= end:
            return category
    return "Unknown"


def preprocess(df):

    # feature engineering

    # make all the columns same format

    df.columns = df.columns.str.lower()

    # create categories of ages
    if "age" in df.columns:

        df["age_category"] = df["age"].apply(age_to_category)

    # standardization

    # category column (unique value > 10)
    cat_columns = [
        column for column in df.columns if df[column].nunique() < 10
    ]

    #convert into categories

    for col in cat_columns:
        df[col] = df[col].astype('category')

    non_cat_columns = [col for col in df.columns if col not in cat_columns]


    #apply standardization on continuous columns 
    standard_scaler = StandardScaler()
    df[non_cat_columns] = standard_scaler.fit_transform(df[non_cat_columns])


    #get dummies 
    cat_features = [col for col in cat_columns if df[col].nunique() > 3]
    df_final = pd.get_dummies(df, columns = cat_features, drop_first = True)

    return df_final 


