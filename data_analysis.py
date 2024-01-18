from data_extraction import load_data

def analyze_data():
    data = load_data()
    print(data.head)
    print(data.describe())
    print(data.info())

    print(f"\nData Features: {data.columns}") #Lists data features

    print(f"\nEmpty Values: ")
    print(data.isnull().sum()) # Number of null values per feature

    print("\nDuplicated Values: ")
    for column in data.columns:
        print(column, data[column].duplicated().sum()) # Number of duplicated values per feature

    print("\nUnique Values: ")
    for column in data.columns:
        print(column, data[column].nunique()) # Number of unique values per feature

    print("\nColumn DTypes: ")
    print(data.dtypes)
    
    return data

analyze_data()
