import pandas as pd

def merge_dbs():
    file_names = []
    for file_name in file_names:
        df = pd.read_csv(file_name)
        df = df.drop(columns=['Unnamed: 0'])
        df.drop_duplicates(subset='URL', keep='first', inplace=True)
        df.to_csv('final_db.csv', mode='a', header=False)

def preprocess(cats, words):
    '''
        cats: category names
        words: list of synonyms for the specified categories

    Description:
        Removes any recipes that do not actually involve the cooking
        method specified in cats
    '''
    df = pd.read_csv('final_db.csv')
    df = df.drop(columns='Unnamed: 0')
    df['Method'] = df['Method'].str.lower()
    df['Name'] = df['Name'].str.lower()

    # Create a new dataframe
    new_db = pd.DataFrame(columns=['Name', 'Method','Ingredients', 'Category', 'URL'])

    # Create a smaller copy of df containing only categories mentioned in cats
    short_db = df[df['Category'].str.contains('|'.join(cats), na=False)]

    # Filter out the recipes from short_db that are not about the specified category and store them in new_db
    new_db = short_db[~short_db['Method'].str.contains('|'.join(words), na=False)]
    new_db = new_db[~new_db['Name'].str.contains('|'.join(words), na=False)]

    # Create a csv file for the specific method containing all the recipes to be deleted
    method = new_db[['Index','Name','Method']]

    # Delete the unwanted recipes from the db
    method = method.drop(columns='Unnamed: 0')
    list_ind = method['Index'].tolist()
    for i in list_ind:
        short_db.drop(short_db.loc[short_db['Index'] == i].index, inplace=True)
    short_db.to_csv(f'{cats[0]}.csv')


if __name__=="__main__":
    merge_dbs()
