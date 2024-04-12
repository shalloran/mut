import pandas as pd
from urllib.parse import urlparse
import re
from sklearn.preprocessing import LabelEncoder
import joblib

def process_url_data(csv_file_path, chunk_size=1000):
    def lexicalFE(url):
        if not urlparse(url).scheme:
            url = 'http://' + url
        try:
            parsedURL = urlparse(url)
            domain = parsedURL.netloc
            path = parsedURL.path
            query = parsedURL.query
            domainNoPrefix = domain.replace('www.', '')
            features = {
                'domain': domain,
                'domainNoPrefix': domainNoPrefix,
                'domainLength': len(domain),
                'pathLength': len(path),
                'queryLength': len(query),
                'numPathComponents': len(path.split('/')) - 1,
                'numQueryComponents': len(query.split('&')) if query else 0,
                'hasDigitsInDomain': any(char.isdigit() for char in domain),
                'hasDigitsInPath': any(char.isdigit() for char in path),
                'hasDigitsInQuery': any(char.isdigit() for char in query)
            }
            return features
        except ValueError as e:
            print(f"Error processing URL {url}: {e}")
            return {}

    def descriptiveFE(url):
        parsedURL = urlparse(url)
        domain = parsedURL.netloc.replace('www.', '')
        path = parsedURL.path
        query = parsedURL.query
        path_components = path.split('/')
        filename = path_components[-1] if '.' in path_components[-1] else None
        fileBool = 1 if filename else 0
        file_extension = filename.split('.')[-1] if filename else None
        features = {
            'domainLength': len(domain),
            'pathLength': len(path),
            'queryLength': len(query),
            'numPathComponents': len(path_components),
            'filename': filename,
            'fileNamePresent': fileBool,
            'fileExtension': file_extension,
            'isIpAddress': bool(re.match(r'^\d{1,3}(\.\d{1,3}){3}$', domain)),
            'fileExecutable': file_extension in ['exe', 'bin', 'bat']
        }
        return features

    df = pd.read_csv(csv_file_path)
    df['Lexical_Features'] = df['URL'].apply(lexicalFE)
    df['Descriptive_Features'] = df['URL'].apply(descriptiveFE)

    df_final = pd.DataFrame()

    for start in range(0, df.shape[0], chunk_size):
        end = min(start + chunk_size, df.shape[0])
        df_chunk = df.iloc[start:end].copy()
        df_chunk.reset_index(drop=True, inplace=True)
        lexFeatsDF = pd.json_normalize(df_chunk['Lexical_Features'])
        lexFeatsDF.columns = ['Lexical_' + str(col) for col in lexFeatsDF.columns]
        descFeatsDF = pd.json_normalize(df_chunk['Descriptive_Features'])
        descFeatsDF.columns = ['Descriptive_' + str(col) for col in descFeatsDF.columns]
        df_chunk = pd.concat([df_chunk, lexFeatsDF, descFeatsDF], axis=1)
        df_final = pd.concat([df_final, df_chunk], axis=0, ignore_index=True)

    df_final.drop(['Lexical_Features', 'Descriptive_Features', 'URL'], axis=1, inplace=True)
    df_final_columns = df_final.columns.tolist()
    with open('model_columns.txt', 'w') as f:
        f.write('\n'.join(df_final_columns))
    df_final = df_final.dropna()

    catCols = df_final.select_dtypes(include=['object', 'category']).columns
    for col in catCols:
        if col == 'Classification':
            continue
        le = LabelEncoder()
        df_final[col] = le.fit_transform(df_final[col])
        joblib.dump(le, f'./models-checkpoints/{col}_encoder.joblib')

    return df_final