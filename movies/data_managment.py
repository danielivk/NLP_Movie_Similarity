import re
import nltk
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.pipeline import Pipeline
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


class DataManager:
    @staticmethod
    def remove_empty_values(df):
        df = df.copy()
        df = df[~df.loc[:, 'title'].isna()].reset_index(drop=True)
        return df

    @staticmethod
    def remove_unnecessary_data(df, columns):
        df = df.copy()
        if not isinstance(columns, list):
            columns = [columns]
        for column in columns:
            df[column] = df[column].apply(lambda d: ', '.join([v for v in d.values()]))
        return df

    @staticmethod
    def convert_date_to_year(df, columns):
        df = df.copy()
        if not isinstance(columns, list):
            columns = [columns]
        for column in columns:
            df[column] = df[column].apply(lambda s: s.split('-')[0])
        return df

    @staticmethod
    def normalize(X, stemmer):
      normalized = []
      for x in X:
        words = nltk.word_tokenize(x)
        normalized.append(' '.join([stemmer.stem(word) for word in words if re.match('[a-zA-Z]+', word)]))
      return normalized

    @staticmethod
    def save_in_pickle(path, objects):
        with open(path, "wb") as f:
            pickle.dump(objects, f)

    @staticmethod
    def read_from_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def calculate_similarity(A):
        # base similarity matrix (all dot products)
        # replace this with A.dot(A.T).toarray() for sparse representation
        similarity = np.dot(A, A.T)

        if not isinstance(similarity, np.ndarray):
            similarity = similarity.toarray()

        # squared magnitude of preference vectors (number of occurrences)
        square_mag = np.diag(similarity)

        # inverse squared magnitude
        inv_square_mag = 1 / square_mag

        # if it doesn't occur, set it's inverse magnitude to zero (instead of inf)
        inv_square_mag[np.isinf(inv_square_mag)] = 0

        # inverse of the magnitude
        inv_mag = np.sqrt(inv_square_mag)

        # cosine similarity (elementwise multiply by inverse magnitudes)
        cosine = similarity * inv_mag
        cosine = cosine.T * inv_mag

        return cosine

    @staticmethod
    def find_similar(title, df, sim_matrix, num_top=3):
        index = df[df['title'] == title].index[0]
        vector = sim_matrix[index, :]
        most_similar = df.loc[np.argsort(vector)[::-1][1:num_top+1], 'title']
        return [i for i in np.argsort(vector)[::-1][1:num_top+1]], [i for i in most_similar], [i for i in sim_matrix[index, np.argsort(vector)[::-1][1:num_top+1]]]

    @staticmethod
    def find_similar_for_all(df, sim, num_similars=3):
        df = df.copy()
        indexes_arr = [[] for i in range(num_similars)]
        titles_arr = [[] for i in range(num_similars)]
        pers_arr = [[] for i in range(num_similars)]
        for title in tqdm(df.title):
            indexes, titles, pers = DataManager.find_similar(title, df, sim, num_top=num_similars)
            for i in range(num_similars):
                indexes_arr[i].append(indexes[i])
                titles_arr[i].append(titles[i])
                pers_arr[i].append(pers[i])
        for i in range(num_similars):
            df[f'top{i+1}i'] = indexes_arr[i]
            df[f'top{i+1}t'] = titles_arr[i]
            df[f'top{i+1}%'] = pers_arr[i]
        return df

    @staticmethod
    def save_df_as_json(series, json_file_path):
        with open(json_file_path, 'w') as f:
            for row in series:
                f.write(row + '\n')

    @staticmethod
    def process_data_to_json_file(train_json_path, result_json_path):
        nltk.download('punkt')
        stemmer = SnowballStemmer("english", ignore_stopwords=False)

        movies_df = pd.read_json(train_json_path, lines=True)
        movies_df = DataManager.remove_empty_values(movies_df)
        movies_df_clean = DataManager.remove_unnecessary_data(movies_df, ['languages', 'genres', 'countries'])
        movies_df_clean = DataManager.convert_date_to_year(movies_df_clean, 'release_date')
        movies_df_clean.loc[13942, 'release_date'] = '2010'  # Dataset problem

        plot_pipe = Pipeline([
            ('normalize', FunctionTransformer(DataManager.normalize(stemmer=stemmer), validate=False)),
            ('counter_vectorizer', CountVectorizer(max_df=0.8, min_df=0.2, ngram_range=(1,3))),
            ('tfidf_transform', TfidfTransformer())
        ])
        tfidf_plot_matrix = plot_pipe.fit_transform([x for x in movies_df_clean['plot_summary']])

        genere_pipe = Pipeline([
            ('normalize', FunctionTransformer(DataManager.normalize(stemmer=stemmer), validate=False)),
            ('counter_vectorizer', CountVectorizer(ngram_range=(1,1))),
            ('tfidf_transform', TfidfTransformer())
        ])
        tfidf_genere_matrix = genere_pipe.fit_transform([x for x in movies_df_clean['genres']])

        language_pipe = Pipeline([
            ('normalize', FunctionTransformer(DataManager.normalize(stemmer=stemmer), validate=False)),
            ('counter_vectorizer', CountVectorizer(max_df=len(movies_df_clean)-1, ngram_range=(1,1))),
            ('tfidf_transform', TfidfTransformer())
        ])
        tfidf_language_matrix = language_pipe.fit_transform([x for x in movies_df_clean['languages']])

        plot_file_path = '../data/tfidf_plot.pkl'
        DataManager.save_in_pickle(plot_file_path, tfidf_plot_matrix)

        genere_file_path = '../data/tfidf_genere.pkl'
        DataManager.save_in_pickle(genere_file_path, tfidf_genere_matrix)

        language_file_path = '../data/tfidf_language.pkl'
        DataManager.save_in_pickle(language_file_path, tfidf_language_matrix)

        plot_file_path = '../data/tfidf_plot.pkl'
        tfidf_plot_matrix = DataManager.read_from_pickle(plot_file_path)

        genere_file_path = '../data/tfidf_genere.pkl'
        tfidf_genere_matrix = DataManager.read_from_pickle(genere_file_path)

        language_file_path = '../data/tfidf_language.pkl'
        tfidf_language_matrix = DataManager.read_from_pickle(language_file_path)

        tfidf_full_matrix = np.concatenate([tfidf_plot_matrix.toarray(), tfidf_genere_matrix.toarray(), tfidf_language_matrix.toarray()], axis=1)

        sim = DataManager.calculate_similarity(tfidf_full_matrix)

        movies_df_with_top = DataManager.find_similar_for_all(movies_df_clean, sim)
        DataManager.get_top_n_movies_of_a_movie(movies_df_with_top, 1)

        top_file_path = '../data/top.csv'
        json_output_path = '../data/top.json'

        movies_df_with_top.to_csv(top_file_path, index=False)
        movies_df_with_top = pd.read_csv(top_file_path)
        series = movies_df_with_top.apply(lambda x: x.to_json(), axis=1)
        DataManager.save_df_as_json(series, json_output_path)
