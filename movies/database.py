import pymongo
from movies.movie import Movie


class DataBaseHandler:

    def __init__(self):
        print(f'Mongodb Version: {pymongo.version}')
        MOVIES_DATABASE = 'MoviesDatabase'
        MOVIES_COLLECTION = 'MoviesCollection'
        USER = 'movies_db'
        PASSWORD = '6xZwt0mw1Qf6X9h7'
        CONNECTION_URL = f'mongodb+srv://{USER}:{PASSWORD}@cluster0.ngkki.mongodb.net/{MOVIES_DATABASE}?retryWrites=true&w=majority'

        self.client = pymongo.MongoClient(CONNECTION_URL)
        self.movies_db = self.client[MOVIES_DATABASE]
        self.movies_collection = self.movies_db[MOVIES_COLLECTION]

    def drop_all_collection(self):
        self.movies_collection.drop()

    def push_movie(self, movie_json):
        self.movies_collection.insert_one(movie_json)

    def push_movies(self, movie_json_arr):
        self.movies_collection.insert_many(movie_json_arr)

    def get_movie_object_from_table(self, movie_title: str) -> Movie:
        search_json = {"title": movie_title}
        rv = self.movies_collection.find_one(search_json)
        if rv:
            return Movie(rv)
        return None

    def search_regex_db(self, prefix):
        regex = f'{prefix}.*'
        search_json = {"title": {'$regex': regex}}
        rv = self.movies_collection.find_one(search_json)
        if rv:
            return rv['title']
        return prefix

    def get_top_similar_movies(self, movie_title: str) -> list:
        movie = self.get_movie_object_from_table(movie_title=movie_title)
        top_similar = None if movie is None else \
            [
                self.get_movie_object_from_table(movie_title=movie.top1),
                self.get_movie_object_from_table(movie_title=movie.top2),
                self.get_movie_object_from_table(movie_title=movie.top3),
            ]
        return top_similar


