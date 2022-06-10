import PySimpleGUI as sg
import json
from tqdm import tqdm
from typing import Dict
from movies.database import DataBaseHandler
from movies.data_managment import DataManager
import traceback

sg.ChangeLookAndFeel('Black')

"""
This is a movie search engine given a database that has been pre-processed with top 3 similar movies for each movie.
"""

PROGRAM_TITLE = 'Movie Search Engine'
MOVIE_TITLE = 'MOVIE_TITLE'
SEARCH = '_SEARCH_'
SEARCH_OUTPUT = '_SEARCH_OUTPUT'
SEARCH_MOVIE = 'Search Movie'
MOVIE_NOT_FOUND = 'The requested movie was not found in the database.'
AUTO_COMPLETE = 'AUTO_COMPLETE'


class Gui:
    """ Create a GUI object """
    def __init__(self):
        self.layout: list = \
            [
                [
                    sg.Image(r"../logo.png")
                ],
                [
                    sg.Text(SEARCH_MOVIE, size=(11, 1)),
                    sg.Input(size=(40, 1), focus=True, key=MOVIE_TITLE),
                ],
                [
                    sg.Button('Search', size=(10, 1), bind_return_key=True, key=SEARCH)
                ],
                [
                    sg.Button('Auto Complete', size=(10, 1), bind_return_key=True, key=AUTO_COMPLETE)
                ],
                [
                    sg.Output(size=(100, 25), key=SEARCH_OUTPUT),
                ]
            ]

        self.window: object = sg.Window(PROGRAM_TITLE, self.layout, element_justification='left')

    def clear_search(self):
        self.window[SEARCH_OUTPUT].Update('')

    def update_auto_completed_text(self, text):
        self.window[MOVIE_TITLE].Update(text)


class SearchEngine:
    """ Create a search engine object """

    def __init__(self, max_cache_size=100):
        # Cache Map: Title, [movie object, number of usages]
        self.cache = dict()
        self.max_cache_size = max_cache_size
        self.db_handler = DataBaseHandler()

    def search_movie(self, values: Dict[str, str]) -> str:
        """ Search for similar movies based on features; """

        movie_title = values[MOVIE_TITLE]
        if movie_title in self.cache:
            self.cache[movie_title] = [self.cache[movie_title][0], self.cache[movie_title][1] + 1]
            return str(self.cache[movie_title][0]) + '\n (Found in cache)'

        # Given already has top similar 3
        movie_obj = self.db_handler.get_movie_object_from_table(movie_title=movie_title)
        if movie_obj:
            if len(self.cache) >= self.max_cache_size:
                self.remove_least_used_movie_from_cache()
            self.cache[movie_title] = [movie_obj, 0]
            return str(movie_obj)

        return MOVIE_NOT_FOUND

    def auto_complete(self, values: Dict[str, str]) -> str:
        movie_title_prefix = values[MOVIE_TITLE]
        auto_completed_text = self.db_handler.search_regex_db(prefix=movie_title_prefix)
        return auto_completed_text

    def remove_least_used_movie_from_cache(self) -> None:
        least_used = None
        min_uses = float('inf')
        for key in self.cache.keys():
            movie, usage_count = self.cache[key]
            if usage_count < min_uses:
                min_uses = usage_count
                least_used = movie.title
        self.cache.pop(least_used)

    def pre_process_data_and_upload_to_db(self, csv_file_path, json_file_path):
        DataManager.process_data_to_json_file(csv_file_path, json_file_path)
        with open(json_file_path, 'r+') as f:
            objects = []
            lines = f.readlines()
            for line in tqdm(lines):
                attributes = json.loads(line)
                objects.append(attributes)

            if self.objects:
                self.db_handler.drop_all_collection()
                self.db_handler.push_movies(movie_json_arr=objects)


def main_loop(search_engine, gui):
    try:
        while True:
            event, values = gui.window.read()

            if event is None:
                break

            if event == SEARCH:
                print(f'Searching...\n')
                results = search_engine.search_movie(values=values)
                gui.clear_search()
                print(results)

            if event == AUTO_COMPLETE:
                auto_completed_text = search_engine.auto_complete(values=values)
                gui.update_auto_completed_text(text=auto_completed_text)

    except Exception as e:
        tb = traceback.format_exc()
        sg.Print(f'An error happened.  Here is the info:', e, tb)
        sg.popup_error(f'AN EXCEPTION OCCURRED!', e, tb)


def main():
    """ The main loop for the program """
    gui = Gui()
    search_engine = SearchEngine()
    main_loop(search_engine=search_engine, gui=gui)


if __name__ == '__main__':
    print('Starting program...')
    main()



