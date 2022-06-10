
class Movie:
    def __init__(self, json_obj):
        self.__dict__ = json_obj

    def __str__(self):
        interesting_fields = ['title', 'release_date', 'plot_summary', 'countries', 'genres', 'languages', 'top1t',  'top1%', 'top2t', 'top2%', 'top3t', 'top3%']
        str_representation = {
            'title': 'Movie Title',
            'genres': 'Genres',
            'languages': 'Available Languages',
            'countries': 'Countries',
            'release_date': 'Release Date',
            'plot_summary': 'Plot Summary',
            'top1t': 'Top Similar Movie',
            'top1%': 'Similarity',
            'top2t': 'Second Top Similar Movie,',
            'top2%': 'Similarity',
            'top3t': 'Third Top Similar Movie',
            'top3%': 'Similarity'
        }

        string = ''
        for field in interesting_fields:
            if field not in self.__dict__ or field not in str_representation:
                continue
            value = self.__dict__[field]
            rep = str_representation[field]
            string += f'{rep}: {value}\n' if len(str(value)) != 0 else ''

        return string



