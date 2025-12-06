import pandas as pd


def binarize_multi_threshold(df_raw):
    """
    Binarization with multiple thresholds for numerical features
    Uses 5 page thresholds and 4 decade-based year features
    """
    r = pd.DataFrame()

    r['id'] = df_raw['id']
    r['title'] = df_raw['title']
    r['author'] = df_raw['author']

    # Multiple page thresholds
    r['pages_gt_200'] = df_raw['pages'] > 200
    r['pages_gt_300'] = df_raw['pages'] > 300
    r['pages_gt_400'] = df_raw['pages'] > 400
    r['pages_gt_500'] = df_raw['pages'] > 500
    r['pages_gt_600'] = df_raw['pages'] > 600

    # Decade-based years
    r['year_pre1950'] = (df_raw['year'] < 1950) & (df_raw['year'] > 0)
    r['year_1950_1999'] = (df_raw['year'] >= 1950) & (df_raw['year'] < 2000)
    r['year_2000_2014'] = (df_raw['year'] >= 2000) & (df_raw['year'] < 2015)
    r['year_2015plus'] = df_raw['year'] >= 2015

    # Genres
    r['genre_fiction'] = df_raw.apply(lambda x: _kw(x, ['fiction', 'novel']), axis=1)
    r['genre_nonfiction'] = df_raw.apply(lambda x: _kw(x, ['nonfiction', 'biography']), axis=1)
    r['genre_fantasy'] = df_raw.apply(lambda x: _kw(x, ['fantasy', 'sci-fi']), axis=1)
    r['genre_mystery'] = df_raw.apply(lambda x: _kw(x, ['mystery', 'detective']), axis=1)
    r['genre_children'] = df_raw.apply(lambda x: _kw(x, ['children', 'young adult']), axis=1)

    r['author_russian'] = df_raw['author'].str.lower().str.contains('tolstoy|dostoevsky|pushkin|chekhov', na=False)
    r['is_popular'] = (df_raw['rating'] >= 4.0) | (df_raw['pages'] > 400)
    r['part_of_series'] = df_raw['title'].str.lower().str.contains('book|volume|part|series', na=False)

    r['is_bestseller'] = (
        (df_raw['ratings_count'] >= 500) |
        ((df_raw['year'] <= 1950) & (df_raw['pages'] > 300)) |
        ((df_raw['year'] >= 2010) & r['part_of_series'] & (df_raw['pages'] > 250))
    )

    return r


def _kw(row, keywords):
    text = (str(row.get('categories', '')) + ' ' + str(row.get('description', ''))).lower()
    return any(k in text for k in keywords)


if __name__ == '__main__':
    df_raw = pd.read_csv('books_raw.csv')
    df_v2 = binarize_multi_threshold(df_raw)
    df_v2.to_csv('books_binary_v2.csv', index=False)

    features = [c for c in df_v2.columns if c not in ['id', 'title', 'author']]
    print(f"V2: {len(df_v2)} books, {len(features)} features")