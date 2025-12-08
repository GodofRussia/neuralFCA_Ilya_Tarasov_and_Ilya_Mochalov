"""
Books dataset: collection and binarization
Collects book data from Google Books API and binarizes features for FCA
"""

import requests
import pandas as pd
import time
import re


class BookCollector:
    """Collects raw book data from Google Books API"""

    def __init__(self):
        self.api = "https://www.googleapis.com/books/v1/volumes"

    def search(self, query, max_results=40):
        """Search books by query"""
        try:
            r = requests.get(self.api, params={'q': query, 'maxResults': max_results}, timeout=10)
            if r.status_code == 200:
                return [self._parse(item) for item in r.json().get('items', []) if self._parse(item)]
        except:
            pass
        return []

    def _parse(self, item):
        """Parse book info from API response"""
        try:
            v = item.get('volumeInfo', {})
            if not v.get('title') or not v.get('authors'):
                return None

            year_match = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', v.get('publishedDate', ''))

            return {
                'title': v['title'],
                'author': v['authors'][0],
                'year': int(year_match.group(1)) if year_match else 0,
                'pages': v.get('pageCount', 0),
                'categories': ', '.join(v.get('categories', [])),
                'description': v.get('description', ''),
                'rating': v.get('averageRating', 0),
                'ratings_count': v.get('ratingsCount', 0),
            }
        except:
            return None

    def collect(self, target=600):
        """Collect diverse dataset of books"""
        queries = [
            ('tolstoy', 15), ('dostoevsky', 15), ('shakespeare', 15), ('dickens', 15),
            ('jane austen', 15), ('mark twain', 20), ('hemingway', 15), ('orwell', 15),
            ('harry potter', 15), ('game of thrones', 15), ('lord of the rings', 15),
            ('stephen king', 25), ('dan brown', 20), ('james patterson', 20),
            ('agatha christie', 20), ('tolkien', 15), ('rowling', 15),
            ('fantasy bestseller', 40), ('science fiction bestseller', 40),
            ('mystery bestseller', 40), ('thriller bestseller', 40),
            ('romance bestseller', 35), ('horror bestseller', 35),
            ('children bestseller', 40), ('young adult bestseller', 40),
            ('biography bestseller', 30), ('history bestseller', 30),
            ('philosophy', 25), ('psychology bestseller', 25),
        ]

        books = []
        seen = set()

        for q, lim in queries:
            if len(books) >= target:
                break

            for book in self.search(q, lim):
                key = f"{book['title'].lower()}|{book['author'].lower()}"
                if key not in seen and book['pages'] > 0:
                    seen.add(key)
                    books.append(book)

            time.sleep(1)

        df = pd.DataFrame(books)
        df.insert(0, 'id', range(1, len(df) + 1))
        return df


class BookBinarizer:
    """Binarizes book features for Formal Concept Analysis"""

    def binarize(self, df):
        """Convert raw features to binary"""
        r = pd.DataFrame()

        r['id'] = df['id']
        r['title'] = df['title']
        r['author'] = df['author']

        r['pages_gt_300'] = df['pages'] > 300
        r['pages_gt_500'] = df['pages'] > 500

        r['year_recent'] = df['year'] >= 2015
        r['year_classic'] = (df['year'] <= 1950) & (df['year'] > 0)

        r['genre_fiction'] = df.apply(lambda x: self._kw(x, ['fiction', 'novel']), axis=1)
        r['genre_nonfiction'] = df.apply(lambda x: self._kw(x, ['nonfiction', 'biography']), axis=1)
        r['genre_fantasy'] = df.apply(lambda x: self._kw(x, ['fantasy', 'sci-fi']), axis=1)
        r['genre_mystery'] = df.apply(lambda x: self._kw(x, ['mystery', 'detective']), axis=1)
        r['genre_children'] = df.apply(lambda x: self._kw(x, ['children', 'young adult']), axis=1)

        r['author_russian'] = df['author'].str.lower().str.contains('tolstoy|dostoevsky|pushkin|chekhov', na=False)
        r['is_popular'] = (df['rating'] >= 4.0) | (df['pages'] > 400)
        r['part_of_series'] = df['title'].str.lower().str.contains('book|volume|part|series', na=False)

        r['is_bestseller'] = (
            (df['ratings_count'] >= 500) |
            ((df['year'] <= 1950) & (df['pages'] > 300)) |
            ((df['year'] >= 2010) & r['part_of_series'] & (df['pages'] > 250))
        )

        return r

    def _kw(self, row, keywords):
        """Check if keywords present in text"""
        text = (str(row.get('categories', '')) + ' ' + str(row.get('description', ''))).lower()
        return any(k in text for k in keywords)


if __name__ == '__main__':
    collector = BookCollector()
    df_raw = collector.collect(target=600)
    df_raw.to_csv('books_raw.csv', index=False)
    print(f"Raw: {len(df_raw)} books")

    binarizer = BookBinarizer()
    df_bin = binarizer.binarize(df_raw)
    df_bin.to_csv('books_binary.csv', index=False)
    print(f"Binary: {len(df_bin)} books, {df_bin['is_bestseller'].sum()} bestsellers")