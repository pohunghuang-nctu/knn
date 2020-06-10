#!/usr/bin/python3
import pandas as pd
import sys
import os
import json
import matplotlib.pyplot as plt
from texttable import Texttable


def toDf(item_path, dropCols=None):
    pf_list = []
    counter = 0
    for chunk in chunks(item_path):
        if dropCols is not None:
            chunk = chunk.drop(labels=dropCols, axis='columns')
        pf_list.append(chunk)
        counter += len(chunk.index)
        print('read %d of %s' % (counter, item_path))
    return pd.concat(pf_list)


def chunks(item_path):
    chunkSize = 1000000
    return pd.read_csv(item_path, header=0, chunksize=chunkSize)


def min_watch_of_topN(dataset_path, ratings):
    ur = ratings.pivot_table(index=['userId'], values=['movieId'], aggfunc='count').rename(columns={'movieId': 'movies'})
    ur = ur.sort_values(('movies'), ascending=False)
    total_users = len(ur.index)   
    percentage_list = []
    min_watched = []
    for i in range(1, 101):
        percent_i_idx = int(round((i * total_users)/100))-1
        if percent_i_idx < 0:
            percent_i_idx = 0
        if percent_i_idx > (total_users-1):
            percent_i_idx = total_users-1
        percent_i_record = ur.iloc[percent_i_idx]
        # print('index percent %d is %d' % (i, percent_i_idx))
        percentage_list.append(i)
        min_watched.append(percent_i_record['movies'].item())
    top_watchs = pd.DataFrame({'top N % of user': percentage_list, 'Minimal watched': min_watched})
    # print(top_watchs)
    top_watchs.plot(x='top N % of user', y='Minimal watched', title='Minimal movies watched / top N percent users', grid=True)
    plt.show()


def min_watched_of_topN(dataset_path, ratings):
    ur = ratings.pivot_table(index=['movieId'], values=['userId'], aggfunc='count').rename(columns={'userId': 'users'})
    ur = ur.sort_values(('users'), ascending=False)
    total_users = len(ur.index)   
    percentage_list = []
    min_watched = []
    for i in range(1, 101):
        if (i % 5) != 0 or (i < 5):
            continue
        percent_i_idx = int(round((i * total_users)/100))-1
        if percent_i_idx < 0:
            percent_i_idx = 0
        if percent_i_idx > (total_users-1):
            percent_i_idx = total_users-1
        percent_i_record = ur.iloc[percent_i_idx]
        # print('index percent %d is %d' % (i, percent_i_idx))
        percentage_list.append(i)
        min_watched.append(percent_i_record['users'].item())
    top_watchs = pd.DataFrame({'top N % of movies': percentage_list, 'Minimal watchers': min_watched})
    print(top_watchs.head())
    # print(top_watchs)
    top_watchs.plot(x='top N % of movies', y='Minimal watchers', title='Minimal watchers / top N percent movies', grid=True)
    plt.show()


def group_by_hour(dataset_path, ratings):
    rating_times = pd.to_datetime(ratings['timestamp'], unit='s')
    print(rating_times.min())
    print(rating_times.max())
    ratings['rating_hour'] = rating_times.dt.hour
    ratings['rating_time'] = rating_times
    start = pd.to_datetime('2000/01/01', format='%Y/%m/%d')
    end = pd.to_datetime('2018/01/01', format='%Y/%m/%d')
    period = (end - start).days
    print('period = %d days' % period)
    filtered = ratings[(ratings['rating_time'] >= start) & (ratings['rating_time'] < end)]
    ur = filtered.pivot_table(index=['rating_hour'], values=['userId'], aggfunc='count').rename(columns={'userId': 'watch_times'})
    ur['average watch in hour'] = ur['watch_times']/period
    ur.plot(y='average watch in hour', title='Watched movies time-distributed', grid=True)
    plt.show()
    # print(ur.head(n=24))
    # times = pd.DatetimeIndex(ratings.timestamp)
    # grp_by_hr = ratings.groupby([times.hour])
    # print(grp_by_hr.head())


def top_100_rated_movies(dataset_path):
    ratings = toDf(os.path.join(dataset_path, 'ratings.csv'), dropCols=['timestamp', 'userId'])
    movies = toDf(os.path.join(dataset_path, 'movies.csv'), dropCols=['genres'])
    average_rating = ratings.groupby(by='movieId').mean()
    rating_count = ratings.groupby(by='movieId').count().rename(columns={'rating': 'rating_count'})
    movie_stat = average_rating.merge(rating_count, left_index=True, right_index=True).sort_values(by='rating_count', ascending=False).reset_index()
    movie_stat = movie_stat.merge(movies, left_on='movieId', right_on='movieId')
    total_movies_rated = len(movie_stat.index)
    print(total_movies_rated)
    # print(average_rating.head())
    # print(rating_count.head())
    theMiddle = int(total_movies_rated * 0.2)
    middle100 = movie_stat.iloc[theMiddle-100:theMiddle].drop(columns=['movieId'])
    print('From %d to %d, the mean rate is %f' % (theMiddle-100, theMiddle, middle100['rating'].mean()))
    middle100.plot(y=['rating'], title='Rating of Not so hot movies', grid=True)
    plt.show()
    hot100 = movie_stat.iloc[0:100].drop(columns=['movieId'])
    print('From 0 to 99, the mean rate is %f' % hot100['rating'].mean())
    # print(movie_stat.head())
    tb = Texttable()
    tb.set_cols_align(['l', 'r', 'r'])
    tb.set_cols_dtype(['f', 'i', 't'])
    tb.header(hot100.columns.get_values())
    tb.add_rows(hot100.values, header=False)
    with open('hot_100.txt', 'w') as ofile:
        ofile.write(tb.draw().encode('utf8'))
    hot100.plot(y=['rating'], title='Rating of Hot 100 movies', grid=True)
    hot_sucks = hot100[hot100['rating'] < 3.2]
    tb.reset()
    tb.header(hot_sucks.columns.get_values())
    tb.add_rows(hot_sucks.values, header=False)
    with open('hot_sucks.txt', 'w') as ofile:
        ofile.write(tb.draw().encode('utf8'))
    # print(hot100.mean())
    # print(cool100.mean())
    plt.show()


def test(dataset_path):
    tags = toDf(os.path.join(dataset_path, 'tags.csv'), dropCols=['timestamp', 'userId'])
    toy_story_tags = tags[tags['movieId'] == 1]
    print(toy_story_tags.head())


def main():
    dataset_path = sys.argv[1]
    test(dataset_path)
    # ratings = toDf(os.path.join(dataset_path, 'ratings.csv'), dropCols=['rating'])
    # min_watch_of_topN(dataset_path, ratings)
    # min_watched_of_topN(dataset_path, ratings)
    # group_by_hour(dataset_path, ratings)
    # top_100_rated_movies(dataset_path)



if __name__ == '__main__':
    main()

