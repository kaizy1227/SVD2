#Thêm thư viện cần sử dụng
import numpy as np
import pandas as pd


#Đọc dataset của MovieLens (Link tải 1m data: https://grouplens.org/datasets/movielens/1m/)
data = pd.io.parsers.read_csv('data/ratings.dat', 
    names=['user_id', 'movie_id', 'rating', 'time'],
    engine='python', delimiter='::')
movie_data = pd.io.parsers.read_csv('data/movies.dat',
    names=['movie_id', 'title', 'genre'],
    engine='python', delimiter='::')


#Tạo ma trận xếp hạng (rows: movies, columns: users)
ratings_mat = np.ndarray(
    shape=(np.max(data.movie_id.values), np.max(data.user_id.values)),
    dtype=np.uint8)
ratings_mat[data.movie_id.values-1, data.user_id.values-1] = data.rating.values

#Chuẩn hóa ma trận (trừ giá trị trung bình đi)
normalised_mat = ratings_mat - np.asarray([(np.mean(ratings_mat, 1))]).T

#Hàm tính toán (SVD)
A = normalised_mat.T / np.sqrt(ratings_mat.shape[0] - 1)
U, S, V = np.linalg.svd(A)

#Hàm tính toán độ đồng dạng cosine (sắp xếp theo gần giống nhất và trả về Top N)
def top_cosine_similarity(data, movie_id, top_n=10):
    index = movie_id - 1 # Movie_id bắt đầu từ 1 
    movie_row = data[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    similarity = np.dot(movie_row, data.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

#Hàm in ra Top phim có liên quan nhất
def print_similar_movies(movie_data, movie_id, top_indexes):
    print('Đề xuất 10 phim có liên quan cho phim {0}: \n'.format(
    movie_data[movie_data.movie_id == movie_id].title.values[0]))
    for id in top_indexes + 1:
        print(movie_data[movie_data.movie_id == id].title.values[0])

#Movie_id để tìm phim liên quan, top_n in ra N kết quả    
k = 50
movie_id = 11 # (lấy ra id của phim cần tìm trong movies.dat)
top_n = 10
sliced = V.T[:, :k] # Dữ liệu đại diện
indexes = top_cosine_similarity(sliced, movie_id, top_n)

#In ra Top phim có liên quan 
print_similar_movies(movie_data, movie_id, indexes)
