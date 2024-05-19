import numpy as np

def find_top_n_elements_with_indexes(arr, n):
    if n > len(arr):
        raise ValueError("n should not be greater than the length of the array")

    # 배열의 원소와 인덱스를 튜플로 만들고, 값에 따라 정렬
    indexed_arr = sorted(((value, index) for index, value in enumerate(arr)), reverse=True, key=lambda x: x[0])
    
    # 가장 큰 n개의 원소와 인덱스 추출
    top_n_elements = indexed_arr[:n]
    
    # 값과 인덱스 분리
    values = [value for value, index in top_n_elements]
    indexes = [index for value, index in top_n_elements]
    
    return values, indexes

if __name__ == "__main__":
    # 예제 배열과 n 값
    array = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
    n = 4

    # 함수 실행 및 결과 출력
    top_values, top_indexes = find_top_n_elements_with_indexes(array, n)
    print("Top n values:", top_values)
    print("Indexes of top n values:", top_indexes)
