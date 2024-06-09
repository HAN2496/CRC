def extract_unique_elements(arr):
    # 중복을 제거하고 정렬된 리스트를 만듭니다.
    unique_elements = sorted(set(arr))
    
    # 0부터 최대값 n까지의 숫자만 포함하도록 필터링합니다.
    filtered_elements = [x for x in unique_elements if 0 <= x <= len(unique_elements) - 1]
    
    return filtered_elements

# 예제 배열
arr = [0, 0, 1, 1, 2, 3, 3, 4, 5]

# 함수 호출
result = extract_unique_elements(arr)
print(result)  # [0, 1, 2, 3, 4]
