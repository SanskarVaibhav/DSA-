def prefix_sum(arr):
    n = len(arr)
    prefix = [0] * n
    prefix[0] = arr[0]
    for i in range(1, n):
        prefix[i] = prefix[i-1] + arr[i]
    return prefix

def range_sum(prefix, l, r):
    return prefix[r] - (prefix[l-1] if l > 0 else 0)

# Example Usage
arr = [2, 4, 6, 8, 10]
ps = prefix_sum(arr)
print(ps)           # [2, 6, 12, 20, 30]
print(range_sum(ps, 1, 3))  # 18
