What is a Prefix Sum Array?
A **prefix sum array** (also called a cumulative sum array) is a helper array where each element at index i contains the sum of all elements from the original array's start up to index i.

**Formula:**
prefix[i] = arr + arr[1] + ... + arr[i]

**Key Applications**
1.**Range Sum Queries**: Compute any subarray sum in O(1) time
2.**Equilibrium Index Problems**
3.**Subarray Sum Equals K**  (LeetCode 560)
4.**Product Except Self** (LeetCode 238 - modified prefix product)
5.**Interval Aggregation** in databases/time-series data
