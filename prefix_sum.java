public class PrefixSum {
    public static int[] calculatePrefix(int[] arr) {
        int[] prefix = new int[arr.length];
        prefix[0] = arr[0];
        for(int i=1; i<arr.length; i++) {
            prefix[i] = prefix[i-1] + arr[i];
        }
        return prefix;
    }

    public static int rangeSum(int[] prefix, int l, int r) {
        return prefix[r] - (l > 0 ? prefix[l-1] : 0);
    }
    
    // Driver code
    public static void main(String[] args) {
        int[] arr = {1, 3, 5, 7};
        int[] ps = calculatePrefix(arr);
        System.out.println(Arrays.toString(ps));  // [1, 4, 9, 16]
    }
}
