#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

bool is_prime(int n) {
    if (n < 2) {
        return false;
    }
    int limit = sqrt(n);
    for(int i = 2; i <= limit; i++) {
        if (n % i == 0) {
            return false;
        }
    }
    return true;
}

long long next_prime(int n) {
    if (n < 2) {
        return 2;
    }
    long long candidate = n + 1;
    if (candidate % 2 == 0) {
        candidate++;
    }
    while (true) {
        if (is_prime(candidate)) {
            return candidate;
        }
        candidate += 2;
    }
}

vector<int> prime_factors(long long n) {
    vector<int> prime_factors;
    while (n % 2 == 0) {
        prime_factors.push_back(2);
        n = n / 2;
    }
    for (long long i = 3; i <= sqrt(n); i += 2) {
        while (n % i == 0) {
            prime_factors.push_back(i);
            n = n / i;
        }
    }
    if (n > 2) {
        prime_factors.push_back(n);
    }
    return prime_factors;
}

long long gcd(long long a, long long b) {
    if (a == 0) return b;
    if (b == 0) return a;
    while (a != b) {
        if (a > b) {
            a = a - b;
        }
        else {
            b = b - a;
        }
    }
    return a;
}

long long lcm(long long a, long long b) {
    return a * b / gcd(a, b);
}

bool is_divisible(int a, int b) {
    return a % b == 0;
}

long long modular_exponentiation(long long base, long long exponent, long long mod) {
    long long result = 1;
    base = base % mod;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result = (result * base) % mod;
        }
        base = (base * base) % mod;
        exponent /= 2;
    }
    return result;
}

long long divisor_sum(long long n) {
    long long sum = 0;
    for (long long i = 1; i * i <= n; i++) {
        if (n % i == 0) {
            sum += i;
            if (i != n / i && i != 1) {
                sum += n / i;
            }
        }
    }
    return sum;
}

bool is_perfect_number(int n) {
    return divisor_sum(n) == n;
}

bool is_armstrong_number(int n) {
    if (n < 0) return false;

    int original = n;
    int sum = 0;
    int size = to_string(n).length();

    while (n != 0) {
        int digit = n % 10;
        sum += pow(digit, size);
        n /= 10;
    }

    return sum == original;
}

long long factorial(int n) {
    if (n < 0) {
        throw invalid_argument("Factorial is not defined for negative numbers.");
    }
    long long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

unsigned int phi(unsigned int n) {
    unsigned int result = n;
    for (unsigned int p = 2; p * p <= n; p++) {
        if (n % p == 0) {
            while (n % p == 0) {
                n /= p;
            }
            result -= result / p;
        }
    }
    if (n > 1) {
        result -= result / n;
    }
    return result;
}

// MATRICES

vector<vector<int>> create_matrix(int rows, int cols, int default_value = 0) {
    return vector<vector<int>>(rows, vector<int>(cols, default_value));
}

vector<vector<int>> add_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    if (mat1.empty() || mat2.empty() || mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        throw invalid_argument("Matrices must have the same nonzero dimensions.");
    }
    int rows = mat1.size();
    int cols = mat1[0].size();
    vector<vector<int>> result(rows, vector<int>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i][j] = mat1[i][j] + mat2[i][j];
        }
    }
    return result;
}

vector<vector<int>> multiply_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    int rows_A = mat1.size();
    int cols_A = mat1[0].size();
    int cols_B = mat2[0].size();
    vector<vector<int>> result(rows_A, vector<int>(cols_B, 0));
    for (int i = 0; i < rows_A; i++) {
        for (int j = 0; j < cols_B; j++) {
            for (int k = 0; k < cols_A; k++) {
                result[i][j] += mat1[i][k] * mat2[k][j];
            }
        }
    }
    return result;
}

vector<vector<int>> transpose_mat(const vector<vector<int>>& mat) {
    int rows = mat.size();
    int cols = mat[0].size();
    vector<vector<int>> result(cols, vector<int>(rows));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = mat[i][j];
        }
    }
    return result;
}

int determinant(const vector<vector<int>>& mat) {
    int n = mat.size();
    if (n == 1) {
        return mat[0][0];
    }
    if (n == 2) {
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0];
    }
    int det = 0;
    for (int col = 0; col < n; col++) {
        vector<vector<int>> submatrix(n - 1, vector<int>(n - 1));
        for (int i = 1; i < n; i++) {
            int subcol = 0;
            for (int j = 0; j < n; j++) {
                if (j == col) continue;
                submatrix[i - 1][subcol] = mat[i][j];
                subcol++;
            }
        }
        int cofactor = (col % 2 == 0 ? 1 : -1) * mat[0][col] * determinant(submatrix);
        det += cofactor;
    }
    return det;
}

vector<vector<int>> subtract_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2) {
    if (mat1.size() != mat2.size() || mat1[0].size() != mat2[0].size()) {
        throw invalid_argument("Matrices must have the same dimensions for subtraction.");
    }
    int rows = mat1.size();
    int cols = mat1[0].size();
    vector<vector<int>> result(rows, vector<int>(cols, 0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = mat1[i][j] - mat2[i][j];
        }
    }
    return result;
}

int matrix_trace(const vector<vector<int>>& mat) {
    int n = mat.size();
    if (n == 0 || mat[0].size() != n) {
        throw invalid_argument("Matrix must be square for trace calculation.");
    }
    int trace = 0;
    for (int i = 0; i < n; ++i) {
        trace += mat[i][i];
    }
    return trace;
}

//Advanced math functions
double sin(double x) {
    double result = 0.0;
    double term = x;
    int n = 1;
    while (x > M_PI) x -= 2 * M_PI;
    while (x < -M_PI) x += 2 * M_PI;
    for (int i = 1; i <= 10; ++i) {
        result += term;
        term *= -x * x / ((2 * n) * (2 * n + 1));
        ++n;
    }
    return result;
}

double cos(double x) {
    double result = 0.0;
    double term = 1.0;
    int n = 1;
    while (x > M_PI) x -= 2 * M_PI;
    while (x < -M_PI) x += 2 * M_PI;
    // Taylor series approximation
    for (int i = 1; i <= 10; ++i) {
        result += term;
        term *= -x * x / ((2 * n - 1) * (2 * n));
        ++n;
    }
    return result;
}

double tan(double x) {
    double cos_x = cos(x);
    if (cos_x == 0.0) {
        throw runtime_error("Tangent is undefined for this value.");
    }
    return sin(x) / cos_x;
}

double exp(double x) {
    double result = 1.0;
    double term = 1.0;
    for (int i = 1; i <= 20; ++i) {
        term *= x / i;
        result += term;
    }
    return result;
}

double log(double x) {
    if (x <= 0.0) {
        throw runtime_error("Logarithm is undefined for non-positive values.");
    }
    double y = 1.0;
    for (int i = 0; i < 20; ++i) {
        y = y - (exp(y) - x) / exp(y);
    }
    return y;
}

// Statistics and Probability
double mean(const vector<double>& data) {
    if (data.empty()) {
        throw invalid_argument("Data cannot be empty.");
    }
    double sum = 0.0;
    for (double x : data) {
        sum += x;
    }
    return sum / data.size();
}

double median(vector<double> data) {
    if (data.empty()) {
        throw invalid_argument("Data cannot be empty.");
    }
    sort(data.begin(), data.end());
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    } else {
        return data[n / 2];
    }
}

long long combination(int n, int k) {
    if (n < k || n < 0 || k < 0) {
        throw invalid_argument("Invalid input parameters.");
    }
    if (k > n - k) {
        k = n - k;
    }
    long long result = 1;
    for (int i = 1; i <= k; ++i) {
        result = result * (n - k + i) / i;
    }
    return result;
}

long long permutation(int n, int k) {
    if (n < k || n < 0 || k < 0) {
        throw invalid_argument("Invalid input parameters.");
    }
    long long result = 1;
    for (int i = 0; i < k; ++i) {
        result *= (n - i);
    }
    return result;
}

double binomial_probability(int n, int k, double p) {
    if (n < k || n < 0 || k < 0 || p < 0 || p > 1) {
        throw invalid_argument("Invalid input parameters.");
    }
    return combination(n, k) * pow(p, k) * pow(1 - p, n - k);
}

double poisson_probability(int k, double lambda) {
    if (k < 0 || lambda <= 0) {
        throw invalid_argument("Invalid input parameters.");
    }
    return exp(-lambda) * pow(lambda, k) / factorial(k);
}

double variance(const vector<double>& data) {
    if (data.empty()) {
        throw invalid_argument("Data cannot be empty.");
    }
    double mu = mean(data);
    double sum_sq = 0.0;
    for (double x : data) {
        sum_sq += (x - mu) * (x - mu);
    }
    return sum_sq / data.size();
}

double standard_deviation(const vector<double>& data) {
    return sqrt(variance(data));
}

double cumulative_binomial_probability(int n, int k, double p) {
    if (n < k || n < 0 || k < 0 || p < 0 || p > 1) {
        throw invalid_argument("Invalid input parameters.");
    }
    double cumulative_prob = 0.0;
    for (int i = 0; i <= k; ++i) {
        cumulative_prob += binomial_probability(n, i, p);
    }
    return cumulative_prob;
}

//Vector operations
vector<double> vector_add(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw invalid_argument("Vectors must have the same dimension.");
    }
    vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

vector<double> vector_subtract(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw invalid_argument("Vectors must have the same dimension.");
    }
    vector<double> result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

vector<double> scalar_multiply(const vector<double>& v, double scalar) {
    vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] * scalar;
    }
    return result;
}

double dot_product(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw invalid_argument("Vectors must have the same dimension.");
    }
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

vector<double> cross_product(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != 3 || v2.size() != 3) {
        throw invalid_argument("Cross product is only defined for 3D vectors.");
    }
    return {
        v1[1] * v2[2] - v1[2] * v2[1], // x-component
        v1[2] * v2[0] - v1[0] * v2[2], // y-component
        v1[0] * v2[1] - v1[1] * v2[0]  // z-component
    };
}

double vector_magnitude(const vector<double>& v) {
    double sum_sq = 0.0;
    for (double x : v) {
        sum_sq += x * x;
    }
    return sqrt(sum_sq);
}

vector<double> vector_normalize(const vector<double>& v) {
    double mag = vector_magnitude(v);
    if (mag == 0.0) {
        throw invalid_argument("Cannot normalize a zero vector.");
    }
    return scalar_multiply(v, 1.0 / mag);
}

double vector_angle(const vector<double>& v1, const vector<double>& v2) {
    double dot = dot_product(v1, v2);
    double mag1 = vector_magnitude(v1);
    double mag2 = vector_magnitude(v2);
    if (mag1 == 0.0 || mag2 == 0.0) {
        throw invalid_argument("Vectors must have non-zero magnitude.");
    }
    return acos(dot / (mag1 * mag2));
}

vector<double> vector_projection(const vector<double>& v1, const vector<double>& v2) {
    double dot = dot_product(v1, v2);
    double mag2_sq = dot_product(v2, v2);
    if (mag2_sq == 0.0) {
        throw invalid_argument("Cannot project onto a zero vector.");
    }
    return scalar_multiply(v2, dot / mag2_sq);
}

double vector_distance(const vector<double>& v1, const vector<double>& v2) {
    if (v1.size() != v2.size()) {
        throw invalid_argument("Vectors must have the same dimension.");
    }
    double sum_sq = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        sum_sq += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(sum_sq);
}