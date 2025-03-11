#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
using namespace std;

// Function prototypes
bool is_prime(int n);
long long next_prime(int n);
vector<int> prime_factors(long long n);
long long gcd(long long a, long long b);
long long lcm(long long a, long long b);
bool is_divisible(int a, int b);
long long modular_exponentiation(long long base, long long exponent, long long mod);
long long divisor_sum(long long n);
bool is_perfect_number(int n);
bool is_armstrong_number(int n);
long long factorial(int n);
unsigned int phi(unsigned int n);
vector<vector<int>> create_matrix(int rows, int cols, int default_value = 0);
vector<vector<int>> add_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);
vector<vector<int>> multiply_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);
vector<vector<int>> transpose_mat(const vector<vector<int>>& mat);
int determinant(const vector<vector<int>>& mat);
vector<vector<int>> subtract_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);
int matrix_trace(const vector<vector<int>>& mat);
double sin(double x);
double cos(double x);
double tan(double x);
double exp(double x);
double log(double x);
double mean(const vector<double>& data);
double median(vector<double> data);
long long combination(int n, int k);
long long permutation(int n, int k);
double binomial_probability(int n, int k, double p);
double poisson_probability(int k, double lambda);
double variance(const vector<double>& data);
double standard_deviation(const vector<double>& data);
double cumulative_binomial_probability(int n, int k, double p);
vector<double> vector_add(const vector<double>& v1, const vector<double>& v2);
vector<double> vector_subtract(const vector<double>& v1, const vector<double>& v2);
vector<double> scalar_multiply(const vector<double>& v, double scalar);
double dot_product(const vector<double>& v1, const vector<double>& v2);
vector<double> cross_product(const vector<double>& v1, const vector<double>& v2);
double vector_magnitude(const vector<double>& v);
vector<double> vector_normalize(const vector<double>& v);
double vector_angle(const vector<double>& v1, const vector<double>& v2);
vector<double> vector_projection(const vector<double>& v1, const vector<double>& v2);
double vector_distance(const vector<double>& v1, const vector<double>& v2);

// Test functions
void test_is_prime() {
    assert(is_prime(2) == true);
    assert(is_prime(3) == true);
    assert(is_prime(4) == false);
    assert(is_prime(29) == true);
    assert(is_prime(30) == false);
    cout << "test_is_prime passed" << endl;
}

void test_next_prime() {
    assert(next_prime(2) == 3);
    assert(next_prime(29) == 31);
    assert(next_prime(30) == 31);
    cout << "test_next_prime passed" << endl;
}

void test_prime_factors() {
    vector<int> factors = prime_factors(56);
    assert(factors == vector<int>({2, 2, 2, 7}));
    factors = prime_factors(29);
    assert(factors == vector<int>({29}));
    cout << "test_prime_factors passed" << endl;
}

void test_gcd() {
    assert(gcd(56, 98) == 14);
    assert(gcd(48, 18) == 6);
    cout << "test_gcd passed" << endl;
}

void test_lcm() {
    assert(lcm(15, 20) == 60);
    assert(lcm(17, 19) == 323);
    cout << "test_lcm passed" << endl;
}

void test_is_divisible() {
    assert(is_divisible(10, 2) == true);
    assert(is_divisible(10, 3) == false);
    cout << "test_is_divisible passed" << endl;
}

void test_modular_exponentiation() {
    assert(modular_exponentiation(2, 10, 1000) == 24);
    assert(modular_exponentiation(3, 3, 5) == 2);
    cout << "test_modular_exponentiation passed" << endl;
}

void test_divisor_sum() {
    assert(divisor_sum(28) == 28);
    assert(divisor_sum(12) == 16);
    cout << "test_divisor_sum passed" << endl;
}

void test_is_perfect_number() {
    assert(is_perfect_number(28) == true);
    assert(is_perfect_number(12) == false);
    cout << "test_is_perfect_number passed" << endl;
}

void test_is_armstrong_number() {
    assert(is_armstrong_number(153) == true);
    assert(is_armstrong_number(123) == false);
    cout << "test_is_armstrong_number passed" << endl;
}

void test_factorial() {
    assert(factorial(5) == 120);
    assert(factorial(0) == 1);
    cout << "test_factorial passed" << endl;
}

void test_phi() {
    assert(phi(9) == 6);
    assert(phi(10) == 4);
    cout << "test_phi passed" << endl;
}

void test_create_matrix() {
    vector<vector<int>> mat = create_matrix(2, 3, 1);
    assert(mat.size() == 2);
    assert(mat[0].size() == 3);
    assert(mat[0][0] == 1);
    cout << "test_create_matrix passed" << endl;
}

void test_add_matrices() {
    vector<vector<int>> mat1 = {{1, 2}, {3, 4}};
    vector<vector<int>> mat2 = {{5, 6}, {7, 8}};
    vector<vector<int>> result = add_matrices(mat1, mat2);
    assert(result[0][0] == 6);
    assert(result[1][1] == 12);
    cout << "test_add_matrices passed" << endl;
}

void test_multiply_matrices() {
    vector<vector<int>> mat1 = {{1, 2}, {3, 4}};
    vector<vector<int>> mat2 = {{2, 0}, {1, 2}};
    vector<vector<int>> result = multiply_matrices(mat1, mat2);
    assert(result[0][0] == 4);
    assert(result[1][1] == 8);
    cout << "test_multiply_matrices passed" << endl;
}

void test_transpose_mat() {
    vector<vector<int>> mat = {{1, 2, 3}, {4, 5, 6}};
    vector<vector<int>> result = transpose_mat(mat);
    assert(result[0][0] == 1);
    assert(result[1][1] == 5);
    cout << "test_transpose_mat passed" << endl;
}

void test_determinant() {
    vector<vector<int>> mat = {{1, 2}, {3, 4}};
    assert(determinant(mat) == -2);
    cout << "test_determinant passed" << endl;
}

void test_subtract_matrices() {
    vector<vector<int>> mat1 = {{1, 2}, {3, 4}};
    vector<vector<int>> mat2 = {{1, 1}, {1, 1}};
    vector<vector<int>> result = subtract_matrices(mat1, mat2);
    assert(result[0][0] == 0);
    assert(result[1][1] == 3);
    cout << "test_subtract_matrices passed" << endl;
}

void test_matrix_trace() {
    vector<vector<int>> mat = {{1, 2}, {3, 4}};
    assert(matrix_trace(mat) == 5);
    cout << "test_matrix_trace passed" << endl;
}

void test_sin() {
    assert(abs(sin(M_PI / 2) - 1.0) < 1e-6);
    assert(abs(sin(0) - 0.0) < 1e-6);
    cout << "test_sin passed" << endl;
}

void test_cos() {
    assert(abs(cos(0) - 1.0) < 1e-6);
    assert(abs(cos(M_PI / 2) - 0.0) < 1e-6);
    cout << "test_cos passed" << endl;
}

void test_tan() {
    assert(abs(tan(M_PI / 4) - 1.0) < 1e-6);
    cout << "test_tan passed" << endl;
}

void test_exp() {
    assert(abs(exp(1) - M_E) < 1e-6);
    assert(abs(exp(0) - 1.0) < 1e-6);
    cout << "test_exp passed" << endl;
}

void test_log() {
    assert(abs(log(M_E) - 1.0) < 1e-6);
    cout << "test_log passed" << endl;
}

void test_mean() {
    vector<double> data = {1, 2, 3, 4, 5};
    assert(mean(data) == 3.0);
    cout << "test_mean passed" << endl;
}

void test_median() {
    vector<double> data = {1, 3, 2};
    assert(median(data) == 2.0);
    data = {1, 3, 2, 4};
    assert(median(data) == 2.5);
    cout << "test_median passed" << endl;
}

void test_combination() {
    assert(combination(5, 2) == 10);
    assert(combination(10, 3) == 120);
    cout << "test_combination passed" << endl;
}

void test_permutation() {
    assert(permutation(5, 2) == 20);
    assert(permutation(10, 3) == 720);
    cout << "test_permutation passed" << endl;
}

void test_binomial_probability() {
    assert(abs(binomial_probability(5, 2, 0.5) - 0.3125) < 1e-6);
    cout << "test_binomial_probability passed" << endl;
}

void test_poisson_probability() {
    assert(abs(poisson_probability(2, 1) - 0.18394) < 1e-5);
    cout << "test_poisson_probability passed" << endl;
}

void test_variance() {
    vector<double> data = {1, 2, 3, 4, 5};
    assert(abs(variance(data) - 2.0) < 1e-6);
    cout << "test_variance passed" << endl;
}

void test_standard_deviation() {
    vector<double> data = {1, 2, 3, 4, 5};
    assert(abs(standard_deviation(data) - sqrt(2.0)) < 1e-6);
    cout << "test_standard_deviation passed" << endl;
}

void test_cumulative_binomial_probability() {
    assert(abs(cumulative_binomial_probability(5, 2, 0.5) - 0.5) < 1e-6);
    cout << "test_cumulative_binomial_probability passed" << endl;
}

void test_vector_add() {
    vector<double> v1 = {1, 2, 3};
    vector<double> v2 = {4, 5, 6};
    vector<double> result = vector_add(v1, v2);
    assert(result == vector<double>({5, 7, 9}));
    cout << "test_vector_add passed" << endl;
}

void test_vector_subtract() {
    vector<double> v1 = {1, 2, 3};
    vector<double> v2 = {4, 5, 6};
    vector<double> result = vector_subtract(v1, v2);
    assert(result == vector<double>({-3, -3, -3}));
    cout << "test_vector_subtract passed" << endl;
}

void test_scalar_multiply() {
    vector<double> v = {1, 2, 3};
    vector<double> result = scalar_multiply(v, 2.0);
    assert(result == vector<double>({2, 4, 6}));
    cout << "test_scalar_multiply passed" << endl;
}

void test_dot_product() {
    vector<double> v1 = {1, 2, 3};
    vector<double> v2 = {4, 5, 6};
    assert(dot_product(v1, v2) == 32);
    cout << "test_dot_product passed" << endl;
}

void test_cross_product() {
    vector<double> v1 = {1, 2, 3};
    vector<double> v2 = {4, 5, 6};
    vector<double> result = cross_product(v1, v2);
    assert(result == vector<double>({-3, 6, -3}));
    cout << "test_cross_product passed" << endl;
}

void test_vector_magnitude() {
    vector<double> v = {3, 4};
    assert(vector_magnitude(v) == 5);
    cout << "test_vector_magnitude passed" << endl;
}

void test_vector_normalize() {
    vector<double> v = {3, 4};
    vector<double> result = vector_normalize(v);
    assert(abs(vector_magnitude(result) - 1.0) < 1e-6);
    cout << "test_vector_normalize passed" << endl;
}

void test_vector_angle() {
    vector<double> v1 = {1, 0};
    vector<double> v2 = {0, 1};
    assert(abs(vector_angle(v1, v2) - M_PI / 2) < 1e-6);
    cout << "test_vector_angle passed" << endl;
}

void test_vector_projection() {
    vector<double> v1 = {1, 2};
    vector<double> v2 = {2, 0};
    vector<double> result = vector_projection(v1, v2);
    assert(result == vector<double>({1, 0}));
    cout << "test_vector_projection passed" << endl;
}

void test_vector_distance() {
    vector<double> v1 = {1, 2};
    vector<double> v2 = {4, 6};
    assert(abs(vector_distance(v1, v2) - 5.0) < 1e-6);
    cout << "test_vector_distance passed" << endl;
}

int main() {
    test_is_prime();
    test_next_prime();
    test_prime_factors();
    test_gcd();
    test_lcm();
    test_is_divisible();
    test_modular_exponentiation();
    test_divisor_sum();
    test_is_perfect_number();
    test_is_armstrong_number();
    test_factorial();
    test_phi();
    test_create_matrix();
    test_add_matrices();
    test_multiply_matrices();
    test_transpose_mat();
    test_determinant();
    test_subtract_matrices();
    test_matrix_trace();
    test_sin();
    test_cos();
    test_tan();
    test_exp();
    test_log();
    test_mean();
    test_median();
    test_combination();
    test_permutation();
    test_binomial_probability();
    test_poisson_probability();
    test_variance();
    test_standard_deviation();
    test_cumulative_binomial_probability();
    test_vector_add();
    test_vector_subtract();
    test_scalar_multiply();
    test_dot_product();
    test_cross_product();
    test_vector_magnitude();
    test_vector_normalize();
    test_vector_angle();
    test_vector_projection();
    test_vector_distance();

    cout << "All tests passed!" << endl;
    return 0;
}
