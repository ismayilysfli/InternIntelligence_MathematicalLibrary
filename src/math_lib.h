#ifndef MATHLIB_H
#define MATHLIB_H
#include <vector>
#include <cmath>
using namespace std;

namespace MathLib {

    // Number Theory Functions

    // Checks if a number is prime by testing divisibility up to the square root of the number.
    bool is_prime(int n);

    // Finds the next prime number greater than the given number by checking odd candidates.
    long long next_prime(int n);

    // Returns the prime factors of a number by dividing it by 2 and odd numbers up to its square root.
    vector<int> prime_factors(long long n);

    // Computes the greatest common divisor (GCD) of two numbers using the Euclidean algorithm.
    long long gcd(long long a, long long b);

    // Computes the least common multiple (LCM) of two numbers using the GCD.
    long long lcm(long long a, long long b);

    // Checks if one number is divisible by another by testing the remainder of division.
    bool is_divisible(int a, int b);

    // Computes (base^exponent) % mod efficiently using the fast exponentiation method.
    long long modular_exponentiation(long long base, long long exponent, long long mod);

    // Calculates the sum of all divisors of a number by checking divisibility up to its square root.
    long long divisor_sum(long long n);

    // Checks if a number is a perfect number by comparing it to the sum of its divisors.
    bool is_perfect_number(int n);

    // Checks if a number is an Armstrong number by summing the digits raised to the power of the number of digits.
    bool is_armstrong_number(int n);

    // Computes the factorial of a number by multiplying all integers from 1 to n.
    long long factorial(int n);

    // Computes Euler's totient function (phi) by counting numbers coprime to n.
    unsigned int phi(unsigned int n);


    // Matrix Operations

    // Creates a matrix with the given number of rows and columns, initialized to a default value.
    vector<vector<int>> create_matrix(int rows, int cols, int default_value = 0);

    // Adds two matrices element-wise. Matrices must have the same dimensions.
    vector<vector<int>> add_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);

    // Multiplies two matrices using the standard matrix multiplication algorithm.
    vector<vector<int>> multiply_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);

    // Transposes a matrix by swapping rows and columns.
    vector<vector<int>> transpose_mat(const vector<vector<int>>& mat);

    // Computes the determinant of a square matrix using recursive expansion.
    int determinant(const vector<vector<int>>& mat);

    // Subtracts two matrices element-wise. Matrices must have the same dimensions.
    vector<vector<int>> subtract_matrices(const vector<vector<int>>& mat1, const vector<vector<int>>& mat2);


    // Advanced Math Functions

    // Computes the sine of an angle using a Taylor series approximation.
    double sin(double x);

    // Computes the cosine of an angle using a Taylor series approximation.
    double cos(double x);

    // Computes the tangent of an angle as the ratio of sine to cosine.
    double tan(double x);

    // Computes the exponential function using a Taylor series approximation.
    double exp(double x);

    // Computes the natural logarithm using the Newton-Raphson method.
    double log(double x);


    // Statistics and Probability Functions

    // Computes the mean (average) of a dataset by summing all values and dividing by the count.
    double mean(const vector<double>& data);

    // Computes the median of a dataset by sorting and finding the middle value.
    double median(vector<double> data);

    // Computes the number of ways to choose k items from n items without regard to order.
    long long combination(int n, int k);

    // Computes the number of ways to arrange k items from n items with regard to order.
    long long permutation(int n, int k);

    // Computes the probability of exactly k successes in n trials using the binomial formula.
    double binomial_probability(int n, int k, double p);

    // Computes the probability of k events occurring in a fixed interval using the Poisson formula.
    double poisson_probability(int k, double lambda);

    // Computes the variance of a dataset by measuring the spread of values around the mean.
    double variance(const vector<double>& data);

    // Computes the standard deviation of a dataset as the square root of the variance.
    double standard_deviation(const vector<double>& data);

    // Computes the probability of up to k successes in n trials using the binomial formula.
    double cumulative_binomial_probability(int n, int k, double p);


    // Vector Operations

    // Adds two vectors element-wise. Vectors must have the same dimension.
    vector<double> vector_add(const vector<double>& v1, const vector<double>& v2);

    // Subtracts one vector from another element-wise. Vectors must have the same dimension.
    vector<double> vector_subtract(const vector<double>& v1, const vector<double>& v2);

    // Multiplies a vector by a scalar, scaling each element by the scalar.
    vector<double> scalar_multiply(const vector<double>& v, double scalar);

    // Computes the dot product of two vectors by summing the products of corresponding elements.
    double dot_product(const vector<double>& v1, const vector<double>& v2);

    // Computes the cross product of two 3D vectors using the standard formula.
    vector<double> cross_product(const vector<double>& v1, const vector<double>& v2);

    // Computes the magnitude (length) of a vector using the Euclidean norm.
    double vector_magnitude(const vector<double>& v);

    // Normalizes a vector by dividing each element by its magnitude, resulting in a unit vector.
    vector<double> vector_normalize(const vector<double>& v);

    // Computes the angle between two vectors using the dot product and magnitudes.
    double vector_angle(const vector<double>& v1, const vector<double>& v2);

    // Projects one vector onto another using the dot product and magnitude.
    vector<double> vector_projection(const vector<double>& v1, const vector<double>& v2);

    // Computes the Euclidean distance between two vectors by summing the squared differences.
    double vector_distance(const vector<double>& v1, const vector<double>& v2);

} // namespace MathLib

#endif // MATHLIB_H
