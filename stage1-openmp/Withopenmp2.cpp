#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <omp.h>

const double A1 = -1.0;
const double B1 =  1.0;
const double A2 = -0.6;
const double B2 =  0.6;
const double F_VAL = 1.0;

// =======================================
// Check if point (x, y) is inside domain D: x^2 + 4y^2 < 1
// =======================================
bool if_is_in_D(double x, double y) {
    return (x * x + 4.0 * y * y < 1.0);
}

// =======================================
// Compute the part of a segment lying inside the domain D
// =======================================
double cal_seg_len_in_D(double const_coord, double start_var, double end_var, bool is_ver) {
    double lij;
    if (is_ver) {
        double x0 = const_coord;
        if (std::abs(x0) >= 1.0) lij = 0.0;
        else {
            double y_min = -std::sqrt(std::max(0.0, (1.0 - x0 * x0) / 4.0));
            double y_max =  std::sqrt(std::max(0.0, (1.0 - x0 * x0) / 4.0));
            lij = std::max(0.0, std::min(end_var, y_max) - std::max(start_var, y_min));
        }
    } else {
        double y0 = const_coord;
        if (std::abs(2.0 * y0) >= 1.0) lij = 0.0;
        else {
            double x_min = -std::sqrt(std::max(0.0, 1.0 - 4.0 * y0 * y0));
            double x_max =  std::sqrt(std::max(0.0, 1.0 - 4.0 * y0 * y0));
            lij = std::max(0.0, std::min(end_var, x_max) - std::max(start_var, x_min));
        }
    }
    return lij;
}

// =======================================
// Fictitious region initialization
// =======================================
void fic_reg(std::vector<std::vector<double>>& a,
                              std::vector<std::vector<double>>& b,
                              std::vector<std::vector<double>>& B,
                              int M, int N, double h1, double h2, double eps) {

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M; ++i) {
        for (int j = 1; j <= N; ++j) {
            double x_i = A1 + i * h1;
            double y_j = A2 + j * h2;

            double lij_a = cal_seg_len_in_D(x_i - 0.5 * h1, y_j - 0.5 * h2, y_j + 0.5 * h2, true);
            if (std::abs(lij_a - h2) < 1e-9) a[i][j] = 1.0;
            else if (lij_a < 1e-9) a[i][j] = 1.0 / eps;
            else a[i][j] = (lij_a / h2) + (1.0 - (lij_a / h2)) / eps;

            double lij_b = cal_seg_len_in_D(y_j - 0.5 * h2, x_i - 0.5 * h1, x_i + 0.5 * h1, false);
            if (std::abs(lij_b - h1) < 1e-9) b[i][j] = 1.0;
            else if (lij_b < 1e-9) b[i][j] = 1.0 / eps;
            else b[i][j] = (lij_b / h1) + (1.0 - (lij_b / h1)) / eps;
        }
    }

    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M - 1; ++i)
        for (int j = 1; j <= N - 1; ++j) {
            double x_i = A1 + i * h1;
            double y_j = A2 + j * h2;
            B[i][j] = if_is_in_D(x_i, y_j) ? F_VAL : 0.0;
        }
}

// =======================================
// Dot product
// =======================================
double dot(const std::vector<std::vector<double>>& u,
           const std::vector<std::vector<double>>& v,
           int M, int N, double h1, double h2) {
    double sum = 0.0;
    #pragma omp parallel for collapse(2) reduction(+:sum)
    for (int i = 1; i <= M - 1; ++i)
        for (int j = 1; j <= N - 1; ++j)
            sum += u[i][j] * v[i][j];
    return sum * h1 * h2;
}

// =======================================
// Apply A
// =======================================
std::vector<std::vector<double>> mat_A(const std::vector<std::vector<double>>& w,
                                         const std::vector<std::vector<double>>& a,
                                         const std::vector<std::vector<double>>& b,
                                         int M, int N, double h1, double h2) {
    std::vector<std::vector<double>> Aw(M, std::vector<double>(N, 0.0));
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= M - 1; ++i)
        for (int j = 1; j <= N - 1; ++j) {
            double Ax = -1.0 / h1 * (a[i+1][j] * (w[i+1][j] - w[i][j]) / h1 - a[i][j] * (w[i][j] - w[i-1][j]) / h1);
            double Ay = -1.0 / h2 * (b[i][j+1] * (w[i][j+1] - w[i][j]) / h2 - b[i][j] * (w[i][j] - w[i][j-1]) / h2);
            Aw[i][j] = Ax + Ay;
        }
    return Aw;
}

// =======================================
// Apply D^{-1}
// =======================================
std::vector<std::vector<double>> mat_D(const std::vector<std::vector<double>>& r,
                                            const std::vector<std::vector<double>>& a,
                                            const std::vector<std::vector<double>>& b,
                                            int M, int N, double h1, double h2) {
    std::vector<std::vector<double>> z(M + 1, std::vector<double>(N + 1, 0.0));
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < M; ++i)
        for (int j = 1; j < N; ++j) {
            double D_ij = (a[i+1][j] + a[i][j]) / (h1*h1) + (b[i][j+1] + b[i][j]) / (h2*h2);
            z[i][j] = (D_ij != 0.0) ? r[i][j] / D_ij : 0.0;
        }
    return z;
}

// =======================================
// Preconditioned Conjugate Gradient
// =======================================
void solve(int M, int N, double delta, int max_iter) {
    double h1 = (B1 - A1) / M;
    double h2 = (B2 - A2) / N;
    double eps = std::max(h1, h2) * std::max(h1, h2);

    std::vector<std::vector<double>> B(M, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> a(M + 1, std::vector<double>(N + 1, 0.0));
    std::vector<std::vector<double>> b(M + 1, std::vector<double>(N + 1, 0.0));
    fic_reg(a, b, B, M, N, h1, h2, eps);

    std::vector<std::vector<double>> w(M + 1, std::vector<double>(N + 1, 0.0));
    std::vector<std::vector<double>> w_prev = w;  // ⬅️ 保存上一次迭代结果
    std::vector<std::vector<double>> r = B;
    std::vector<std::vector<double>> z = mat_D(r, a, b, M, N, h1, h2);
    std::vector<std::vector<double>> p = z;

    double zr_old = dot(z, r, M, N, h1, h2);

    for (int k = 1; k <= max_iter; ++k) {
        std::vector<std::vector<double>> Ap = mat_A(p, a, b, M, N, h1, h2);
        double denom = dot(Ap, p, M, N, h1, h2);
        if (denom < 1e-15) break;

        double alpha = zr_old / denom;

        // 更新 w(k+1)
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                w[i][j] += alpha * p[i][j];

        // 更新 r(k+1)
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                r[i][j] -= alpha * Ap[i][j];

        z = mat_D(r, a, b, M, N, h1, h2);
        double zr_new = dot(z, r, M, N, h1, h2);
        double beta = zr_new / zr_old;
        zr_old = zr_new;

        // 更新 p(k+1)
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j)
                p[i][j] = z[i][j] + beta * p[i][j];

        // === ⬇️ 模长差收敛判断部分（文档公式） ===
        double diff_norm = 0.0;
        #pragma omp parallel for collapse(2) reduction(+:diff_norm)
        for (int i = 1; i < M; ++i)
            for (int j = 1; j < N; ++j) {
                double diff = w[i][j] - w_prev[i][j];
                diff_norm += diff * diff;
            }
        diff_norm = std::sqrt(diff_norm * h1 * h2);

        if (diff_norm < delta) {
            std::cout << "Converged after " << k << " iterations (||w(k+1)-w(k)|| < δ)." << std::endl;
            break;
        }

        // 更新 w_prev
        w_prev = w;
    }
}

// =======================================
// Main
// =======================================
int main() {
    int M = 400, N = 600;
    double delta = 1e-6;
    int max_iter = (M - 1) * (N - 1);

    std::cout << "--- (Variant 9: Ellipse x^2 + 4y^2 < 1, OpenMP Test) ---\n";
    std::cout << "Grid: M=" << M << ", N=" << N << "\n";
    std::cout << "--------------------------------------------------------\n";

    // thread settings to test
    std::vector<int> threads = {2, 4, 8, 16};

    for (int t : threads) {
        omp_set_num_threads(t);
        double t1 = omp_get_wtime();
        solve(M, N, delta, max_iter);
        double t2 = omp_get_wtime();
        std::cout << "Threads = " << std::setw(2) << t
                  << " | Time = " << std::fixed << std::setprecision(3)
                  << (t2 - t1) << " s\n";
    }

    std::cout << "--------------------------------------------------------\n";
    return 0;
}


