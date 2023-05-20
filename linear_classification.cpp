// Fichier "linear_classification.cpp"

#include <vector>
#include <numeric>

extern "C" {
void linear_classification(double* X_data, int* Y_data, int rows, int cols, double learning_rate, int max_iterations, double* w_data) {
        double* w = new double[cols + 1]; // Ajout de 1 pour l'intercept {w0, w1, w2}
        for (int i = 0; i <= cols; i++) {
            w[i] = 0.0;
        }

        // Algorithme de Rosenblatt
        for (int i = 0; i < max_iterations; i++) {
            for (int j = 0; j < rows; j++) {
                double* x = &X_data[j * (cols + 1)];
                int y = Y_data[j];
                if (y * (w[0] + std::inner_product(x, x + cols + 1, w + 1, 0.0)) <= 0) {
                    w[0] += learning_rate * y;
                    for (int k = 1; k <= cols; k++) {
                        w[k] += learning_rate * y * x[k];
                    }
                }
            }
        }
        for (int i = 0; i <= cols; i++) {
            w_data[i] = w[i];
        }
        delete[] w;
    }
}




