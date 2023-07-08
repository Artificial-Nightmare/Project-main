#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {

void rosenblatt(double *X_data, double *Y_data, int rows, int cols, double learning_rate, int max_iterations, double *w)
{
    // Ajout du biais à X
    std::vector<double> X(rows * (cols + 1));
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            X[i * (cols + 1) + j] = X_data[i * cols + j];
        }
        X[i * (cols + 1) + cols] = 1.0;
    }

    // Initialisation des poids à zéro ou à des valeurs aléatoires
    std::vector<double> w_vec(cols + 1);
    for (int i = 0; i <= cols; i++)
    {
        w_vec[i] = 0.0; // ou rand() / double(RAND_MAX) pour des valeurs aléatoires
    }

    // Entraînement du perceptron pendant max_iterations itérations
    for (int iter = 0; iter < max_iterations; iter++)
    {
        std::vector<double> gradient(cols + 1, 0.0);

        for (int j = 0; j < rows; j++)
        {
            double *x = &X[j * (cols + 1)];
            double y = Y_data[j];
            double dot_product = std::inner_product(x, x + cols + 1, w_vec.begin(), 0.0);

            if (y * dot_product <= 0)
            {
                for (int k = 0; k <= cols; k++)
                {
                    gradient[k] += y * x[k];
                }
            }
        }

        // Mise à jour des poids en utilisant la descente de gradient
        for (int k = 0; k <= cols; k++)
        {
            w_vec[k] += learning_rate * gradient[k];
        }
    }

    // Copie des poids w vers le tableau de sortie
    for (int i = 0; i <= cols; i++)
    {
        w[i] = w_vec[i];
    }
}

}  // extern "C"