#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {

void rosenblatt(double *X_data, double *Y_data, int rows, int cols, double learning_rate, int max_iterations, double *w, int nb_class)
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
    std::vector<std::vector<double>> w_vec(nb_class, std::vector<double>(cols + 1));
    for (int i = 0; i < nb_class; i++)
    {
        for (int j = 0; j <= cols; j++)
        {
            w_vec[i][j] = 0.0; // ou rand() / double(RAND_MAX) pour des valeurs aléatoires
        }
    }

    // Entraînement du perceptron pendant max_iterations itérations
    for (int iter = 0; iter < max_iterations; iter++)
    {
        std::vector<std::vector<double>> gradient(nb_class, std::vector<double>(cols + 1, 0.0));

        for (int j = 0; j < rows; j++)
        {
            double *x = &X[j * (cols + 1)];
            double y = Y_data[j];

            for (int i = 0; i < nb_class; i++)
            {
                double dot_product = std::inner_product(x, x + cols + 1, w_vec[i].begin(), 0.0);

                if (i == y && dot_product <= 0)
                {
                    for (int k = 0; k <= cols; k++)
                    {
                        gradient[i][k] += x[k];
                    }
                }
                else if (i != y && dot_product > 0)
                {
                    for (int k = 0; k <= cols; k++)
                    {
                        gradient[i][k] -= x[k];
                    }
                }
            }
        }

        // Mise à jour des poids en utilisant la descente de gradient
        for (int i = 0; i < nb_class; i++)
        {
            for (int k = 0; k <= cols; k++)
            {
                w_vec[i][k] += learning_rate * gradient[i][k];
            }
        }
    }

    // Copie des poids w vers le tableau de sortie
    for (int i = 0; i < nb_class; i++)
    {
        for (int j = 0; j <= cols; j++)
        {
            w[i * (cols + 1) + j] = w_vec[i][j];
        }
    }
}

}  // extern "C"

