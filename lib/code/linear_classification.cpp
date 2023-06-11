#include <iostream>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

extern "C" {

void rosenblatt(double *X_data, double *Y_data, int rows, int cols, double learning_rate, int max_iterations, int nb_classes, double *W)
{
    // Initialisation du générateur de nombres aléatoires
    srand(time(NULL));

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

    // Initialisation des poids avec des valeurs aléatoires centrées sur zéro et d'une variance égale à 1/cols
    double variance = 1.0 / cols;
    std::vector<std::vector<double>> W_vec(nb_classes, std::vector<double>(cols + 1));
    for (int c = 0; c < nb_classes; c++)
    {
        for (int j = 0; j <= cols; j++)
        {
            W_vec[c][j] = variance * ((double) rand() / RAND_MAX - 0.5);
        }
    }

    // Entraînement d'un perceptron pour chaque classe
    int nb_iterations = 0;
    for (int c = 0; c < nb_classes; c++)
    {
        double error = 1.0;
        while (error > 0.01 && nb_iterations < max_iterations)
        {
            // Générer un nombre aléatoire pour sélectionner une observation au hasard
            int random_idx = rand() % rows;
            double *x = &X[random_idx * (cols + 1)];
            double y = (Y_data[random_idx] == c) ? 1 : -1; // 1 si l'observation appartient à la classe c, -1 sinon
            double dot_product = std::inner_product(x, x + cols + 1, W_vec[c].begin(), 0.0);

            // Mettre à jour les poids en fonction de l'observation sélectionnée aléatoirement
            if (y * dot_product < 0)
            {
                for (int j = 0; j <= cols; j++)
                {
                    W_vec[c][j] += learning_rate * y * x[j];
                }
            }

            // Calculer l'erreur
            error = 0.0;
            for (int i = 0; i < rows; i++)
            {
                double *x_i = &X[i * (cols + 1)];
                double y_i = (Y_data[i] == c) ? 1 : -1;
                double dot_product_i = std::inner_product(x_i, x_i + cols + 1, W_vec[c].begin(), 0.0);
                error += std::max(0.0, -y_i * dot_product_i);
            }
            error /= rows;
            nb_iterations++;
        }
    }

    // Les poids sont stockés dans la matrice W pour Python
    for (int c = 0; c < nb_classes; c++)
    {
        for (int j = 0; j <= cols; j++)
        {
            W[c * (cols + 1) + j] = W_vec[c][j];
        }
    }

    printf("Nombre d'itérations effectuées : %d\n", nb_iterations);
}

} // extern "C"
