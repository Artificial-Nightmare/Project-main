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
    std::vector<double> W_vec(nb_classes * (cols + 1));
    for (int i = 0; i < nb_classes; i++)
    {
        for (int j = 0; j <= cols; j++)
        {
            W_vec[i * (cols + 1) + j] = variance * ((double) rand() / RAND_MAX - 0.5);
        }
    }

    // Entraînement d'un perceptron pour chaque classe
    for (int c = 0; c < nb_classes; c++)
    {
        bool misclassified = true;
        for (int i = 0; i < max_iterations && misclassified; i++)
        {
            // Générer un nombre aléatoire pour sélectionner une observation au hasard
            int random_idx = rand() % rows;
            double *x = &X[random_idx * (cols + 1)];
            double y = (Y_data[random_idx] == c) ? 1 : -1; // 1 si l'observation appartient à la classe c, -1 sinon
            double dot_product = std::inner_product(x, x + cols + 1, &W_vec[c * (cols + 1)], 0.0);
            
            // Mettre à jour les poids en fonction de l'observation sélectionnée aléatoirement
            if (y * dot_product < 0)
            {
                for (int j = 0; j <= cols; j++)
                {
                    W_vec[c * (cols + 1) + j] += learning_rate * y * x[j];
                }
                misclassified = true;
            }
            else
            {
                misclassified = false;
            }
        }
    }

    // Les poids sont stockés dans la matrice W pour Python
    for (int i = 0; i < nb_classes * (cols + 1); i++)
    {
        W[i] = W_vec[i];
    }
}


} // extern "C"
