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

    // Initialisation des poids à zéro c'est un vecteur de taille cols + 1 car on a ajouté le biais à X 
    std::vector<double> w_vec(cols + 1);
for (int i = 0; i <= cols; i++)
{
    w_vec[i] = rand() / double(RAND_MAX);
} // random ou 0 ?
    // Entraînement du perceptron pendant max_iterations itérations
    for (int i = 0; i < max_iterations; i++)
    {
        bool misclassified = false;
        for (int j = 0; j < rows; j++)
        {
            double *x = &X[j * (cols + 1)];
            double y = Y_data[j];
            double dot_product = std::inner_product(x, x + cols + 1, w_vec.begin(), 0.0);
            if (y * dot_product <= 0)
            {
                misclassified = true;
                w_vec[0] += learning_rate * y;
                for (int k = 1; k <= cols; k++)
                {
                    w_vec[k] += learning_rate * y * x[k];
                }
            }
        }
        if (!misclassified)
        {
            break;
        }
        // Vérifier si la droite de séparation a le bon sens et inverser les poids si nécessaire
    }

    // les poids sortie w (pour Python) 
    for (int i = 0; i <= cols; i++)
    {
        w[i] = w_vec[i];
    }
}


}  // extern "C"
