#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

extern "C" {

void rosenblatt(double *X_data, double *Y_data, int rows, int cols, double learning_rate, int max_iterations, int nb_classes, double *W) {
    // Vérification que les arguments en entrée sont valides
    if (rows <= 0 || cols <= 0 || nb_classes <= 0) {
        std::cerr << "Erreur : les dimensions des données et des poids doivent être strictement positives" << std::endl;
        return;
    }

    // Ajout d'un biais aux données
    int nb_features = cols + 1;
    std::vector<double> X(rows * nb_features);
    for (int i = 0; i < rows; i++) {
        X[i * nb_features] = 1.0;
        for (int j = 1; j < nb_features; j++) {
            X[i * nb_features + j] = X_data[i * cols + j - 1];
        }
    }

    // Initialisation aléatoire des poids
    std::srand(std::time(nullptr));
    std::vector<double> weights(nb_features * nb_classes);
    for (int i = 0; i < nb_features * nb_classes; i++) {
        weights[i] = ((double) std::rand() / RAND_MAX) - 0.5;
    }

    // Boucle d'apprentissage
    int t;
    for (t = 0; t < max_iterations; t++) {
        // Tirage aléatoire d'un exemple
        int i = std::rand() % rows;

        // Calcul des scores pour chaque classe
        std::vector<double> scores(nb_classes);
        for (int c = 0; c < nb_classes; c++) {
            double score = 0.0;
            for (int j = 0; j < nb_features; j++) {
                score += X[i * nb_features + j] * weights[c * nb_features + j];
            }
            scores[c] = score;
        }

        // Calcul de la classe prédite
        int y_pred = 0;
        double max_score = scores[0];
        for (int c = 0; c < nb_classes; c++) {
            if (scores[c] > max_score) {
                y_pred = c;
                max_score = scores[c];
            }
        }

        // Mise à jour des poids si la prédiction est incorrecte
        if (y_pred != (int) Y_data[i]) {
            int y_true = (int) Y_data[i];
            for (int j = 0; j < nb_features; j++) {
                weights[y_true * nb_features + j] += learning_rate * X[i * nb_features + j];
                weights[y_pred * nb_features + j] -= learning_rate * X[i * nb_features + j];
            }
        }
    }

    std::cout << "Nombre d'iterations : " << t << std::endl;

    // Copie des poids dans le tableau de sortie
    std::copy(weights.begin(), weights.end(), W);
}

} // extern "C"
