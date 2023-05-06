#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>

// Fonction de classification linéaire (algorithme de Rosenblatt)
std::vector<double> linear_classification(std::vector<std::vector<double>> X, std::vector<int> Y, double learning_rate, int max_iterations, std::string filename)
{
    std::vector<double> w(X[0].size()+1, 0); // Ajout de 1 pour l'intercept {w0, w1, w2}
    std::ofstream outFile(filename);

    // Algorithme de Rosenblatt
    for (int i = 0; i < max_iterations; i++) {
        for (int j = 0; j < X.size(); j++) {
            std::vector<double> x = X[j];
            int y = Y[j];
            // Ajout d'un 1 au début du vecteur d'entrée pour l'intercept
            x.insert(x.begin(), 1.0);
            if (y * std::inner_product(x.begin(), x.end(), w.begin(), 0.0) <= 0) {
                for (int k = 0; k < w.size(); k++) {
                    w[k] += learning_rate * y * x[k];
                }
            }
        }
    }

    // Écriture des données de sortie dans un fichier texte
    outFile << w[0] << "," << w[1] << "," << w[2] << std::endl;

    return w;
}

int main()
{
    // Données d'entraînement
    std::vector<std::vector<double>> X = {{1, 1}, {2, 3}, {3, 3}};
    std::vector<int> Y = {1, -1, -1};

    // Paramètres de l'algorithme
    double learning_rate = 0.1;
    int max_iterations = 10;
    std::string filename = "data.txt";

    // Exécution de la classification linéaire et enregistrement des résultats
    std::vector<double> w = linear_classification(X, Y, learning_rate, max_iterations, filename);

    return 0;
}
