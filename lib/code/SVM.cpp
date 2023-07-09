#include <iostream>
#include <vector>
#include <cmath>

// Structure pour stocker les exemples d'entraînement
struct Example {
    std::vector<double> features;
    int label;
};

// Fonction pour entraîner le SVM
void trainSVM(const std::vector<Example>& trainingData, double& weight1, double& weight2, double& bias, int numIterations) {
    // Initialisation des poids et du biais à zéro
    weight1 = 0.0;
    weight2 = 0.0;
    bias = 0.0;

    // Entraînement du SVM
    // Pour chaque itération et chaque exemple d'entraînement :
    for (int iteration = 0; iteration < numIterations; iteration++) {
        // Pour chaque exemple d'entraînement : 

        for (const Example& example : trainingData) {
            // Calcul de la prédiction et de la perte
            double prediction = example.features[0] * weight1 + example.features[1] * weight2 + bias;
            
            double loss = example.label - prediction;

            // Mise à jour des poids et du biais
            weight1 += loss * example.features[0];
            weight2 += loss * example.features[1];
            bias += loss;
        }
    }
}

// Fonction pour prédire le label avec le SVM entraîné
int predictSVM(const std::vector<double>& features, double weight1, double weight2, double bias) {
    double prediction = features[0] * weight1 + features[1] * weight2 + bias;
    return (prediction >= 0) ? 1 : 0;
}


int main() {
    // Données d'entraînement XOR
    std::vector<Example> trainingData = {
        {{0, 0}, -1},
        {{0, 1}, 1},
        {{1, 0}, 1},
        {{1, 1}, -1}
    };

    double weight1, weight2, bias;

    // Entraînement du SVM
    trainSVM(trainingData, weight1, weight2, bias, 100000);

    // Prédiction du XOR en utilisant le SVM entraîné
    std::vector<std::vector<double>> testData = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    for (const std::vector<double>& features : testData) {
        int prediction = predictSVM(features, weight1, weight2, bias);
        std::cout << "Input: " << features[0] << " " << features[1] << " => Prediction: " << prediction << std::endl;
    }

    return 0;
}
