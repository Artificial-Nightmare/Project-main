#include <iostream>

extern "C" {


// Structure pour stocker les exemples d'entraînement
struct TrainingExample {
    double* features;
    int label;
};

// Fonction pour entraîner le SVM avec l'algorithme du perceptron
void trainSVM(double* trainingData, size_t numExamples, size_t numFeatures, double* weights, double& bias, int numIterations) {
    // Initialiser les poids et le biais à zéro
    for (size_t i = 0; i < numFeatures; i++) {
        weights[i] = 0.0;
    }
    bias = 0.0;

    // Boucle d'entraînement
    for (int iteration = 0; iteration < numIterations; iteration++) {
        int misclassifiedCount = 0;

        for (size_t i = 0; i < numExamples; i++) {
            double* example = trainingData + i * (numFeatures + 1);
            double prediction = bias;

            for (size_t j = 0; j < numFeatures; j++) {
                prediction += weights[j] * example[j];
            }

            // Mettre à jour les poids et le biais en cas d'erreur de classification
            if (example[numFeatures] * prediction <= 0) {
                for (size_t j = 0; j < numFeatures; j++) {
                    weights[j] += example[j] * example[numFeatures];
                }
                bias += example[numFeatures];
                misclassifiedCount++;
            }
        }

        // Affichage de l'évolution de l'entraînement
        std::cout << "Iteration: " << iteration << " Misclassified Count: " << misclassifiedCount << std::endl;

        // Arrêter l'entraînement si tous les exemples sont correctement classés
        if (misclassifiedCount == 0) {
            break;
        }
    }
}

// Fonction pour prédire la classe d'un exemple avec le SVM entraîné
int predictSVM(const double* features, const double* weights, double bias, size_t numFeatures) {
    double prediction = bias;

    for (size_t i = 0; i < numFeatures; i++) {
        prediction += weights[i] * features[i];
    }

    return (prediction >= 0) ? 1 : -1;
}

}