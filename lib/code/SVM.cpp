#include <cmath>
#include <iostream>

extern "C" {

    void trainSVM(const double* trainingData, size_t numExamples, size_t numFeatures, double* weights, double* bias, int numIterations) {
        // Initialiser les poids et le biais à zéro
        for (size_t i = 0; i < numFeatures; i++) {
            weights[i] = 0.0;
        }
        *bias = 0.0;

        // Boucle d'entraînement
        for (int iteration = 0; iteration < numIterations; iteration++) {
            double totalLoss = 0.0;
            int correctCount = 0;

            for (size_t i = 0; i < numExamples; i++) {
                const double* example = trainingData + i * (numFeatures + 1);
                double prediction = *bias;

                for (size_t j = 0; j < numFeatures; j++) {
                    prediction += weights[j] * example[j];
                }

                double loss = example[numFeatures] * prediction;

                // Mettre à jour les poids et le biais en fonction de la perte
                if (loss <= 0) {
                    for (size_t j = 0; j < numFeatures; j++) {
                        weights[j] += example[j] * example[numFeatures];
                    }
                    *bias += example[numFeatures];
                } else {
                    correctCount++;
                }

                totalLoss += std::abs(loss);
            }

            if (iteration % 500 == 0) {
                double accuracy = static_cast<double>(correctCount) / numExamples * 100.0;
                std::cout << "Iteration: " << iteration << " Loss: " << totalLoss << " Accuracy: " << accuracy << "%" << std::endl;
            }
        }
    }

    // Fonction pour prédire le label avec le SVM entraîné
    int predictSVM(const double* features, const double* weights, double bias, size_t numFeatures) {
        double prediction = bias;

        for (size_t i = 0; i < numFeatures; i++) {
            prediction += weights[i] * features[i];
        }

        return (prediction >= 0) ? 1 : -1;
    }
}
