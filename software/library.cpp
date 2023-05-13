#include <vector>
#include <numeric>
#include <fstream>

#ifdef WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT // Pour les autres plateformes que Windows
#endif

extern "C"
{
EXPORT std::vector<double> linear_classification(std::vector<std::vector<double>> X, std::vector<int> Y, double learning_rate, int max_iterations, std::string filename)
{
    std::vector<double> w(X[0].size()+1, 0);
    std::ofstream outFile(filename);

    for (int i = 0; i < max_iterations; i++) {
        for (int j = 0; j < X.size(); j++) {
            std::vector<double> x = X[j];
            int y = Y[j];
            x.insert(x.begin(), 1.0);
            if (y * std::inner_product(x.begin(), x.end(), w.begin(), 0.0) <= 0) {
                for (int k = 0; k < w.size(); k++) {
                    w[k] += learning_rate * y * x[k];
                }
            }
        }
    }
    outFile << w[0] << "," << w[1] << "," << w[2] << std::endl;

    return w;
}
}
