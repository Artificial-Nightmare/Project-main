#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <ctype.h>


extern "C" {
 

using namespace std;

class MyMLP {
public:
    MyMLP(vector<int> npl) : d(npl), L(npl.size() - 1) {
        W.resize(L + 1);
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int l = 1; l <= L; ++l) {
            W[l].resize(d[l-1] + 1);
            for (int i = 0; i <= d[l-1]; ++i) {
                W[l][i].resize(d[l] + 1);
                for (int j = 1; j <= d[l]; ++j) {
                    W[l][i][j] = (j == 0) ? 0.0 : dis(gen);
                }
            }
        }

        X.resize(L + 1);
        for (int l = 0; l <= L; ++l) {
            X[l].resize(d[l] + 1);
            for (int j = 0; j <= d[l]; ++j) {
                X[l][j] = (j == 0) ? 1.0 : 0.0;
            }
        }

        deltas.resize(L + 1);
        for (int l = 0; l <= L; ++l) {
            deltas[l].resize(d[l] + 1);
            for (int j = 0; j <= d[l]; ++j) {
                deltas[l][j] = 0.0;
            }
        }
    }

    void propagate(vector<double> inputs, bool is_classification) {
        for (int j = 0; j < d[0]; ++j) {
            X[0][j + 1] = inputs[j];
        }
        for (int l = 1; l <= L; ++l) {
            for (int j = 1; j <= d[l]; ++j) {
                double total = 0.0;
                for (int i = 0; i <= d[l - 1]; ++i) {
                    total += W[l][i][j] * X[l - 1][i];
                }
                if (l < L || is_classification) {
                    total = std::tanh(total);
                }
                X[l][j] = total;
            }
        }
    }

    vector<double> predict(vector<double> inputs, bool is_classification) {
        propagate(inputs, is_classification);
        vector<double> output;
        output.reserve(d[L]);
        for (int j = 1; j <= d[L]; ++j) {
            output.push_back(X[L][j]);
        }
        return output;
    }

    void train(vector<vector<double>> all_samples_inputs, vector<vector<double>> all_samples_expected_outputs,
              bool is_classification, int iteration_count, double alpha) {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(0, all_samples_inputs.size() - 1);
        for (int it = 0; it < iteration_count; ++it) {
            int k = dis(gen);
            auto inputs_k = all_samples_inputs[k];
            auto y_k = all_samples_expected_outputs[k];

            propagate(inputs_k, is_classification);
            for (int j = 1; j <= d[L]; ++j) {
                deltas[L][j] = X[L][j] - y_k[j - 1];
                if (is_classification) {
                    deltas[L][j] *= (1.0 - X[L][j] * X[L][j]);
                }
            }
            for (int l = L - 1; l >= 1; --l) {
                for (int i = 1; i <= d[l-1]; ++i) {
                    double total = 0.0;
                    for (int j = 1; j <= d[l]; ++j) {
                        total += W[l+1][i][j] * deltas[l+1][j];
                    }
                    deltas[l][i] = (1.0 - X[l][i] * X[l][i]) * total;
                }
            }
            for (int l = 1; l <= L; ++l) {
                for (int i = 0; i <= d[l-1]; ++i) {
                    for (int j = 1; j <= d[l]; ++j) {
                        W[l][i][j] -= alpha * X[l-1][i] * deltas[l][j];
                    }
                }
            }
        }
    }

private:
    vector<int> d;
    int L;
    vector<vector<vector<double>>> W;
    vector<vector<double>> X;
    vector<vector<double>> deltas;
};

int main(int argc, char *argv[]) {
    vector<int> npl = {2, 3, 1};
    MyMLP mlp(npl);

    vector<vector<double>> samples_inputs = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
    vector<vector<double>> samples_expected_outputs = {{0.0}, {1.0}, {1.0}, {0.0}};

    int iteration_count = 10000;
    double alpha = 0.1;
    bool is_classification = true;

    if (argc == 3) {
        iteration_count = atoi(argv[1]);
        alpha = atof(argv[2]);
    } else if (argc == 4) {
        iteration_count = atoi(argv[1]);
        alpha = atof(argv[2]);
        is_classification = (bool) atoi(argv[3]);
    }

    mlp.train(samples_inputs, samples_expected_outputs, is_classification, iteration_count, alpha);

    for (auto inputs : samples_inputs) {
        auto output = mlp.predict(inputs, is_classification);
        cout << "inputs: (" << inputs[0] << ", " << inputs[1] << ") -> output: " << output[0] << endl;
    }

    return 0;
}
}