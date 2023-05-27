#include "MyMLP.h"
#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <vector>

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

void predict(double* inputs, int inputs_size, bool is_classification, double* output, int outputs_size) {
    vector<double> inputs_vec(inputs, inputs + inputs_size);
    vector<double> output_vec;

    propagate(inputs_vec, is_classification);
    output_vec.resize(d[L]);
    for (int j = 1; j <= d[L]; ++j) {
        output_vec[j-1] = X[L][j];
    }

    for (int i = 0; i < outputs_size; ++i) {
        output[i] = output_vec[i];
    }
}


    void train(double* samples_inputs, double* samples_expected_outputs,
           int samples_size, int inputs_size, int outputs_size,
           bool is_classification, int iteration_count, double alpha) {
    vector<vector<double>> all_samples_inputs(samples_size, vector<double>(inputs_size));
    vector<vector<double>> all_samples_expected_outputs(samples_size, vector<double>(outputs_size));

    for (int i = 0; i < samples_size; ++i) {
        for (int j = 0; j < inputs_size; ++j) {
            all_samples_inputs[i][j] = samples_inputs[i*inputs_size + j];
        }
        for (int j = 0; j < outputs_size; ++j) {
            all_samples_expected_outputs[i][j] = samples_expected_outputs[i*outputs_size + j];
        }
    }

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis(0, samples_size - 1);
    for (int it = 0; it < iteration_count; ++it) {
        int k = dis(gen);
        vector<double> inputs_k = all_samples_inputs[k];
        vector<double> y_k = all_samples_expected_outputs[k];

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
/*
int main(int argc, char *argv[]) {
    vector<int> npl = {2, 3, 1};
    MyMLP mlp(npl);

    double samples_inputs[] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    double samples_expected_outputs[] = {0.0, 1.0, 1.0, 0.0};

    int samples_size = 4;
    int inputs_size = 2;
    int outputs_size = 1;
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

    mlp.train(samples_inputs, samples_expected_outputs, samples_size, inputs_size, outputs_size,
              is_classification, iteration_count, alpha);

    for (int i = 0; i < 4; ++i) {
        double inputs[] = {samples_inputs[i*inputs_size], samples_inputs[i*inputs_size + 1]};
        double output[1];
        mlp.predict(inputs, 2, is_classification, output, 1);
        cout << "inputs: (" << inputs[0] << ", " << inputs[1] << ") -> output: " << output[0] << endl;
    }

    return 0;
}
*/


}
