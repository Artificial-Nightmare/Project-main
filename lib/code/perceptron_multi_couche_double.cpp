#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <ctype.h>
#include <vector>

using namespace std;

extern "C"
{
    struct MLP
    {
        vector<int> d;
        int L;
        vector<vector<vector<double>>> W;
        vector<vector<double>> X;
        vector<vector<double>> deltas;
    };

   MLP* createMLP(int* npl, int npl_size)
{
    MLP* mlp = new MLP();
    mlp->d = vector<int>(npl, npl + npl_size);
    mlp->L = npl_size - 1;
    mlp->W.resize(mlp->L + 1);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dis(-1.0, 1.0);

    for (int l = 1; l <= mlp->L; ++l)
    {
        mlp->W[l].resize(mlp->d[l - 1] + 1);
        for (int i = 0; i <= mlp->d[l - 1]; ++i)
        {
            mlp->W[l][i].resize(mlp->d[l] + 1);
            for (int j = 1; j <= mlp->d[l]; ++j)
            {
                mlp->W[l][i][j] = (j == 0) ? 0.0 : dis(gen);
            }
        }
    }

    mlp->X.resize(mlp->L + 1);
    for (int l = 0; l <= mlp->L; ++l)
    {
        mlp->X[l].resize(mlp->d[l] + 1);
        for (int j = 0; j <= mlp->d[l]; ++j)
        {
            mlp->X[l][j] = (j == 0) ? 1.0 : 0.0;
        }
    }

    mlp->deltas.resize(mlp->L + 1);
    for (int l = 0; l <= mlp->L; ++l)
    {
        mlp->deltas[l].resize(mlp->d[l] + 1);
        for (int j = 0; j <= mlp->d[l]; ++j)
        {
            mlp->deltas[l][j] = 0.0;
        }
    }

    return mlp;
}

    void propagate(MLP* mlp, vector<double> inputs, bool is_classification)
    {
        for (int j = 0; j < mlp->d[0]; ++j)
        {
            mlp->X[0][j + 1] = inputs[j];
        }
        for (int l = 1; l <= mlp->L; ++l)
        {
            for (int j = 1; j <= mlp->d[l]; ++j)
            {
                double total = 0.0;
                for (int i = 0; i <= mlp->d[l - 1]; ++i)
                {
                    total += mlp->W[l][i][j] * mlp->X[l - 1][i];
                }
                if (l < mlp->L || is_classification)
                {
                    total = std::tanh(total);
                }
                mlp->X[l][j] = total;
            }
        }
    }

    void predict(MLP* mlp, double *inputs, int inputs_size, bool is_classification, double *output, int outputs_size)
    {
        vector<double> inputs_vec(inputs, inputs + inputs_size);
        vector<double> output_vec;

        propagate(mlp, inputs_vec, is_classification);
        output_vec.resize(mlp->d[mlp->L]);
        for (int j = 1; j <= mlp->d[mlp->L]; ++j)
        {
            output_vec[j - 1] = mlp->X[mlp->L][j];
        }

        for (int i = 0; i < outputs_size; ++i)
        {
            output[i] = output_vec[i];
        }
    }

    void train(MLP* mlp, double *samples_inputs, double *samples_expected_outputs,
               int samples_size, int inputs_size, int outputs_size,
               bool is_classification, int iteration_count, double alpha)
    {
        vector<vector<double>> all_samples_inputs(samples_size, vector<double>(inputs_size));
        vector<vector<double>> all_samples_expected_outputs(samples_size, vector<double>(outputs_size));

        for (int i = 0; i < samples_size; ++i)
        {
            for (int j = 0; j < inputs_size; ++j)
            {
                all_samples_inputs[i][j] = samples_inputs[i * inputs_size + j];
            }
            for (int j = 0; j < outputs_size; ++j)
            {
                all_samples_expected_outputs[i][j] = samples_expected_outputs[i * outputs_size + j];
            }
        }

        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<int> dis(0, samples_size - 1);
        for (int it = 0; it < iteration_count; ++it)
        {
            int k = dis(gen);
            vector<double> inputs_k = all_samples_inputs[k];
            vector<double> y_k = all_samples_expected_outputs[k];

            propagate(mlp, inputs_k, is_classification);
            for (int j = 1; j <= mlp->d[mlp->L]; ++j)
            {
                mlp->deltas[mlp->L][j] = mlp->X[mlp->L][j] - y_k[j - 1];
                if (is_classification)
                {
                    mlp->deltas[mlp->L][j] *= (1.0 - mlp->X[mlp->L][j] * mlp->X[mlp->L][j]);
                }
            }
            for (int l = mlp->L - 1; l >= 1; --l)
            {
                for (int i = 1; i <= mlp->d[l - 1]; ++i)
                {
                    double total = 0.0;
                    for (int j = 1; j <= mlp->d[l]; ++j)
                    {
                        total += mlp->W[l + 1][i][j] * mlp->deltas[l + 1][j];
                    }
                    mlp->deltas[l][i] = (1.0 - mlp->X[l][i] * mlp->X[l][i]) * total;
                }
            }
            for (int l = 1; l <= mlp->L; ++l)
            {
                for (int i = 0; i <= mlp->d[l - 1]; ++i)
                {
                    for (int j = 1; j <= mlp->d[l]; ++j)
                    {
                        mlp->W[l][i][j] -= alpha * mlp->X[l - 1][i] * mlp->deltas[l][j];
                    }
                }
            }
        }
    }

    void deleteMLP(MLP* mlp)
    {
        delete mlp;
    }
}

