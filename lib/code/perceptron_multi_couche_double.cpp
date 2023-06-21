#include <iostream>
#include <cmath>
#include <random>
#include <cstdlib>
#include <cstring>
#include <ctype.h>

using namespace std;

extern "C"
{
    struct MLP
    {
        int* d;
        int L;
        double*** W;
        double** X;
        double** deltas;

        
    };


    MLP* createMLP(int* npl, int npl_size)
    {
        MLP* mlp = new MLP();
        mlp->d = new int[npl_size];
        memcpy(mlp->d, npl, npl_size * sizeof(int));
        mlp->L = npl_size - 1;
        mlp->W = new double**[mlp->L + 1];
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(-1.0, 1.0);

        for (int l = 1; l <= mlp->L; ++l)
        {
            mlp->W[l] = new double*[mlp->d[l - 1] + 1];
            for (int i = 0; i <= mlp->d[l - 1]; ++i)
            {
                mlp->W[l][i] = new double[mlp->d[l] + 1];
                for (int j = 1; j <= mlp->d[l]; ++j)
                {
                    mlp->W[l][i][j] = (j == 0) ? 0.0 : dis(gen);
                }
            }
        }

        mlp->X = new double*[mlp->L + 1];
        for (int l = 0; l <= mlp->L; ++l)
        {
            mlp->X[l] = new double[mlp->d[l] + 1];
            for (int j = 0; j <= mlp->d[l]; ++j)
            {
                mlp->X[l][j] = (j == 0) ? 1.0 : 0.0;
            }
        }

        mlp->deltas = new double*[mlp->L + 1];
        for (int l = 0; l <= mlp->L; ++l)
        {
            mlp->deltas[l] = new double[mlp->d[l] + 1];
            for (int j = 0; j <= mlp->d[l]; ++j)
            {
                mlp->deltas[l][j] = 0.0;
            }
        }

        return mlp;
    }

    void propagate(MLP* mlp, double* inputs, bool is_classification)
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
        propagate(mlp, inputs, is_classification);
        for (int j = 0; j <= mlp->d[mlp->L]; ++j)
        {
            output[j - 1] = mlp->X[mlp->L][j];
        }
    }

    void train(MLP* mlp, double *samples_inputs, double *samples_expected_outputs,
               int samples_size, int inputs_size, int outputs_size,
               bool is_classification, int iteration_count, double alpha)
    {
        double** all_samples_inputs = new double*[samples_size];
        double** all_samples_expected_outputs = new double*[samples_size];

        for (int i = 0; i < samples_size; ++i)
        {
            all_samples_inputs[i] = new double[inputs_size];
            all_samples_expected_outputs[i] = new double[outputs_size];
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
            double* inputs_k = all_samples_inputs[k];
            double* y_k = all_samples_expected_outputs[k];

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

        for (int i = 0; i < samples_size; ++i)
        {
            delete[] all_samples_inputs[i];
            delete[] all_samples_expected_outputs[i];
        }
        delete[] all_samples_inputs;
        delete[] all_samples_expected_outputs;
    }

    void deleteMLP(MLP* mlp)
    {
        for (int l = 1; l <= mlp->L; ++l)
        {
            for (int i = 0; i <= mlp->d[l - 1]; ++i)
            {
                delete[] mlp->W[l][i];
            }
            delete[] mlp->W[l];
        }
        delete[] mlp->W;

        for (int l = 0; l <= mlp->L; ++l)
        {
            delete[] mlp->X[l];
            delete[] mlp->deltas[l];
        }
        delete[] mlp->X;
        delete[] mlp->deltas;

        delete[] mlp->d;
        delete mlp;
    }
}
