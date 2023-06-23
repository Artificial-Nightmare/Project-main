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
        MLP* mlp = new MLP;

        mlp->L = npl_size - 1;
        mlp->d = new int[npl_size];

        for(int i = 0; i < npl_size; i++)
        {
            mlp->d[i] = npl[i];
        }

        mlp->W = new double**[mlp->L];
        mlp->X = new double*[mlp->L + 1];
        mlp->deltas = new double*[mlp->L];

        for(int i = 0; i < mlp->L; i++)
        {
            mlp->W[i] = new double*[mlp->d[i+1]];
            mlp->deltas[i] = new double[mlp->d[i+1]];

            for(int j = 0; j < mlp->d[i+1]; j++)
            {
                mlp->W[i][j] = new double[mlp->d[i]+1];

                for(int k = 0; k < mlp->d[i]+1; k++)
                {
                    mlp->W[i][j][k] = ((double)rand() / RAND_MAX) * 2 - 1;
                }
            }
        }

        for(int i = 0; i < mlp->L + 1; i++)
        {
            mlp->X[i] = new double[mlp->d[i]+1];
        }

        return mlp;
    }

    void propagate(MLP* mlp, double* inputs, bool is_classification)
    {
        for(int i = 0; i < mlp->d[0]; i++)
        {
            mlp->X[0][i] = inputs[i];
        }

        mlp->X[0][mlp->d[0]] = 1;

        for(int i = 1; i < mlp->L + 1; i++)
        {
            for(int j = 0; j < mlp->d[i]; j++)
            {
                double sum = 0;

                for(int k = 0; k < mlp->d[i-1]+1; k++)
                {
                    sum += mlp->W[i-1][j][k] * mlp->X[i-1][k];
                }

                mlp->X[i][j] = 1 / (1 + exp(-sum));
            }

            if(i != mlp->L || !is_classification)
            {
                mlp->X[i][mlp->d[i]] = 1;
            }
        }
    }

    void predict(MLP* mlp, double *inputs, int inputs_size, bool is_classification, double *output, int outputs_size)
    {
        propagate(mlp, inputs, is_classification);

        for(int i = 0; i < outputs_size; i++)
        {
            output[i] = mlp->X[mlp->L][i];
        }
    }

 void train(MLP* mlp, double *samples_inputs, double *samples_expected_outputs,
           int samples_size, int inputs_size, int outputs_size,
           bool is_classification, int iteration_count, double alpha)
    {
        double loss = 0;
        double accuracy = 0;
        
        for(int i = 0; i < iteration_count; i++)
        {
            int index = rand() % samples_size;

            propagate(mlp, &samples_inputs[index * inputs_size], is_classification);

            for(int j = 0; j < outputs_size; j++)
            {
                mlp->deltas[mlp->L-1][j] = (samples_expected_outputs[index * outputs_size + j] - mlp->X[mlp->L][j]) * mlp->X[mlp->L][j] * (1 - mlp->X[mlp->L][j]);
                loss += pow(samples_expected_outputs[index * outputs_size + j] - mlp->X[mlp->L][j], 2);
                if(is_classification)
                {
                    if(round(mlp->X[mlp->L][j]) == samples_expected_outputs[index * outputs_size + j])
                    {
                        accuracy += 1;
                    }
                }
            }

            for(int j = mlp->L - 2; j >= 0; j--)
            {
                for(int k = 0; k < mlp->d[j+1]; k++)
                {
                    double sum = 0;

                    for(int l = 0; l < mlp->d[j+2]; l++)
                    {
                        sum += mlp->W[j+1][l][k] * mlp->deltas[j+1][l];
                    }

                    mlp->deltas[j][k] = sum * mlp->X[j+1][k] * (1 - mlp->X[j+1][k]);
                }
            }

            for(int j = 0; j < mlp->L; j++)
            {
                for(int k = 0; k < mlp->d[j+1]; k++)
                {
                    for(int l = 0; l < mlp->d[j]+1; l++)
                    {
                        if(l == mlp->d[j])
                        {
                            mlp->W[j][k][l] += alpha * mlp->deltas[j][k];
                        }
                        else
                        {
                            mlp->W[j][k][l] += alpha * mlp->deltas[j][k] * mlp->X[j][l];
                        }
                    }
                }
            }
            
            if(i % 500 == 0)
            {
                loss = loss / (outputs_size * 500);
                accuracy = accuracy / (outputs_size * 500);
                printf("Iteration %d: Loss = %.4f, Accuracy = %.2f%%\n", i, loss, accuracy * 100);
                loss = 0;
                accuracy = 0;
            }
        }
    }


    void deleteMLP(MLP* mlp)
    {
        for(int i = 0; i < mlp->L; i++)
        {
            for(int j = 0; j < mlp->d[i+1]; j++)
            {
                delete[] mlp->W[i][j];
            }

            delete[] mlp->W[i];
            delete[] mlp->deltas[i];
        }

        for(int i = 0; i < mlp->L + 1; i++)
        {
            delete[] mlp->X[i];
        }

        delete[] mlp->d;
        delete[] mlp->W;
        delete[] mlp->X;
        delete[] mlp->deltas;

        delete mlp;
    }
}
