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
        int* d; // nombre de neurones dans chaque couche
        int L; // nombre de couches
        double*** W; // poids des connexions entre les neurones
        double** X; // valeurs des neurones dans chaque couche
        double** deltas; // erreur de chaque neurone dans chaque couche

        
    };


    MLP* createMLP(int* npl, int npl_size)
    {
        MLP* mlp = new MLP;
        mlp->L = npl_size;
        mlp->d = new int[npl_size];
        mlp->W = new double**[npl_size-1];
        mlp->X = new double*[npl_size];
        mlp->deltas = new double*[npl_size];
        
        for(int i=0; i<npl_size; i++){
            mlp->d[i] = npl[i];
            mlp->X[i] = new double[npl[i]];
            mlp->deltas[i] = new double[npl[i]];
            if(i!=npl_size-1){
                mlp->W[i] = new double*[npl[i]+1];
                for(int j=0; j<npl[i]+1; j++){
                    mlp->W[i][j] = new double[npl[i+1]];
                }
            }
        }  
        // Initialisation des poids avec des valeurs aléatoires entre -1 et 1
        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(-1.0, 1.0);
        for(int i=0; i<npl_size-1; i++){
            for(int j=0; j<npl[i]+1; j++){
                for(int k=0; k<npl[i+1]; k++){
                    mlp->W[i][j][k] = dis(gen);
                }
            }
        }
        
        return mlp;
    }

    void propagate(MLP* mlp, double* inputs, bool is_classification){
    for (int j = 0; j < mlp->d[0]; ++j){
        mlp->X[0][j + 1] = inputs[j];
    }
    for (int l = 1; l <= mlp->L; ++l){
        for (int j = 1; j <= mlp->d[l]; ++j){
            double total = 0.0;
            for (int i = 0; i <= mlp->d[l - 1]; ++i){
                total += mlp->W[l-1][i][j-1] * mlp->X[l - 1][i];
            }
            if (l < mlp->L || is_classification){
                total = 1.0 / (1.0 + exp(-total));
            }
            mlp->X[l][j] = total;
        }
    }
}


void predict(MLP* mlp, double *inputs, int inputs_size, bool is_classification, double *output, int outputs_size){
    // Vérification de la taille des entrées
    if(inputs_size != mlp->d[0]){
        cout << "Erreur : la taille des entrées ne correspond pas au nombre de neurones de la première couche." << endl;
        return;
    }
    
    propagate(mlp, inputs, is_classification);
    
    // Copie des sorties dans le tableau de sortie
    memcpy(output, mlp->X[mlp->L], outputs_size * sizeof(double));
}

    void train(MLP* mlp, double *samples_inputs, double *samples_expected_outputs,
           int samples_size, int inputs_size, int outputs_size,
           bool is_classification, int iteration_count, double alpha){
        
        // Boucle d'entraînement
        for(int it=0; it<iteration_count; it++){
            double loss = 0.0;
            double accuracy = 0.0;
            for(int i=0; i<samples_size; i++){
                // Propagation des entrées dans le réseau
                propagate(mlp, samples_inputs+i*inputs_size, is_classification);
                
                // Calcul de l'erreur de chaque neurone dans chaque couche
                for(int j=0; j<mlp->d[mlp->L-1]; j++){
                    mlp->deltas[mlp->L-1][j] = mlp->X[mlp->L-1][j] * (1.0 - mlp->X[mlp->L-1][j]) * (samples_expected_outputs[i*outputs_size+j] - mlp->X[mlp->L-1][j]);
                }
                for(int j=mlp->L-2; j>=0; j--){
                    for(int k=0; k<mlp->d[j]; k++){
                        double sum = 0.0;
                        for(int l=0; l<mlp->d[j+1]; l++){
                            sum += mlp->W[j][k][l] * mlp->deltas[j+1][l];
                        }
                        mlp->deltas[j][k] = mlp->X[j][k] * (1.0 - mlp->X[j][k]) * sum;
                    }
                }
                
                // Mise à jour des poids des connexions entre les neurones
                for(int j=0; j<mlp->L-1; j++){
                    for(int k=0; k<mlp->d[j]+1; k++){
                        for(int l=0; l<mlp->d[j+1]; l++){
                            mlp->W[j][k][l] += alpha * mlp->X[j][k] * mlp->deltas[j+1][l];
                        }
                    }
                }
                
                // Calcul du loss et de l'accuracy
                for(int j=0; j<mlp->d[mlp->L-1]; j++){
                    double expected_output = samples_expected_outputs[i*outputs_size+j];
                    double actual_output = mlp->X[mlp->L-1][j];
                    loss += pow(expected_output - actual_output, 2);
                    if(is_classification){
                        if(expected_output == 1.0 && actual_output >= 0.5){
                            accuracy += 1.0;
                        } else if(expected_output == 0.0 && actual_output < 0.5){
                            accuracy += 1.0;
                        }
                    } else {
                        accuracy += 1.0 - abs(expected_output - actual_output);
                    }
                }
            }
            loss /= samples_size;
            accuracy /= samples_size;
            
            // Affichage du loss et de l'accuracy toutes les 500 itérations
            if(it % 500 == 0){
                cout << "Iteration " << it << ", loss " << loss << ", accuracy " << accuracy << endl;
            }
        }
    }


    void deleteMLP(MLP* mlp)
    {
        for(int i=0; i<mlp->L-1; i++){
            for(int j=0; j<mlp->d[i]+1; j++){
                delete[] mlp->W[i][j];
            }
            delete[] mlp->W[i];
        }
        delete[] mlp->W;
        
        for(int i=0; i<mlp->L; i++){
            delete[] mlp->X[i];
            delete[] mlp->deltas[i];
        }
        delete[] mlp->X;
        delete[] mlp->deltas;
        
        delete[] mlp->d;
        
        delete mlp;
    }
}
