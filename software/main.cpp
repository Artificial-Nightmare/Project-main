#include <iostream>
#include <vector>
#include <numeric>


/*
3 fonctions pour faire une classification linéaire :
 - 
 -
 -
*/



int main()
{
    // X the data points
    std::vector<std::vector<double>> X = {{1, 1}, {2, 3}, {3, 3}};
    std::vector<int> Y = {1, -1, -1};  
    // Y the labels
    std::vector<double> w(X[0].size()+1, 0); // Adding 1 for the intercept term
    // learning rate
    double learning_rate = 0.1;
    int max_iterations = 10;
    // Rosenblatt
    for (int i = 0; i < max_iterations; i++) {
        for (int j = 0; j < X.size(); j++) {
            std::vector<double> x = X[j];
            int y = Y[j];
            // Add a 1 to the start of the input vector for the intercept on ne passe pas forcément a l'origine
            x.insert(x.begin(), 1.0);
            if (y * std::inner_product(x.begin(), x.end(), w.begin(), 0.0) <= 0) {
                for (int k = 0; k < w.size(); k++) {
                    w[k] += learning_rate * y * x[k];
                }
            }
        }
    }  
    return 0;
}
