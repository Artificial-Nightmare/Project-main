
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

pair<double, double> regression_lineaire(vector<double> X, vector<double> Y) {

    double X_sum = accumulate(X.begin(), X.end(), 0.0);
    double Y_sum = accumulate(Y.begin(), Y.end(), 0.0);
    double XX_sum = inner_product(X.begin(), X.end(), X.begin(), 0.0);
    double XY_sum = inner_product(X.begin(), X.end(), Y.begin(), 0.0);

    double n = X.size();
    double slope = (n * XY_sum - X_sum * Y_sum) / (n * XX_sum - X_sum * X_sum);
    double intercept = (Y_sum - slope * X_sum) / n;

    return make_pair(slope, intercept);
}

int main() {
    vector<double> X = { 1, 2, 3, 4, 5 };
    vector<double> Y = { 1, 3, 5, 7, 9 };

    pair<double, double> result = regression_lineaire(X, Y);

    cout << "Pente : " << result.first << endl;
    cout << "Ordonnée à l'origine : " << result.second << endl;

    return 0;
}
