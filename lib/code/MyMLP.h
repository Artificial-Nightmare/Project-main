#ifndef MYMLP_H
#define MYMLP_H

#include <vector>

#ifdef MYMLP_EXPORTS
#define MYMLP_API __declspec(dllexport)
#else
#define MYMLP_API __declspec(dllimport)
#endif

using namespace std;

class MYMLP_API MyMLP {
public:
    MyMLP(vector<int> npl);
    ~MyMLP();
    void propagate(vector<double>& inputs, bool is_classification);
    void train(double* samples_inputs, double* samples_expected_outputs, int samples_size, int inputs_size, int outputs_size, bool is_classification, int iteration_count, double alpha);
    vector<vector<double> > X;
    vector<vector<double> > W;
    vector<int> d;
    int L;
};

typedef void* MyMLPHandle;

extern "C" {
    MYMLP_API MyMLPHandle create_MyMLP(int* npl, int npl_size);
    MYMLP_API void free_MyMLP(MyMLPHandle handle);
    MYMLP_API void propagate_MyMLP(MyMLPHandle handle, double* inputs, int inputs_size, bool is_classification);
    MYMLP_API void predict_MyMLP(MyMLPHandle handle, double* inputs, int inputs_size, bool is_classification, double* output, int outputs_size);
    MYMLP_API void train_MyMLP(MyMLPHandle handle, double* samples_inputs, double* samples_expected_outputs, int samples_size, int inputs_size, int outputs_size, bool is_classification, int iteration_count, double alpha);
}

#endif
