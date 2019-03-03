//
// Created by davelinux on 28.2.19.
//

#ifndef PERCEPTON_NEURON_H
#define PERCEPTON_NEURON_H

#include <iostream>
#include <vector>


class Neuron{
private:
    int m_epochs = 100;
    double eLearn = 0.1;
    std::vector<double> m_errors;
public:
     //std::vector<T> Input;
     std::vector<double> Weight;
     int Bias = 1;
     double W_Bias = 0.1;
     double Probabilities( std::vector<double> Input);
     int Predict(std::vector<double> Input);
     int Train(std::vector<std::vector<double>> Input,std::vector<double> Output);



};


#endif //PERCEPTON_NEURON_H
