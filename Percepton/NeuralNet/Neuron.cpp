//
// Created by davelinux on 28.2.19.
//

#include "Neuron.h"
#include <opencv2/opencv.hpp>
#include <random>

double Neuron::Probabilities( std::vector<double > aInput)
{

    if(aInput.size() != Weight.size())
    {
        throw "Input Size have different size like weigth ";
    }

    double probabilities = Bias * W_Bias ;
    for (int i = 0; i < aInput.size(); i++)
    {
        probabilities += aInput.at(i) * Weight.at(i);

    }
    return probabilities;

}

int  Neuron::Predict(std::vector<double> Input)
{
    double lPb= Probabilities(Input);

    double Activation = 1/(1+exp(-(lPb)));

    if(Activation >= 1)
    {
        return  1;
    } else {
        return  -1;

    }
}

int Neuron::Train(std::vector<std::vector<double>> Input,std::vector<double> Output)
{

    std::uniform_real_distribution<double> unif(0,1);
    std::default_random_engine re;


    for (int i = 0; i < Input.at(0).size(); i++) // X[0].size() + 1 -> I am using +1 to add the bias term
    {
        Weight.push_back(unif(re));
    }

    for (int i = 0; i < m_epochs; i++)
    {
        int errors = 0;

        for (int j = 0; j < Input.size(); j++)
        {
            float update = eLearn * (Output.at(j) - Predict(Input.at(j)) );
            for (int w = 0; w < Weight.size(); w++)
            {
                Weight.at(w) += update * Input.at(j).at(w);
            }
            W_Bias = update;
            errors += update != 0 ? 1 : 0;
        }
        m_errors.push_back(errors);
    }
}