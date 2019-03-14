//
// Created by daveliinux on 13.3.19.
//

#include <random>
#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(int num_input,int num_hidden, int num_output, double hiddenLayer_bias, double outputLayer_bias) {
    _num_inputs = num_input;

    _hiddenLayer = new NeuronLayer(num_hidden,hiddenLayer_bias);
    _outputLayer = new NeuronLayer(num_output,outputLayer_bias);

    int_weight_input_hidden();
    int_weight_hidden_output();
}

void NeuralNetwork::int_weight_input_hidden() {

    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_real_distribution<double> gen(0,1);

    for(auto nNeuron : _hiddenLayer->getNeuron())
    {
        for (int i = 0; i < _num_inputs ; ++i)
        {
            double index = gen(rng);
            nNeuron->AddWeight(index);
        }
    }

}

void NeuralNetwork::int_weight_hidden_output() {
    std::random_device device;
    std::mt19937 rng(device());
    std::uniform_real_distribution<double> gen(0.1);

    for(auto nNeuron : _outputLayer->getNeuron())
    {
        for (int i = 0; i < _hiddenLayer->getNeuron().size() ; ++i)
        {

            nNeuron->AddWeight( gen(rng) );
        }
    }

}

std::ostream &operator<<(std::ostream &os, const NeuralNetwork &nNN) {


    os << "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄" << std::endl;
    os << "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄" << std::endl;
    os << "NEURAL NETWORK" << std::endl;
    os << "Inputs: " <<  nNN._num_inputs <<std::endl;
    os << "HIDDEN LAYER" << std::endl;
    os <<  *nNN._hiddenLayer << std::endl;
    os << "Output LAYER" << std::endl;
    os <<  *nNN._outputLayer << std::endl;

    os << "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄" << std::endl;
    os << "▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄" << std::endl;

    return  os;
}

std::vector<double> NeuralNetwork::feed_forward(std::vector<double> inputs) {
    auto hiddenLayerOutput = _hiddenLayer->feed_forward(inputs);
    return _outputLayer->feed_forward(hiddenLayerOutput);
}

void NeuralNetwork::Train(std::vector<double> inputs, std::vector<double> outputs) {

    feed_forward(inputs);








}

double NeuralNetwork::Calculate_Error(std::vector<double> inputs, std::vector<double> output) {

    int total_error = 0;
    for (int i = 0; i < inputs.size() ; ++i) {

        feed_forward(inputs);
        for (int j = 0; j < output.size(); ++j) {
            total_error += _outputLayer->getNeuron()[j]->CalErorr(output[j]);

        }

    }

    return  total_error;

}
