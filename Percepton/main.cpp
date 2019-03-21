#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <numeric>
#include <chartdir.h>
#include <random>
#include "NeuralNet/Neuron.h"
#include "BackPropagation/NeuralNetwork.h"

#include <opencv2/opencv.hpp>
#include <dlib/mlp.h>
using namespace dlib;

using namespace std;


void cv1()
{

    std::vector<std::vector<double>> TrainSet;
    std::vector<double> result;
    for(int x=-100;x < 100;x+=1) {
        for (int y = -100; y < 100; y+=1) {
            std::vector<double> vector = {(double)x,(double)y};
            TrainSet.push_back(vector);
            if(2*x-1 <= y) {

                result.push_back(1);
            }
            else
            {
                result.push_back(-1);
            }
        }
    }

    Neuron lN ;
    lN.Train(TrainSet,result);


    std::random_device rde;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gene(rde()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dise(150, 450);
    std::vector<std::vector<double>> TestSet;
    std::vector<double> resultTest;

    for(int i = 0;i < 150;i+=1) {

        std::vector<double> vector = {(double)dise(gene),(double)dise(gene)};
        TestSet.push_back(vector);
        if(2*vector.at(0)-1 <= vector.at(1)) {

            resultTest.push_back(1);
        }
        else
        {
            resultTest.push_back(-1);
        }
    }

    auto e = lN.Test(TestSet,resultTest);
    std::cout << "Test :" << TestSet.size() << " Error: " << e << " Uspesnost: " << 100 - (double)(100.0/(double)TestSet.size())*e << " %";
    //std::cout << lN.Predict({0,5});
    std::cout.flush();



    // Create a XYChart object of size 450 x 420 pixels
    XYChart *c = new XYChart(450, 420);

    // Set the plotarea at (55, 65) and of size 350 x 300 pixels, with a light grey border
    // (0xc0c0c0). Turn on both horizontal and vertical grid lines with light grey color (0xc0c0c0)
    c->setPlotArea(55, 65, 350, 300, -1, -1, 0xc0c0c0, 0xc0c0c0, -1);

    // Add a legend box at (50, 30) (top of the chart) with horizontal layout. Use 12pt Times Bold
    // Italic font. Set the background and border color to Transparent.
    c->addLegend(50, 30, false, "timesbi.ttf", 12)->setBackground(Chart::Transparent);

    // Add a title to the chart using 18pt Times Bold Itatic font.
    c->addTitle("Graph", "timesbi.ttf", 18);


    // Set the axes line width to 3 pixels
    c->xAxis()->setWidth(3);
    c->yAxis()->setWidth(3);

    double xl[] = {150,-150};
    double yl[] = {2*150-1,2*-150-1};
    LineLayer *layer1 = c->addLineLayer(DoubleArray(yl, (int)(sizeof(xl) / sizeof(double)
    )), 0xff3333, "");
    layer1->setXData(DoubleArray(xl, (int)(sizeof(yl) / sizeof(double))));

    // Set the line width to 3 pixels
    layer1->setLineWidth(3);

    // Use 9 pixel square symbols for the data points
    layer1->getDataSet(0)->setDataSymbol(Chart::SquareSymbol, 9);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(-150, 150);

    for(int i = 0;i < 150;i+=1) {

        double x[] = {(double) dis(gen)};
        double y[] = {(double) dis(gen)};
        double r[] = {(double) lN.Predict({x[0], y[0]})};
        // Add an orange (0xff9933) scatter chart layer, using 13 pixel diamonds as symbols

        c->addScatterLayer(DoubleArray(x, (int) (sizeof(double)/sizeof(double))), DoubleArray(
                y, (int) ((sizeof(double)/sizeof(double)))), "",
                           Chart::DiamondSymbol, 5, (r[0] == 1) ? 0xff9933 : 0x33ff33);

    }



    // Output the chart
    c->makeChart("linefill.png");
    cv::Mat s = cv::imread("linefill.png",cv::IMREAD_ANYCOLOR);
    cv::namedWindow("tet", 0);
    cv::imshow("tet",s);
    cv::waitKey(0);

    //free up resources
    delete c;


}

void cv2() {

    std::vector<std::vector<double>> training_set = {{0, 0},
                                                     {0, 1},
                                                     {1, 0},
                                                     {1, 1}};
    std::vector<std::vector<double>> R = {{0},
                                          {1},
                                          {1},
                                          {0}};


    NeuralNetwork nn = NeuralNetwork(2, 2, 1);

    for (int i = 0; i < 1000000; ++i) {
        for (int j = 0; j < 4; ++j) {


            auto input = training_set[j];
            auto output = R[j];

            nn.Train(input, output);
            //cout << nn.Calculate_Error(input,output);


           // cout << nn;

        }
        for(auto nItem : nn.feed_forward({0,0}))
        {
            cout << "0 0 = " << nItem << " " << std::endl;
        };

        for(auto nItem : nn.feed_forward({0,1}))
        {
            cout << "0 1 = " << nItem << " " << std::endl;
        };
        for(auto nItem : nn.feed_forward({1,0}))
        {
            cout << "1 0 = " << nItem << " " << std::endl;
        };

        for(auto nItem : nn.feed_forward({1,1})) {
            cout << "1 1 = " << nItem << " " << std::endl;

        }
    }

}



void cv4()
{

}
int main() {

    // The mlp takes column vectors as input and gives column vectors as output.  The dlib::matrix
    // object is used to represent the column vectors. So the first thing we do here is declare
    // a convenient typedef for the matrix object we will be using.

    // This typedef declares a matrix with 2 rows and 1 column.  It will be the
    // object that contains each of our 2 dimensional samples.   (Note that if you wanted
    // more than 2 features in this vector you can simply change the 2 to something else)
    typedef matrix<double, 2, 1> sample_type;


    // make an instance of a sample matrix so we can use it below
    sample_type sample;

    // Create a multi-layer perceptron network.   This network has 2 nodes on the input layer
    // (which means it takes column vectors of length 2 as input) and 5 nodes in the first
    // hidden layer.  Note that the other 4 variables in the mlp's constructor are left at
    // their default values.
    mlp::kernel_1a_c net(2,10);

    // Now let's put some data into our sample and train on it.  We do this
    // by looping over 41*41 points and labeling them according to their
    // distance from the origin.



    std::vector<sample_type> lsample;
    std::vector<double> lResult;

    for (int i = 0; i < 100; ++i) {

        for (double r = 2.4; r < 4.0; r += 0.1) {
            double lLastResult = 0.1;

            for (int c = 0; c < 100; ++c) {

                double lRx = r * lLastResult * (1 - lLastResult);

                sample_type lS;
                lS(0) = r;
                lS(1) = lLastResult;

                net.train(lS, lRx);

                lLastResult = lRx;


            }
        }
    }



    // Now we have trained our mlp.  Let's see how well it did.
    // Note that if you run this program multiple times you will get different results. This
    // is because the mlp network is randomly initialized.

    // each of these statements prints out the output of the network given a particular sample.

    sample(0) = 2.5; //r
    sample(1) = 0.6; //value
    cout << "This sample should be close to 1 and it is classified as a " << net(sample) << endl;


    return 1;
}