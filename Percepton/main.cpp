#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/plot.hpp>
#include <numeric>
#include <chartdir.h>
#include <random>
#include "NeuralNet/Neuron.h"


using namespace std;
void PrintGraf(cv::Mat &Dest, std::vector<std::vector<double>> X,  std::vector<double> p)
{
    Dest = cv::Mat(1000,1000,CV_8UC3,cv::Scalar(255, 255, 255));


    int Yp = 2*200+1;
    int Yp2 = 2* 200+1;



    cv::line(Dest,cv::Point(200 + 500,Yp2+500),cv::Point(-200-500,Yp-500),cv::Scalar(255,0,0));




/*
    for(int i=0; i < X.size();i++)
    {
        auto Point = cvPoint(X.at(i)[0],X.at(i)[1]);
        cvDrawRect(&Dest,Point,cvPoint(Point.x+1,Point.y+1),&cvScale(0,0,0));
    }
*/
}

int main() {

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


    //free up resources
    delete c;
    cv::Mat s = cv::imread("linefill.png",cv::IMREAD_ANYCOLOR);
    cv::namedWindow("tet", 0);
    cv::imshow("tet",s);
    cv::waitKey(0);
    //free up resources    delete c;
    return 0;


}