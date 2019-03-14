cmake_minimum_required(VERSION 3.13)
project(Percepton)

set(CMAKE_CXX_STANDARD 17)


add_executable(Percepton main.cpp NeuralNet/Neuron.cpp NeuralNet/Neuron.h  BackPropagation/Neuron.cpp BackPropagation/Neuron.h BackPropagation/NeuronLayer.cpp BackPropagation/NeuronLayer.h BackPropagation/NeuralNetwork.cpp BackPropagation/NeuralNetwork.h)

find_package( OpenCV REQUIRED )
target_link_libraries(Percepton ${OpenCV_LIBS} )


INCLUDE_DIRECTORIES(  /home/daveliinux/GIT/VSB-NAVY/ChartDirector/include)
LINK_DIRECTORIES(  /home/daveliinux/GIT/VSB-NAVY/ChartDirector/lib/ )


file(GLOB LIBRARIES "/home/daveliinux/GIT/VSB-NAVY/ChartDirector/lib/*.so")
message("LIBRARIES = ${LIBRARIES}")

TARGET_LINK_LIBRARIES(Percepton  ${LIBRARIES})
