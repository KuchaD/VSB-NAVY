cmake_minimum_required(VERSION 3.13)
project(Percepton)

set(CMAKE_CXX_STANDARD 17)


add_executable(Percepton main.cpp NeuralNet/Neuron.cpp NeuralNet/Neuron.h)

find_package( OpenCV REQUIRED )
target_link_libraries(Percepton ${OpenCV_LIBS} )


INCLUDE_DIRECTORIES(  /home/davelinux/Stažené/ChartDirector/include)
LINK_DIRECTORIES(  /home/davelinux/Stažené/ChartDirector/lib/ )


file(GLOB LIBRARIES "/home/davelinux/Stažené/ChartDirector/lib/*.so")
message("LIBRARIES = ${LIBRARIES}")

TARGET_LINK_LIBRARIES(Percepton  ${LIBRARIES})
