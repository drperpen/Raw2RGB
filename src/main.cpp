
#include <iostream>
#include "common_types.h"

int main(int, char**){
    cv::Mat imgage = cv::Mat::zeros(3, 3, CV_8UC3);

    std::cout << imgage.size() << std::endl;
    std::cout << "Hello, from Raw2RGB!\n";
}
