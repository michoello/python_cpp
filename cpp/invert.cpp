// listinvert/invert.cpp
#include <vector>
#include <iostream>

#include "invert.h"

std::vector<int> invert(const std::vector<int> &v) {
    return std::vector<int>(v.rbegin(), v.rend());
}

