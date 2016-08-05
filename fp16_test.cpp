#include <iostream>
#include <iomanip>
#include <bitset>
#include "caffe/util/float16.hpp"
using namespace std;
using namespace caffe;

void printBinary32(float y) {
    int* x = reinterpret_cast<int*>(&y);
    bitset<sizeof(float) * CHAR_BIT> bits(*x);

    string s = bits.to_string<char, std::char_traits<char>, std::allocator<char> >();

    // sign
    cout << s[0];
    cout << " ";
    // exponent
    for (int i = 1; i <= 8; ++i) cout << s[i];
    cout << " ";
    // fraction
    for (int i = 9; i <= 31; ++i) cout << s[i];
    cout << endl;
}

void printBinary16(float16 y) {
    short* x = reinterpret_cast<short*>(&y);
    bitset<sizeof(float16) * CHAR_BIT> bits(*x);

    string s = bits.to_string<char, std::char_traits<char>, std::allocator<char> >();

    // sign
    cout << s[0];
    cout << " ";
    // exponent
    for (int i = 1; i <= 5; ++i) cout << s[i];
    cout << " ";
    // fraction
    for (int i = 6; i <= 15; ++i) cout << s[i];
    cout << endl;
}

template <typename T, typename Y>
T Get(const Y& y) {
    return (T)y;
}

int main() {
    //float x = 3.14159265358979323846;
    float x = 0.000271828175755218;
    float16 y = Get<float16>(x);

    cout.precision(15);
    //cout.setf(ios::fixed);

    cout << "float32[10]:\t" << x << endl;
    cout << "float16[10]:\t" << y << endl;

    cout << "float32[2]:\t"; printBinary32(x); 
    cout << "float16[2]:\t"; printBinary16(y);
}
