#include <iostream>


struct Pixel
{
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 0;

  unsigned sum() {
    return a;
  }
  Pixel operator*(const int &c) const {
    return Pixel{uint8_t(r * (c / 256.)), uint8_t(g * (c / 256.)), uint8_t(b * (c / 256.)), uint8_t(255)};
  }
  Pixel operator+(const Pixel &pix) const {
    return Pixel{uint8_t(r + pix.r), uint8_t(g + pix.g), uint8_t(b + pix.b), uint8_t(255)};
  }
};


int main(void) {

    Pixel pix{220, 210, 111, 255};
    Pixel back{0, 0, 0, 20};
    pix = pix * (256 - back.a) + back * back.a;
    std::cout << 1. - back.a / 256. << std::endl;
    std::cout << 1. - back.a / 256. << std::endl;
    std::cout << int(pix.a) << " " << int(pix.g) << " " << int(pix.b) << " " << int(pix.r) << std::endl;

    return 0;
}