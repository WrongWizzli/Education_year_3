#ifndef MAIN_IMAGE_H
#define MAIN_IMAGE_H

#include <string>
#include <iostream>
constexpr int tileSize = 16;
struct Pixel
{
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 0;

  unsigned sum() {
    return a;
  }
  Pixel operator*(const uint8_t &c) const {
    return Pixel{uint8_t(r * (c / 256.)), uint8_t(g * (c / 256.)), uint8_t(b * (c / 256.)), uint8_t(255)};
  }
  Pixel operator+(const Pixel &pix) const {
    return Pixel{uint8_t(r + pix.r), uint8_t(g + pix.g), uint8_t(b + pix.b), uint8_t(255)};
  }
};

constexpr Pixel backgroundColor{0, 0, 0, 0};

struct Image
{
  explicit Image(const std::string &a_path);
  Image(int a_width, int a_height, int a_channels);

  int Save(const std::string &a_path);

  int Width()    const { return width; }
  int Height()   const { return height; }
  int Channels() const { return channels; }
  size_t Size()  const { return size; }
  Pixel* Data()        { return  data; }

  Pixel GetPixel(int x, int y) { return data[width * y + x];}
  void  PutPixel(int x, int y, const Pixel &pix) { data[width* y + x] = pix; }
  void PutalphaPixel(int x, int y, const Pixel &pix) {data[width * y + x] = data[width * y + x] * (256 - pix.a) + pix * pix.a;}

  ~Image();

private:
  int width = -1;
  int height = -1;
  int channels = 3;
  size_t size = 0;
  Pixel *data = nullptr;
  bool self_allocated = false;
};



#endif //MAIN_IMAGE_H
