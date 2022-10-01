#ifndef MAIN_IMAGE_H
#define MAIN_IMAGE_H

#include <string>

constexpr int tileSize = 32;

struct Pixel
{
  uint8_t r = 0;
  uint8_t g = 0;
  uint8_t b = 0;
  uint8_t a = 255;
};

constexpr Pixel backgroundColor{0, 0, 0, 0};

struct Image
{
  explicit Image(const std::string &a_path);
  Image() {empty = true; self_allocated = false;}
  Image(int a_width, int a_height, int a_channels);
  Image(const Image &im);

  int Save(const std::string &a_path);

  int Width()    const { return width; }
  int Height()   const { return height; }
  int Channels() const { return channels; }
  size_t Size()  const { return size; }
  Pixel* Data()        { return  data; }

  Pixel GetPixel(int x, int y) const { return data[width * y + x];}
  void  PutPixel(int x, int y, const Pixel &pix) { data[width* y + x] = pix; }
  void AlphaMix(int x, int y, const Pixel &pix, int a) {
  data[x + y * width].r = (u_int(data[x + y * width].r) * (255 - a) + u_int(pix.r) * a) >> 8;
  data[x + y * width].g = (u_int(data[x + y * width].g) * (255 - a) + u_int(pix.g) * a) >> 8;
  data[x + y * width].b = (u_int(data[x + y * width].b) * (255 - a) + u_int(pix.b) * a) >> 8;
}
  ~Image();

private:
  int width = -1;
  int height = -1;
  int channels = 3;
  size_t size = 0;
  Pixel *data = nullptr;
  bool self_allocated = false;
  bool empty = false;
};



#endif //MAIN_IMAGE_H
