#ifndef MAIN_PLAYER_H
#define MAIN_PLAYER_H

#include "Image.h"
#include "tiles.h"
enum class MovementDir
{
  UP,
  DOWN,
  LEFT,
  RIGHT
};

struct Player
{
  explicit Player(Point pos = {.x = 10, .y = 10}) :
                 coords(pos), old_coords(coords) {};

  bool Moved() const;
  void ProcessInput(MovementDir dir, u_char *char_map, int w);
  void Draw(Image &screen, double time);
  void set_attack_time(double time) {attack_time = time;}
  void set_Coord(Point coord) {old_coords = coords = coord;}
  double Last_damage() const {return damage_time;}
  double HP() const {return hitpoint;}
  Point Coord() const {return coords;}
  int Coins() const {return coins;}
  void get_damage(double time) {
    if (time - damage_time >= 2) {
      hitpoint -= 0.5; 
      damage_time = time;
    }
  }
private:
  Point coords {.x = 10, .y = 10};
  Point old_coords {.x = 10, .y = 10};
  Pixel color {.r = 255, .g = 255, .b = 0, .a = 255};
  int move_speed = 1;
  double attack_time = -2;
  bool last_side = false;
  double hitpoint = 3;
  double damage_time = 0;
  int coins = 0;

};

#endif //MAIN_PLAYER_H
