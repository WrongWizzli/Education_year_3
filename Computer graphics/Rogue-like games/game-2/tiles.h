#ifndef TILES_H
#define TILES_H

#include "Image.h"
#include <vector>

constexpr int WINDOW_WIDTH = 1000, WINDOW_HEIGHT = 600;

struct Point
{
  int x;
  int y;
};

class Monster {
  private:
  struct Point pos;
  bool right_move = true;
  int max_dist;
  int tile_distance;
  int move_speed = 1;
  bool is_alive = true;

  public:
  explicit Monster(int move_distance, Point position) {
    max_dist = move_distance;
    tile_distance = move_distance;
    pos = position;
  }
  Monster (const Monster &m) {pos = m.pos, right_move = m.right_move, tile_distance = m.tile_distance;
                              move_speed = m.move_speed, is_alive = m.is_alive; max_dist = m.max_dist;}
  int x() const {return pos.x;}
  int y() const {return pos.y;}
  bool Last_side() const {return right_move;}
  bool Alive() const {return is_alive;}
  int Distance() const {return tile_distance;}
  int Speed() const {return move_speed;}
  void update_position();
  void Killed() {is_alive = false;}
};

int abs(int x);
struct Point read_map(Image &map, u_char *char_map, const char *path, int w, int h);
struct Point change_map(Image &map, u_char *char_map, const char *path, int w, int h);
void draw_moved_map(Image &map, Image& screen, Point &coords);
bool trap_location(Point coords, Image & screen, double time);
bool monster_location(Point coords, Image &screen, double time);
void draw_hp(Image &screen, double hp);
void attack_monster(struct Point coords);
int E_action(u_char *char_map, int w, Point coords);
void blackout(Image &screen, Image &theme, int a);

const Image FLOOR("./resources/tiles/floor_3.png");
const Image WALL("./resources/tiles/wall_mid.png");
const Image TUNNEL("./resources/tiles/hole.png");

const std::vector<Image> PLAYER{Image("./resources/tiles/knight_f_idle_anim_f2.png"), Image("./resources/tiles/knight_f_idle_anim_f3.png"),
                                Image("./resources/tiles/knight_f_run_anim_f1.png"), Image("./resources/tiles/knight_f_hit_anim_f0.png"),
                                Image("./resources/tiles/knight_f_run_anim_f0.png"), Image("./resources/tiles/knight_f_run_anim_f1.png"),
                                Image("./resources/tiles/knight_f_run_anim_f2.png"), Image("./resources/tiles/knight_f_run_anim_f3.png"),
                                Image("./resources/tiles/knight_f_idle_anim_f0.png"), Image("./resources/tiles/knight_f_idle_anim_f1.png"), 
                                Image("./resources/tiles/knight_f_idle_anim_f2.png"), Image("./resources/tiles/knight_f_idle_anim_f3.png")};

const std::vector<Image> TRAP{Image("./resources/tiles/floor_spikes_anim_f0.png"), Image("./resources/tiles/floor_spikes_anim_f1.png"),
                              Image("./resources/tiles/floor_spikes_anim_f2.png"), Image("./resources/tiles/floor_spikes_anim_f3.png")};

const std::vector<Image> HEART{Image("./resources/tiles/ui_heart_full.png"), Image("./resources/tiles/ui_heart_half.png"), Image("./resources/tiles/ui_heart_empty.png")};

const std::vector<Image> MONSTER{Image("./resources/tiles/swampy_run_anim_f0.png"), Image("./resources/tiles/swampy_run_anim_f1.png"),
                                 Image("./resources/tiles/swampy_run_anim_f2.png"), Image("./resources/tiles/swampy_run_anim_f3.png")};


#endif