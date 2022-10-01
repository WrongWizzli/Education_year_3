#ifndef MAIN_PLAYER_H
#define MAIN_PLAYER_H

#include "Image.h"
#include "Texture.h"

struct Point {
    int x;
    int y;
};

enum class MovementDir {
    UP,
    DOWN,
    LEFT,
    RIGHT
};

struct Player {
    explicit Player(Point pos = {.x = 10, .y = 10}) :
                    coords(pos), old_coords(coords) {};

    bool Moved() const;
    void ProcessInput(MovementDir dir, std::vector<bool> &wall);
    void Draw(Image &screen, int h_map, int w_map);
    void change_coords(int x, int y);
    struct Point get_coords() const;
    Stats get_player_info();
    void collect_reward(int id);
    void get_spike_damage(double time);
    void get_mob_damage(int dmg, double time);
    bool draw_wounded(double time, Image &screen);
    int get_attack () { return attack;}
    int get_flasks_and_hp (double time);
    void consume_flask() {flasks--;}
    void Next_level_offset(int h_map, int w_map);
    void change_pos(int x, int y) {coords.x = old_coords.x = x;
                                   coords.y = old_coords.y = y;}

    private:
    double hp = 5;
    double hp_regen = 0;
    int attack = 1;
    int flasks = 10;
    double flask_time = 0;
    double flask_cooldown = 3;
    double hp_regen_time = 0;
    double last_damage_time = -10;
    double wounded_time = -10;
    Point coords {.x = 10, .y = 10};
    Point old_coords {.x = 10, .y = 10};
    Pixel color {.r = 255, .g = 0, .b = 255, .a = 255};
    double move_speed = 2;
    int idle_cnt = 0, moving_cnt = 0;

};


#endif //MAIN_PLAYER_H
