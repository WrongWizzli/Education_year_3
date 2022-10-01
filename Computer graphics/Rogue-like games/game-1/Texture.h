
#ifndef TEXTURE_H
#define TEXTURE_H

#include <iostream>
#include <fstream>
#include "Image.h"
#include <vector>
#include <cstring>

#define GLFW_DLL
#include <GLFW/glfw3.h>
constexpr GLsizei WINDOW_WIDTH = 1413, WINDOW_HEIGHT = 666;

//SIZES OF GAME WINDOW WITHOUT HUD
const int WINDOW_WIDTH_G = 1133, WINDOW_HEIGHT_G = 666;

struct Properties {
    const char *path;
    int h;
    int w;
};

struct Stats {
    double hp;
    int flasks;
    int attack;
    double flask_cd;
    double regen;
};

struct Dir_k {
    double kx;
    double ky;
};

struct Player_Pos {
    int x = 0;
    int y = 0;
};
//Struct for moving objects

struct Directional_Shot {
    double x = -1;
    double y = -1;
    double kx;
    double ky;
    int damage;
    
    bool operator==(const Directional_Shot &a) {
        if (this->x == a.x && this->y == a.y && this->kx == a.kx && this->ky == a.ky && this->damage == a.damage)
            return true;
        return false;
    }

};
const int Multiplier = 2;
const int Tile_size = 16 * Multiplier;
//*------------------------------------------------------------------------------------------

class Texture {

    Pixel *data = nullptr;
    int width;
    int height;

    void read_image(const char *path);

    public:

    Texture(){}
    Texture(int w, int h);
    Texture(const struct Properties Obj);
    Texture(const Texture &c);
    ~Texture();
    Texture &operator=(const Texture &c);
    Texture mirror() const;
    Texture flip() const;
    
    int get_w() const;
    int get_h() const;

    bool not_empty_pix(int x, int y) const;
    Pixel GetPixel(int x, int y) const;
    void  PutPixel(int x, int y, const Pixel &pix);

};
//?=========================================================================================
//TODO MOBS
double max(double x, double y);
class Mob {

    int mob_skin;
    int mob_x, mob_y;
    double last_frame = 0;
    double last_move_frame = -50;
    unsigned char state = 'i';      //'i' (idle) || 'm' (moving)
    double hp;
    double attack;
    double basic_mob_speed = 1.2;
    unsigned char mob_type;
    public:

    std::vector<Texture> skin;
    int shot_x, shot_y;
    double last_shot_frame = -50;

    Mob(unsigned char mob_type, int skin_id, int x, int y);
    ~Mob(){}
    bool detect_player(int tr_x, int tr_y, int x, int y);
    void mob_animation(Image &screen, double time, int w, int h, int tr_x, int tr_y);
    int proceed_move(unsigned char *symmap, int tr_x, int tr_y, int x, int y, int width, int type);

    void change_state(unsigned char s) {state = s;}
    void change_x(int x) {mob_x = x;}
    void change_y(int y) {mob_y = y;}
    unsigned char get_state() const {return state;}
    int get_x() const {return mob_x;}
    int get_y() const {return mob_y;}
    double get_hp() const {return hp;}
    double get_attack() const {return attack;}
    unsigned char get_type() const {return mob_type;}
    void reduce_hp(double damage)  {hp = max(0, hp -damage);}
};

//TODO--------------------------------------------------------------------------------------
enum ITEMS {STAFF_ID, STAFF_S_ID, HEART_ID, FLASK_ID, MANA_ID, HP_REGEN_ID};
int give_chest_reward();
void transform_coord(int &x, int &y, int part);
void draw_next_level_theme(Image &screen);
void draw_endgame_theme(Image &screen, int lost, uint8_t alpha);
void fade_in(Image &screen, uint8_t a);
void fade_out(Image &screen, uint8_t a);
//*------------------------------------------------------------------------------------------
//! Map class contains everything to build map and keep it ok + methods for working with non-moving objects
class Map {
    
    private:
    Pixel *data = nullptr;
    unsigned char *symmap = nullptr;
    int cnt;
    int height, width;
    int part = 0;
    struct Player_Pos ppos;
    double basic_speed = 5;

    public:
    void put_texture(const Texture &tex, int y, int x, char sym);
    bool read_map(const char *path);
    unsigned char draw_map(int x, int y, Image &screen);
    std::vector<bool> wallcheck(int x, int y);

    Map(){}
    Map(const Map &c);
    Map(const char *path, int h, int w);
    ~Map();
    Map &operator=(const Map &c);

    Pixel GetPixel(int x, int y) const;
    Player_Pos GetPos() const;
    int h() const;
    int w() const;
    int get_part() const;

    //? e button_ops
    bool take_chest(int x, int y);
    bool end_level(int x, int y);
    bool end_game(int x, int y);

    //?spike_ops
    private:
    std::vector<std::pair<double, int>> spikes{};
    public:
    bool is_spike(int x, int y, double lastFrame);
    void cooldown_spike(double lastFrame);
    
    //?shots
    private:
    struct Directional_Shot a;
    std::vector<struct Directional_Shot> shots{20, a};
    Directional_Shot player_shot;
    int prev_part;
    public:
    bool make_damage();                 //? Make damage to mobs with a shot
    void proceed_player_shot(Image &screen);
    void make_shot(double mx, double my, int x, int y, int attack);

    //?mob interact
    private:
    std::vector<class Mob>mobs{};
    public:
    //void proceed_distant_shot(int x, int y);
    int necro_prev_part;
    int necro_anim(Image &screen, double delta, int shot_x, int shot_y, int x, int y);
    int check_mobs(Image &screen, int x, int y, double time);
};
//*------------------------------------------------------------------------------------------
class Hud {
    private:
    Pixel *data = nullptr;
    int height = WINDOW_HEIGHT_G, width = 280;
    int off_w = WINDOW_WIDTH_G;
    double hearts = 5;
    int flasks = 10;
    int attack = 1;
    int part = 0;
    public:
    void redraw_hud(Image &screen, int part, Stats &stats);

    void draw_background();
    void draw_hearts();
    void draw_flasks();
    void draw_mini_map_background();
    void draw_mini_map_player_layout(int part);
    void draw_staff(int attack);
    void draw_mana();
    void draw_regen();

    void put_texture(const Texture &tex, int y, int x);
    struct Pixel GetPixel(int x, int y) const;
    Hud();
    ~Hud();
};

//?=============================================================================================


//*------------------------------------------------------------------------------------------
const struct Properties FLOOR1 = {"./resources/txt/floor_1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties FLOOR2 = {"./resources/txt/floor_2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties FLOOR3 = {"./resources/txt/floor_3.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties ESCAPE0 = {"./resources/txt/floor_ladder.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties CHEST0 = {"./resources/txt/chest_full_open_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties CHEST1 = {"./resources/txt/chest_full_open_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties CHEST2 = {"./resources/txt/chest_full_open_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties DOOR0 = {"./resources/txt/doors_leaf_closed.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties DOOR1 = {"./resources/txt/doors_leaf_open.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties CHORT0 = {"./resources/txt/chort_idle_anim_f0.txt", 24 * Multiplier, 16 * Multiplier};
const struct Properties CHORT1 = {"./resources/txt/chort_idle_anim_f1.txt", 24 * Multiplier, 16 * Multiplier};
const struct Properties CHORT2 = {"./resources/txt/chort_idle_anim_f2.txt", 24 * Multiplier, 16 * Multiplier};
const struct Properties CHORT3 = {"./resources/txt/chort_idle_anim_f3.txt", 24 * Multiplier, 16 * Multiplier};

const struct Properties KNIGHT0 = {"./resources/txt/knight_m_idle_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties KNIGHT1 = {"./resources/txt/knight_m_idle_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties KNIGHT2 = {"./resources/txt/knight_m_idle_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties KNIGHT3 = {"./resources/txt/knight_m_idle_anim_f3.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties WALL0 = {"./resources/txt/wall_mid.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties WALL1 = {"./resources/txt/wall_column_mid.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties WALL2 = {"./resources/txt/wall_inner_corner_mid_right.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties COLUMN0 = {"./resources/txt/column_mid.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties ETRAP0 = {"./resources/txt/skull.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties SPIKE0 = {"./resources/txt/floor_spikes_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties SPIKE1 = {"./resources/txt/floor_spikes_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties SPIKE2 = {"./resources/txt/floor_spikes_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties SPIKE3 = {"./resources/txt/floor_spikes_anim_f3.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties HERO_IDLE1 = {"./resources/txt/wizzard_m_idle_anim_f2.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_IDLE2 = {"./resources/txt/wizzard_m_idle_anim_f3.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_RUN0 = {"./resources/txt/wizzard_m_run_anim_f0.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_RUN1 = {"./resources/txt/wizzard_m_run_anim_f1.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_RUN2 = {"./resources/txt/wizzard_m_run_anim_f2.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_RUN3 = {"./resources/txt/wizzard_m_run_anim_f3.txt", 28 * Multiplier, 16 * Multiplier};
const struct Properties HERO_IDLE0 = {"./resources/txt/wizzard_m_idle_anim_f0.txt", 28 * Multiplier, 16 * Multiplier};

const struct Properties HEART0 = {"./resources/txt/ui_heart_full.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties HEART1 = {"./resources/txt/ui_heart_half.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties HEART2 = {"./resources/txt/ui_heart_empty.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties HP_REGEN0 = {"./resources/txt/hp_regen.txt", 24 * Multiplier, 24 * Multiplier};

const struct Properties MANA0 = {"./resources/txt/mana.txt", 32 * Multiplier, 32 * Multiplier};

const struct Properties BOARDER0 = {"./resources/txt/wall_inner_corner_t_top_left.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties BOARDER1 = {"./resources/txt/wall_inner_top.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties BOARDER2 = {"./resources/txt/wall_side_mid_right.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties BOARDER3 = {"./resources/txt/wall_side_mid_left.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties STAFF0 = {"./resources/txt/weapon_red_magic_staff.txt", 30 * Multiplier, 8 * Multiplier};
const struct Properties STAFF1 = {"./resources/txt/weapon_green_magic_staff.txt", 30 * Multiplier, 8 * Multiplier};
const struct Properties STAFF2 = {"./resources/txt/weapon_super_magic_staff.txt", 30 * Multiplier, 8 * Multiplier};

const struct Properties SHOT0 = {"./resources/txt/red_shot.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties SHOT1 = {"./resources/txt/green_shot.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties SHOT2 = {"./resources/txt/golden_shot.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties WOUND0 = {"./resources/txt/angry.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties FLASK0 = {"./resources/txt/flask_red.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties FLASK1 = {"./resources/txt/flask_green.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties FLASK2 = {"./resources/txt/flask_big_yellow.txt", 16 * Multiplier, 16 * Multiplier};
//? Mobs===========
const struct Properties GOBLIN0 = {"./resources/txt/goblin_idle_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties GOBLIN1 = {"./resources/txt/goblin_idle_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties GOBLIN2 = {"./resources/txt/goblin_idle_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties GOBLIN3 = {"./resources/txt/goblin_run_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties GOBLIN4 = {"./resources/txt/goblin_run_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties GOBLIN5 = {"./resources/txt/goblin_run_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties IMP0 = {"./resources/txt/imp_idle_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties IMP1 = {"./resources/txt/imp_idle_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties IMP2 = {"./resources/txt/imp_idle_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties IMP3 = {"./resources/txt/imp_run_anim_f0.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties IMP4 = {"./resources/txt/imp_run_anim_f1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties IMP5 = {"./resources/txt/imp_run_anim_f2.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties WOGOL0 = {"./resources/txt/wogol_idle_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties WOGOL1 = {"./resources/txt/wogol_idle_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties WOGOL2 = {"./resources/txt/wogol_idle_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties WOGOL3 = {"./resources/txt/wogol_run_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties WOGOL4 = {"./resources/txt/wogol_run_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties WOGOL5 = {"./resources/txt/wogol_run_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};

const struct Properties NECROMANCER0 = {"./resources/txt/necromancer_idle_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties NECROMANCER1 = {"./resources/txt/necromancer_idle_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties NECROMANCER2 = {"./resources/txt/necromancer_idle_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties NECROMANCER3 = {"./resources/txt/necromancer_run_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties NECROMANCER4 = {"./resources/txt/necromancer_run_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties NECROMANCER5 = {"./resources/txt/necromancer_run_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};

const struct Properties ORC_SHAMAN0 = {"./resources/txt/orc_shaman_idle_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties ORC_SHAMAN1 = {"./resources/txt/orc_shaman_idle_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties ORC_SHAMAN2 = {"./resources/txt/orc_shaman_idle_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties ORC_SHAMAN3 = {"./resources/txt/orc_shaman_run_anim_f0.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties ORC_SHAMAN4 = {"./resources/txt/orc_shaman_run_anim_f1.txt", 20 * Multiplier, 16 * Multiplier};
const struct Properties ORC_SHAMAN5 = {"./resources/txt/orc_shaman_run_anim_f2.txt", 20 * Multiplier, 16 * Multiplier};

const struct Properties NECRO_SHOT0 = {"./resources/txt/necro_shot.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties NECRO_SHOT1 = {"./resources/txt/necro_shot_1.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties NECRO_SHOT2 = {"./resources/txt/necro_shot_2.txt", 16 * Multiplier, 16 * Multiplier};
const struct Properties NECRO_SHOT3 = {"./resources/txt/necro_shot_3.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties SKULL0 = {"./resources/txt/skull.txt", 16 * Multiplier, 16 * Multiplier};

const struct Properties SLIDE0 = {"./resources/win_slide.txt", WINDOW_HEIGHT, WINDOW_WIDTH};
const struct Properties SLIDE1 = {"./resources/lose_slide.txt", WINDOW_HEIGHT, WINDOW_WIDTH};
const struct Properties SLIDE2 = {"./resources/nlevel_slide.txt", WINDOW_HEIGHT, WINDOW_WIDTH};
//? vectors
const std::vector<Texture> FLOOR{Texture(FLOOR1), Texture(FLOOR2), Texture(FLOOR3)};
const std::vector<Texture> CHEST{Texture(CHEST0), Texture(CHEST1), Texture(CHEST2)};
const std::vector<Texture> SPIKE{Texture(SPIKE0), Texture(SPIKE1), Texture(SPIKE2), Texture(SPIKE3)};
const std::vector<Texture> HERO{Texture(HERO_IDLE0), Texture(HERO_IDLE1), Texture(HERO_IDLE2), Texture(HERO_RUN0), Texture(HERO_RUN1), Texture(HERO_RUN2), Texture(HERO_RUN3)};
const std::vector<Texture> CHORT{Texture(CHORT0), Texture(CHORT1), Texture(CHORT2), Texture(CHORT3)};
const std::vector<Texture> KNIGHT{Texture(KNIGHT0), Texture(KNIGHT1), Texture(KNIGHT2), Texture(KNIGHT3)};
const std::vector<Texture> WALL{Texture(WALL0), Texture(WALL1), Texture(WALL2)};
const std::vector<Texture> COLUMN{Texture(COLUMN0)};
const std::vector<Texture> ETRAP{Texture(ETRAP0)};
const std::vector<Texture> ESCAPE{Texture(ESCAPE0)};
const std::vector<Texture> HEART{Texture(HEART0), Texture(HEART1), Texture(HEART2)};
const std::vector<Texture> BOARDER{Texture(BOARDER0), Texture(BOARDER1), Texture(BOARDER2), Texture(BOARDER3)};
const std::vector<Texture> STAFF{Texture(STAFF0), Texture(STAFF1), Texture(STAFF2)};
const std::vector<Texture> SHOT{Texture(SHOT0), Texture(SHOT1), Texture(SHOT2)};
const std::vector<Texture> FLASK{Texture(FLASK0), Texture(FLASK1), Texture(FLASK2)};
const std::vector<Texture> GOBLIN{Texture(GOBLIN0),Texture(GOBLIN1), Texture(GOBLIN2),Texture(GOBLIN3), Texture(GOBLIN4), Texture(GOBLIN5)};
const std::vector<Texture> IMP{Texture(IMP0),Texture(IMP1), Texture(IMP2),Texture(IMP3), Texture(IMP4), Texture(IMP5)};
const std::vector<Texture> WOGOL{Texture(WOGOL0),Texture(WOGOL1), Texture(WOGOL2),Texture(WOGOL3), Texture(WOGOL4), Texture(WOGOL5)};
const std::vector<Texture> NECROMANCER{Texture(NECROMANCER0),Texture(NECROMANCER1), Texture(NECROMANCER2),Texture(NECROMANCER3), Texture(NECROMANCER4), Texture(NECROMANCER5)};
const std::vector<Texture> ORC_SHAMAN{Texture(ORC_SHAMAN0),Texture(ORC_SHAMAN1), Texture(ORC_SHAMAN2),Texture(ORC_SHAMAN3), Texture(ORC_SHAMAN4), Texture(ORC_SHAMAN5)};
const Texture WOUND(WOUND0);
const Texture SKULL(SKULL0);
const std::vector<Texture> NECRO_SHOT{Texture(NECRO_SHOT0), Texture(NECRO_SHOT1), Texture(NECRO_SHOT2), Texture(NECRO_SHOT3)};
const std::vector<Texture> SLIDE{Texture(SLIDE0), Texture(SLIDE1), Texture(SLIDE2)};
const Texture MANA(MANA0);
const Texture HP_REGEN(HP_REGEN0);
#endif
