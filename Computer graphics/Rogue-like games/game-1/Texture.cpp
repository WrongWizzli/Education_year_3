#include "Texture.h"
#include <cmath>

template<typename T>
void transform_coord(T &x, T &y, int part, int width, int height) {

    if (part == 1) {
        x += width - WINDOW_WIDTH_G;
    } else if (part == 3) {
        x += width - WINDOW_WIDTH_G;
        y += height - WINDOW_HEIGHT_G;
    } else if (part == 2) {
        y += height - WINDOW_HEIGHT_G;
    }

}

template <typename T>
void back_transform_coord(T &x, T &y, int part, int width, int height) {
    if (part == 1) {
        x -= -WINDOW_WIDTH_G + width;
    } else if (part == 3) {
        x -= width - WINDOW_WIDTH_G;
        y -= height - WINDOW_HEIGHT_G;
    } else if (part == 2) {
        y -= height - WINDOW_HEIGHT_G;
    }
}

template <typename T>
T min(T x, T y) {
    return x > y ? y : x;
}
//*------------------------------------------------------------------------------------
void Texture::read_image(const char *path) {

    int r, g, b, a;

    std::ifstream in(path);
    if (in.is_open()) {
        for (int i = height-1; i >= 0; --i) {
            for (int j = width-1; j >= 0; --j) {
                in >> r >> g >> b >> a;
                data[i * width + j].r = r;
                data[i * width + j].g = g;
                data[i * width + j].b = b;
                data[i * width + j].a = a;
            }
        }    
    }
}
Texture Texture::mirror() const {
    Texture tex(width, height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            tex.PutPixel(width - 1 - j, i, GetPixel(j, i));
        }
    }
    return tex;
}

Texture Texture::flip() const {

    Texture tex(width, height);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            tex.PutPixel(j, height - i - 1, GetPixel(j, i));
        }
    }
    return tex;
}

Texture::Texture(int w, int h) {
    width = w;
    height = h;
    data = new Pixel[width * height] {};
}

Texture::Texture(const struct Properties Obj) {
    this->width = Obj.w, this->height = Obj.h;
    data = new Pixel[width * height] {};
    read_image(Obj.path);
}

Texture::Texture(const Texture &c) {
    if (this->data == nullptr) {
        this->data = new Pixel[c.width * c.height] {};
        this->width = c.width;
        this->height = c.height;
    }
    for (int i = 0; i < c.height * c.width; ++i) {
        data[i] = c.data[i];
    }
}

Texture& Texture::operator=(const Texture &c) {
    if (this == &c) {
        return *this;
    }
    if (data != nullptr) {
        delete[] data;
    }
    data = new Pixel[c.width * c.height] {};
    this->width = c.width, this->height = c.height;
    for (int i = 0; i < height * width; ++i){ 
        data[i] = c.data[i];
    }
    return *this;
}
Texture::~Texture() {
    if (this->data != nullptr)
        delete[] data;
}
    
int Texture::get_h() const {
    return height;
}
int Texture::get_w() const {
    return width;
}

bool Texture::not_empty_pix(int x, int y) const{
    return data[width  * y + x].sum();
}

Pixel Texture::GetPixel(int x, int y) const { 
    return data[width * y + x];
}
void  Texture::PutPixel(int x, int y, const Pixel &pix) { 
    data[width * y + x] = pix; 
}
//*------------------------------------------------------------------------------
std::vector<int> rewards{FLASK_ID, FLASK_ID, HEART_ID, STAFF_ID, FLASK_ID, FLASK_ID, 
                         HEART_ID, FLASK_ID, FLASK_ID, HEART_ID, MANA_ID, FLASK_ID, 
                         FLASK_ID, HEART_ID, FLASK_ID, FLASK_ID, STAFF_S_ID, HEART_ID,
                         FLASK_ID, FLASK_ID, HEART_ID, HP_REGEN_ID, FLASK_ID, HEART_ID, 
                         FLASK_ID, FLASK_ID, HEART_ID, FLASK_ID, FLASK_ID, HEART_ID, 
                         FLASK_ID, FLASK_ID, HEART_ID, FLASK_ID, FLASK_ID, HEART_ID, 
                         FLASK_ID, FLASK_ID, HEART_ID, FLASK_ID, FLASK_ID, HEART_ID, 
                         FLASK_ID, FLASK_ID, HEART_ID, FLASK_ID, FLASK_ID, HEART_ID};
int cnt_reward = 0;

int give_chest_reward() {
    return rewards[cnt_reward++];
}

void draw_next_level_theme(Image &screen) {
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            screen.PutPixel(j, i, SLIDE[2].GetPixel(j, i));
        }
    }
}

void draw_endgame_theme(Image &screen, int lost, uint8_t alpha) {
    Pixel pix = {0, 0, 0, alpha};
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            screen.PutPixel(j, i, SLIDE[lost].GetPixel(j, i));
            screen.PutalphaPixel(j, i, pix);
        }
    }
}

void fade_in(Image &screen, uint8_t a) {
    Pixel fade{0,0,0,a};
    Pixel g;
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            screen.PutalphaPixel(j, i, fade);
        }
    }
}
void fade_out(Image &screen, uint8_t a) {
    Pixel fade{0,0,0,a};
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            screen.PutalphaPixel(j, i, fade);
        }
    }
}
//*------------------------------------------------------------------------------
void Map::put_texture(const Texture &tex, int y, int x, char sym) {

    for (int k = 0; k < tex.get_h(); ++k) {
        for (int m = 0; m < tex.get_w(); ++m) {
            Pixel pix = tex.GetPixel(m, k);
            if (pix.sum() && (y + k) * width + x + m < width * height) {
                data[(y + k) * width + x + m] = tex.GetPixel(m, k);
                symmap[(y + k) * width + x + m] = sym;
            }
        }
    }
}

bool Map::read_map(const char *path) {

    std::ifstream in(path);

    unsigned char sym;

    if (in.is_open()) {
        for (int i = 0; i < height; i += Tile_size) {
            for (int j = 0; j < width; j+=Tile_size) {
                in >> sym;
                switch (sym) {       
                    case '#':
                        put_texture(WALL[0], i, j, sym);
                        break;
                    case 'G':
                        put_texture(FLOOR[1], i, j, sym);                       
                        put_texture(CHEST[0], i, j, sym);
                        break;
                    case 'T':
                        put_texture(SPIKE[0], i, j, sym);
                        spikes.push_back(std::pair<double, int>(0, i * width + j));
                        break;
                    case 'C':
                        put_texture(FLOOR[0], i, j, sym);
                        put_texture(COLUMN[0], i, j, sym);
                        break;
                    case '@':
                        put_texture(FLOOR[0], i, j, sym);
                        put_texture(ESCAPE[0], i, j, sym);
                        ppos.y = i, ppos.x = j;
                        break;
                    case 'x':
                        put_texture(ESCAPE[0], i, j, sym);
                        break;
                    case 'M':
                        mobs.push_back(Mob('M', -1, j, i));
                        put_texture(FLOOR[0], i, j, sym);
                        break;
                    case 'D':
                        mobs.push_back(Mob('D', -1, j, i));
                        put_texture(FLOOR[0], i, j, sym);
                        break;
                    case 'V':
                        put_texture(FLOOR[0], i, j, '.');
                        put_texture(SKULL, i, j, '.');
                        break;
                    case 'Q':
                        put_texture(ESCAPE[0], i, j, sym);
                        break;
                    default:
                        put_texture(FLOOR[0], i, j, sym);
                        break;
                }
            }
        }    
    } else {
        return false;
    }
    return true;
}
Map::Map(const Map &c) {

    if (this->data == nullptr) {
        this->data = new Pixel[c.width * c.height] {};
        this->symmap = new unsigned char[c.width * c.height / Tile_size / Tile_size] {};
        this->width = c.width;
        this->height = c.height;
    }
    //!MAYBE FAIL
    for (int i = 0; i < c.height * c.width; ++i) {
            this->data[i] = c.data[i];
            this->symmap[i] = c.symmap[i];
    }
}


Map& Map::operator=(const Map &c) {
    if (this == &c) {
        return *this;
    }
    if (data != nullptr) {
        delete[] data;
        delete[] symmap;
    }
    data = new Pixel[c.width * c.height] {};
    symmap = new unsigned char[c.width * c.height] {};
    width = c.width, height = c.height;
    cnt = c.cnt;
    part = c.part;
    ppos = c.ppos;
    basic_speed = c.basic_speed;
    spikes = c.spikes;
    a = c.a;
    shots = c.shots;
    player_shot = c.player_shot;
    prev_part = c.prev_part;
    mobs=c.mobs;
    for (int i = 0; i < width * height; ++i) {
        data[i] = c.data[i];
        symmap[i] = c.symmap[i];
    }
    return *this;
}


Map::Map(const char *path, int h, int w) {
    height = 16 * Multiplier * h;
    width = 16 * Multiplier * w;
    data = new Pixel[Multiplier * Multiplier * 16 * 16 * w * h] {};
    symmap = new unsigned char[16 * 16 * Multiplier * Multiplier * w * h] {};
    if (!read_map(path)) {
        std::cout << "Error occured when loading map...\n";
    }
}

Map::~Map() {
    if (data != nullptr)
        delete[] data;
    if (symmap != nullptr)
        delete[] symmap;
}

Pixel Map::GetPixel(int x, int y) const { 
    if (width * y + x < width * height && width * y + x >= 0)
        return data[width * y + x];
    struct Pixel a{.r=0, .g=0, .b=0, .a=0};
    return a;
}

Player_Pos Map::GetPos() const {
    return ppos;
}

int Map::w() const {
    return width;
}

int Map::h() const {
    return height;
}

unsigned char Map::draw_map(int x, int y, Image &screen) {

    Pixel background={0, 0, 0, 0};
    Player_Pos new_coord{x, y};
    int off_x = 0, off_y = 0;
    if (x < 0) {
        part -= 1;
        new_coord.x = WINDOW_WIDTH_G + x;
    } else if (x >= WINDOW_WIDTH_G) {
        part += 1;
        new_coord.x = x - WINDOW_WIDTH_G;
    }
    if (y < 0) {
        part -= 2;
        new_coord.y = y + WINDOW_HEIGHT_G;
    } else if (y >= WINDOW_HEIGHT_G) {
        part += 2;
        new_coord.y = y - WINDOW_HEIGHT_G;
    }
    if (part == 1) {
        off_x = width - WINDOW_WIDTH_G;
    } else if (part == 3) {
        off_x = width - WINDOW_WIDTH_G;
        off_y = height - WINDOW_HEIGHT_G;
    } else if (part == 2) {
        off_y = height - WINDOW_HEIGHT_G;
    }
    for (int i = 0; i < WINDOW_HEIGHT_G; ++i) {
        for (int j = 0; j < WINDOW_WIDTH_G; ++j) {
            screen.PutPixel(j, i, GetPixel(j + off_x, i + off_y));
        }
    }
    return 0;
}
std::vector<bool> Map::wallcheck(int x, int y) {
    
    std::vector<bool> wall{false, false, false, false};

    if (part == 1) {
        x += width - WINDOW_WIDTH_G;
    } else if (part == 3) {
        x += width - WINDOW_WIDTH_G;
        y += height - WINDOW_HEIGHT_G;
    } else if (part == 2) {
        y += height - WINDOW_HEIGHT_G;
    }

    int pos = y * width + x;

    if (symmap[pos + 1 + Tile_size] == '#' || symmap[pos + 1 + Tile_size] == 'C') {
        wall[3] = true;
    }
    if (symmap[pos - 1] == '#' || symmap[pos - 1] == 'C') {
        wall[2] = true;
    }
    if (symmap[pos + width * Tile_size + 4] == '#' || symmap[pos + width * Tile_size + 4] == 'C' ||
        symmap[pos + width * Tile_size + Tile_size - 5] == '#' || symmap[pos + width * Tile_size + Tile_size - 5] == 'C') {
        wall[0] = true;
    }
    if (symmap[pos - width + 1] == '#' || symmap[pos - width + 1] == 'C' || 
        symmap[pos - width + Tile_size - 2] == '#' || symmap[pos - width + Tile_size - 2] == 'C') {
        wall[1] = true;
    }
    return wall;
}

int Map::get_part() const {
    return part;
}

bool Map::take_chest(int x, int y) {
    int off_x = 0, off_y = 0;

    transform_coord(x, y, part, width, height);

    int pos = (y + Tile_size / 2) * width + x + Tile_size / 2;

    if (pos < width * height && symmap[pos] == 'G') {
        int y0 = (y + Tile_size / 2) - (y + Tile_size / 2) % Tile_size;
        int x0 = x + Tile_size / 2 - (x + Tile_size / 2) % Tile_size;
        put_texture(CHEST[2], y0, x0, 'g');
        return true;
    }
    return false;
}

bool Map::end_level(int x, int y) {
    int off_x = 0, off_y = 0;

    transform_coord(x, y, part, width, height);

    int pos = (y + Tile_size / 2) * width + x + Tile_size / 2;

    if (pos < width * height && symmap[pos] == 'x') {
        return true;
    }
    return false;
}

bool Map::end_game(int x, int y) {
    int off_x = 0, off_y = 0;

    transform_coord(x, y, part, width, height);

    int pos = (y + Tile_size / 2) * width + x + Tile_size / 2;

    if (pos < width * height && symmap[pos] == 'Q') {
        return true;
    }
    return false;
}

bool Map::is_spike(int x, int y, double lastFrame) {

    transform_coord(x, y, part, width, height);
    int pos1 = y * width + x;
    int pos2 = y * width + x + Tile_size - 1;
    if (symmap[pos1] == 'T') {
        y -= y % Tile_size;
        x -= x % Tile_size;
        put_texture(SPIKE[3], y, x, 'T');
        for (int i = 0; i < spikes.size(); ++i) {
            if (spikes[i].second == y * width + x) {
                spikes[i].first = lastFrame;
                break;
            }
        }
        return true;
    }
    if (symmap[pos2] == 'T') {
        y -= y % Tile_size;
        x = x + Tile_size - 1 - (x + Tile_size - 1) % Tile_size;
        put_texture(SPIKE[3], y, x, 'T');
        for (int i = 0; i < spikes.size(); ++i) {
            if (spikes[i].second == y * width + x) {
                spikes[i].first = lastFrame;
                break;
            }
        }
        return true;
    }
    return false;
}

void Map::cooldown_spike(double lastFrame) {

    for (int i = 0; i < spikes.size(); ++i) {
        if (spikes[i].first != 0) {
            double delta = lastFrame - spikes[i].first;
            if (delta <= 1.6) {
                put_texture(SPIKE[3], spikes[i].second / width, spikes[i].second % width, 'T');
            } else if (delta <= 3.50) {
                put_texture(SPIKE[2], spikes[i].second / width, spikes[i].second % width, 'T');
            } else if (delta <= 5.00) {
                put_texture(SPIKE[1], spikes[i].second / width, spikes[i].second % width, 'T');
            } else {
                put_texture(SPIKE[0], spikes[i].second / width, spikes[i].second % width, 'T');
                spikes[i].first = 0;
            }
        }
    }
}
double abs(double x) {
    return x > 0 ? x: -x;
}
double sgn(double x) {
    return x > 0 ? 1: -1;
}
void Map::make_shot(double mx, double my, int x, int y, int attack) {

    my = WINDOW_HEIGHT_G - my;
    prev_part = part;
    player_shot = {.x=(double)x, .y=(double)y, .kx=mx - x, .ky = my-y, .damage=attack};

    if (player_shot.ky) {
        double sq = sqrt((mx - x) * (mx - x) + (my - y) * (my - y));
        player_shot.ky = (my - y) / sq;
        player_shot.kx = (mx - x) / sq;
    } else if (player_shot.kx) {
        player_shot.kx = 1;
        player_shot.ky = 0;
    } else {
        player_shot.x = player_shot.y = player_shot.kx = player_shot.ky = player_shot.damage = -1;
    }
}
bool Map::make_damage() {
    for (int i = 0; i < mobs.size(); ++i) {
        if (mobs[i].get_hp() > 0) {
            int tr_x = mobs[i].get_x(), tr_y = mobs[i].get_y();
            back_transform_coord(tr_x, tr_y, part, width, height);
            if (player_shot.x >= tr_x - Tile_size / 2 && player_shot.x < tr_x + Tile_size / 2 && player_shot.y >= tr_y - Tile_size / 2 && player_shot.y < tr_y + Tile_size / 2) {
                mobs[i].reduce_hp(player_shot.damage);
                return true;
            }
        }
    }
    return false;
}
void Map::proceed_player_shot(Image &screen) {

    if (player_shot.x == -1 || player_shot.y == -1) {
        return;
    }

    player_shot.x += player_shot.kx * basic_speed;
    player_shot.y += player_shot.ky * basic_speed;
    double tmp_x = player_shot.x, tmp_y = player_shot.y;
    transform_coord(tmp_x, tmp_y, part, width, height);

    int pos = ((int)tmp_y + Tile_size / 2) * width + ((int)tmp_x + Tile_size / 2);
    if (make_damage()) {
        prev_part = part;
        player_shot.x = player_shot.y = player_shot.kx = player_shot.ky = player_shot.damage = -1;
        return;
    }
    if (symmap[pos] != '#' && symmap[pos] != 'C' && prev_part == part) {
        for (int m = 0; m < Tile_size; ++m) {
            for (int k = 0; k < Tile_size; ++k) {
                Pixel pix = SHOT[player_shot.damage / 2].GetPixel(k, m);
                if (pix.sum() && ((int)player_shot.y + m) * width + (int)player_shot.x + k < width * height) {
                    screen.PutPixel((int)player_shot.x + k, (int)player_shot.y + m, pix);
                } else if ((int)player_shot.y + m >= WINDOW_HEIGHT_G || (int)player_shot.x + k >= WINDOW_WIDTH_G || (int)player_shot.x + k < 0 || (int)player_shot.y + m < 0) {
                    player_shot.x = player_shot.y = player_shot.kx = player_shot.ky = player_shot.damage = -1;
                    return;
                }
            }
        }
    } else {
        prev_part = part;
        player_shot.x = player_shot.y = player_shot.kx = player_shot.ky = player_shot.damage = -1;
    }
}

const int SEARCH_DIST = 7;
int sfield[SEARCH_DIST][SEARCH_DIST] = {};
int Map::necro_anim(Image &screen, double delta, int shot_x, int shot_y, int x, int y) {
    if (delta > 3 || part != necro_prev_part) {
        return 0;
    }
    int dmg = 0;
    if (x + Tile_size / 2 >= shot_x && x + Tile_size / 2 < shot_x + Tile_size && y + Tile_size / 2 > shot_y && y + Tile_size / 2 < shot_y + Tile_size && delta >= 2.25) {
        dmg += 2;
    }
    Texture tex = NECRO_SHOT[int(4 * delta / 3) % 4];
    for (int k = 0; k < tex.get_h(); ++k) {
        for (int m = 0; m < tex.get_w(); ++m) {
            Pixel pix = tex.GetPixel(m, k);
            if (pix.sum() && shot_x + m >= 0 && shot_x + m < WINDOW_WIDTH_G && shot_y + k >= 0 && shot_y + k < WINDOW_HEIGHT_G) {
                screen.PutPixel(shot_x + m, shot_y + k, tex.GetPixel(m, k));
            }
        }
    }
    return dmg;
}
int Map::check_mobs(Image &screen, int x, int y, double time) {
    int dmg = 0;
    for (int i = 0; i < mobs.size(); ++i) {
        int tr_x = mobs[i].get_x(), tr_y = mobs[i].get_y();
        if (mobs[i].get_hp() > 0) {
            back_transform_coord(tr_x, tr_y, part, width, height);
            if (tr_x >= 0 && tr_x < WINDOW_WIDTH_G && tr_y >= 0 && tr_y < WINDOW_HEIGHT_G) {
                if (mobs[i].detect_player(tr_x, tr_y, x, y)) { 
                    if (mobs[i].get_type() == 'D') {
                        if(time - mobs[i].last_shot_frame > 3) {
                            mobs[i].last_shot_frame = time;
                            mobs[i].shot_x = x;
                            mobs[i].shot_y = y;
                            necro_prev_part = part;
                        }
                        mobs[i].proceed_move(symmap, tr_x, tr_y, x, y, width, -1);
                    }
                    if (mobs[i].get_type() == 'M') {
                        dmg += mobs[i].proceed_move(symmap, tr_x, tr_y, x, y, width, 1);
                        int mob_x = mobs[i].get_x(), mob_y = mobs[i].get_y();
                        if (mob_x >= WINDOW_WIDTH_G || mob_y >= WINDOW_HEIGHT_G || mob_y < 0 || mob_x < 0) {
                            mobs[i].change_state('i');
                        }
                    }
                }
                dmg += necro_anim(screen, time - mobs[i].last_shot_frame, mobs[i].shot_x, mobs[i].shot_y, x, y);
                mobs[i].mob_animation(screen, time, width, height, tr_x, tr_y);
            }
        }
    }
    return min(2, dmg);
}
//TODO----------------------------------------------------------------------------------------------------------------------------------

void Hud::put_texture(const Texture &tex, int y, int x) {

    for (int k = 0; k < tex.get_h(); ++k) {
        for (int m = 0; m < tex.get_w(); ++m) {
            Pixel pix = tex.GetPixel(m, k);
            if (pix.sum() && (y + k) * width + x + m < width * height) {
                data[(y + k) * width + x + m] = tex.GetPixel(m, k);
            }
        }
    }
}

Hud::Hud() {
    data = new Pixel[width * height] {};

    draw_background();
    draw_hearts();
    draw_mini_map_background();
    draw_mini_map_player_layout(part);
    draw_staff(attack);
    draw_flasks();
    
}

Hud::~Hud() {
    delete[] data;
}

struct Pixel Hud::GetPixel(int x, int y) const { 
    if (width * y + x < width * height && width * y + x >= 0)
        return data[width * y + x];
    struct Pixel a{.r=0, .g=0, .b=0, .a=0};
    return a;
}

void Hud::redraw_hud(Image &screen, int part, Stats &stats) {
    //changning layout on mini_map
    draw_background();
    if (this->part != part) {
        this->part = part;
    }
    if (this->hearts != stats.hp) {
        this->hearts = stats.hp;
    }
    if (this->attack != stats.attack) {
        this->attack = stats.attack;
    }
    if (this->flasks != stats.flasks) {
        this->flasks = stats.flasks;
    }
    if (stats.flask_cd < 3) {
        draw_mana();
    }
    if (stats.regen) {
        draw_regen();
    }
    draw_mini_map_background();
    draw_mini_map_player_layout(part);
    draw_hearts();
    draw_staff(attack);
    draw_flasks();
    for (int i = 0; i < height ; ++i) {
        for (int j = off_w; j < off_w + width; ++j) {
            screen.PutPixel(j, i, GetPixel(j-off_w, i));
        }
    }
}

void Hud::draw_hearts(){
    int whole = hearts;
    for (int j = 40; j < (whole + 1) * 40; j += 40) {
        put_texture(HEART[0], height - 64, j);
    }
    ++whole;
    if (hearts - whole + 1 != 0) {
        put_texture(HEART[1], height - 64, (whole) * 40);
        ++whole;
    }
    for (int j = whole * 40; j < 240; j += 40) {
        put_texture(HEART[2], height - 64, j);
    }
}

void Hud::draw_flasks() {
    int wf = flasks / 5;
    int rf = flasks % 5;

    for (int full_line = 0; full_line < wf; ++full_line) {
        for (int j = 40; j < 240; j+= 40) {
            put_texture(FLASK[attack / 2], height - 96 - 32 * full_line, j);
        }
    }
    for (int j = 40; j < 40 * (rf + 1); j += 40) {
        put_texture(FLASK[attack / 2], height - 96 - 32 * wf, j);
    }
}

void Hud::draw_background() {
    for (int i = 0; i < height; i += Tile_size) {
        for (int j = 0; j < off_w; j += Tile_size) {
            put_texture(WALL[1], i, j);
        }
    }
    for (int j = 0; j < width; j += Tile_size) {
        put_texture(BOARDER[0], 0, j);
        put_texture(BOARDER[0], height - 8, j);
    }
    for (int i = 0; i < height; i += Tile_size) {
        put_texture(BOARDER[3], i, 0);
        put_texture(BOARDER[3], i, 8);
        put_texture(BOARDER[2], i, width - Tile_size);
        put_texture(BOARDER[2], i, width - Tile_size - 8);
    }
}
void Hud::draw_mini_map_background(){
    for (int i = Tile_size; i < 5 * Tile_size; i += Tile_size) {
        for (int j = 44; j < 6 * Tile_size + 44; j += Tile_size) {
            put_texture(FLOOR[1], i, j);
        }
    }
    for (int j = 44; j < 6 * Tile_size + 44; j += Tile_size) {
        put_texture(BOARDER[0], 5 * Tile_size, j);
        put_texture(BOARDER[0], Tile_size, j);
        put_texture(BOARDER[0], 3 * Tile_size, j);
    }
    for (int i = Tile_size; i < 5 * Tile_size; i += Tile_size) {
        put_texture(BOARDER[3], i, 44);
        put_texture(BOARDER[2], i, 44 + 5 * Tile_size);
        put_texture(BOARDER[2], i, 44 + 2 * Tile_size);
        put_texture(BOARDER[3], i, 44 + 3 * Tile_size);
    }
}
void Hud::draw_mini_map_player_layout(int part){
    int off_x, off_y;
    switch (part) {
        case 0:
        off_x = Tile_size + Tile_size / 2;
        off_y = 44 + Tile_size;
        break;
        case 2:
        off_x = 7 * Tile_size / 2;
        off_y = 44 + Tile_size;
        break;
        case 1:
        off_x = Tile_size + Tile_size / 2;
        off_y = 44 +  4 * Tile_size;
        break;
        default:
        off_x = 7 * Tile_size / 2;
        off_y = 44 +  4 * Tile_size;
    }
    put_texture(HERO[0], off_x, off_y);
}

void Hud::draw_staff(int attack) {
    put_texture(STAFF[attack / 2], height - 144 - Multiplier * STAFF[attack / 2].get_h(), 45);
}

void Hud::draw_mana() {
    put_texture(MANA, height - 144 - Multiplier * STAFF[0].get_h(), 85);
}

void Hud::draw_regen() {
    put_texture(HP_REGEN, height - 144 - Multiplier * STAFF[0].get_h(), 145);
}

//TODO Mobs
Mob::Mob(unsigned char mob_type, int skin_id, int x, int y) {
    this->mob_x = x, this->mob_y = y;
    this->mob_type = mob_type;
    if (skin_id >= 0 && skin_id < 3) {
        this->mob_skin = skin_id;
    } else {
        this->mob_skin = rand() % 3;
    }
    if (mob_type == 'M') {
        hp = 3;
        attack = 1;
        if (!mob_skin) {
            skin = WOGOL;
        } else if (mob_skin == 1) {
            skin = IMP;
        } else {
            skin = GOBLIN;
        }
    } else if (mob_type == 'D') {
        hp = 2;
        attack = 2;
        if (!mob_skin) {
            skin = NECROMANCER;
        } else {
            skin = NECROMANCER;
        }
    }
}

int Mob::proceed_move(unsigned char *symmap, int tr_x, int tr_y, int x, int y, int width, int type) {
    struct Dir_k k;
    
    if (tr_y - y || tr_x - x) {
        double sq = sqrt((tr_x - x) * (tr_x - x) + (tr_y - y) * (tr_y - y));
        k.ky = (tr_y - y) / sq;
        k.kx = (tr_x - x) / sq;
    } else {
        return 1;
    }
    
    unsigned char cell = symmap[mob_x + Tile_size / 2 - int(basic_mob_speed * k.kx * type) + width * (mob_y + Tile_size / 2 - int(basic_mob_speed * k.ky * type))];
    unsigned char cell_cur = symmap[mob_x + Tile_size / 2 + width * (mob_y + Tile_size / 2)];
    if (cell != '#' && cell != 'C') {
        mob_x -= basic_mob_speed * k.kx * type;
        mob_y -= basic_mob_speed * k.ky * type;
    }
    if (type == 1 && x >= tr_x - Tile_size / 2 && x < tr_x + Tile_size / 2 && y >= tr_y - Tile_size / 2 && y < tr_y + Tile_size / 2) {
        return 1;
    }
    return 0;
}
bool Mob::detect_player(int tr_x, int tr_y, int x, int y) {

    double dist = abs(tr_x - x) + abs(tr_y - y);
    if (dist < Tile_size * 6) {
        state = 'm';
        return true;
    }
    state = 'i';
    return false;
}

void Mob::mob_animation(Image &screen, double time, int w, int h, int tr_x, int tr_y) {
    Texture tex;
    if (state == 'i') {
        tex = skin[(int)(6 * time) % 3];
    } else {
        tex = skin[(int)(6 * time) % 3 + 3];
    }
    for (int k = 0; k < tex.get_h(); ++k) {
        for (int m = 0; m < tex.get_w(); ++m) {
            Pixel pix = tex.GetPixel(m, k);
            if (pix.sum() && tr_x + m >= 0 && tr_x + m < WINDOW_WIDTH_G && tr_y + k >= 0 && tr_y + k < WINDOW_HEIGHT_G) {
                screen.PutPixel(tr_x + m, tr_y + k, tex.GetPixel(m, k));
            }
        }
    }
}
