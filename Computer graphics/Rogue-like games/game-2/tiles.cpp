#include <fstream>
#include "tiles.h"
#include <iostream>
int abs(int x) {
    if (x < 0) {
        return -x;
    }
    return x;
}

//Trap part

struct Trap_parameters {
    double activation_time;
    int x;
    int y;
};

std::vector<struct Trap_parameters> traps{};

void place_screen_tile(Image &screen, const Image &tile, int x_offset, int y_offset) {
    for (int i = 0; i < tile.Height(); ++i) {
        for (int j = 0; j < tile.Width(); ++j) {
            if (WINDOW_WIDTH / 2 - x_offset + j < WINDOW_WIDTH && WINDOW_WIDTH / 2 - x_offset + j >= 0 && WINDOW_HEIGHT / 2 - y_offset + i < WINDOW_HEIGHT && WINDOW_HEIGHT / 2 - y_offset + i >= 0)
                screen.PutPixel(WINDOW_WIDTH / 2 - x_offset + j, WINDOW_HEIGHT / 2 - y_offset + i, tile.GetPixel(j, tile.Height() - i - 1));
        }
    }
}

bool trap_location(Point coords, Image &screen, double time) {

    bool stepped = false;
    struct Point tmp{coords.x + tileSize / 4 - (coords.x + tileSize / 4) % tileSize, 
                     coords.y + tileSize / 4 - (coords.y + tileSize / 4) % tileSize};
    for (int f = 0; f < traps.size(); ++f) {
        if (tmp.x == traps[f].x && tmp.y == traps[f].y) {
            place_screen_tile(screen, TRAP[3], coords.x - traps[f].x, coords.y - traps[f].y);
            traps[f].activation_time = time;
            stepped = true;
        } else if (traps[f].activation_time != -2 && abs(traps[f].x - tmp.x) < WINDOW_WIDTH / 2 + tileSize 
                                                  && abs(traps[f].y - tmp.y) < WINDOW_HEIGHT / 2 + tileSize) {
            double delta = time - traps[f].activation_time;
            if (delta < 1.5) {
                place_screen_tile(screen, TRAP[3 - int(delta * 2) % 3], -traps[f].x + coords.x , -traps[f].y + coords.y);
            } else {
                place_screen_tile(screen, TRAP[0], -traps[f].x + coords.x, -traps[f].y + coords.y);
            }
        }
    }
    return stepped;
}

// Monster class methods + extra monster functions

std::vector<class Monster> monsters{};

void Monster::update_position() {
    if (!right_move) {
        pos.x -= move_speed;
        tile_distance -= move_speed;
        if (tile_distance <= 0) {
            right_move = true;
        }
    } else {
        pos.x += move_speed;
        tile_distance += move_speed;
        if (tile_distance >= max_dist * tileSize) {
            right_move = false;
        }
    }
}

void place_monster(Image &screen, const Image &tile, int x_offset, int y_offset, bool right_move) {
    Pixel pixel;
    if (right_move) {
        for (int i = 0; i < tile.Height(); ++i) {
            for (int j = 0; j < tile.Width(); ++j) {
                if (WINDOW_WIDTH / 2 - x_offset + j < WINDOW_WIDTH && WINDOW_WIDTH / 2 - x_offset + j >= 0 && 
                    WINDOW_HEIGHT / 2 - y_offset + i < WINDOW_HEIGHT && WINDOW_HEIGHT / 2 - y_offset + i >= 0) {
                    pixel = tile.GetPixel(j, tile.Height() - i - 1);
                    if (pixel.a != 0)
                        screen.PutPixel(WINDOW_WIDTH / 2 - x_offset + j, WINDOW_HEIGHT / 2 - y_offset + i, pixel);
                }
            }
        }
    } else {
        for (int i = 0; i < tile.Height(); ++i) {
            for (int j = 0; j < tile.Width(); ++j) {
                if (WINDOW_WIDTH / 2 - x_offset + j < WINDOW_WIDTH && WINDOW_WIDTH / 2 - x_offset + j >= 0 && 
                    WINDOW_HEIGHT / 2 - y_offset + i < WINDOW_HEIGHT && WINDOW_HEIGHT / 2 - y_offset + i >= 0) {
                    pixel = tile.GetPixel(tile.Width() - j - 1, tile.Height() - i - 1);
                    if (pixel.a != 0)
                        screen.PutPixel(WINDOW_WIDTH / 2 - x_offset + j, WINDOW_HEIGHT / 2 - y_offset + i, pixel);
                }
            }
        }
    }
}

bool monster_location(Point coords, Image &screen, double time) {
    bool touched = false;
    struct Point tmp{coords.x - coords.x % tileSize, coords.y - coords.y % tileSize};
    struct Point tmp2{coords.x + tileSize / 4, coords.y + tileSize / 4};
    for (int f = 0; f < monsters.size(); ++f) {
        monsters[f].update_position();
        if (monsters[f].Alive()) {
            if (monsters[f].x() == tmp.x && monsters[f].y() == tmp.y) {
                touched = true;
            } else if (abs(monsters[f].x() - tmp.x) < WINDOW_WIDTH / 2 + tileSize && 
                       abs(monsters[f].y() - tmp.y) < WINDOW_HEIGHT / 2 + tileSize) {
                place_monster(screen, MONSTER[int(time * 4) % 4], coords.x - monsters[f].x(), coords.y - monsters[f].y(), monsters[f].Last_side());
                if ((abs(monsters[f].y() - tmp2.y) <= tileSize / 2 || abs(monsters[f].y() - tmp2.y + tileSize) <= tileSize / 2) &&
                    (abs(monsters[f].x() - tmp2.x) <= tileSize / 2 || abs(monsters[f].x() - tmp2.x + tileSize) <= tileSize / 2)) {
                        touched = true;
                }
            }
        }
    }
    return touched;
}

void attack_monster(struct Point coords) {
    struct Point tmp2{coords.x + tileSize / 4, coords.y + tileSize / 4};
    for (int f = 0; f < monsters.size(); ++f) {
        if (monsters[f].Alive()) {
            if ((abs(monsters[f].y() - tmp2.y) <= tileSize / 2 || abs(monsters[f].y() - tmp2.y + tileSize) <= tileSize / 2) &&
                (abs(monsters[f].x() - tmp2.x) <= tileSize / 2 || abs(monsters[f].x() - tmp2.x + tileSize) <= tileSize / 2)) {
                monsters[f].Killed();
                return;
            }
        }
    }
}

//Basic background map processing part
void place_tile(Image &map, u_char *char_map, char c, const Image &tile, int x_offset, int y_offset) {
    for (int i = 0; i < tile.Height(); ++i) {
        for (int j = 0; j < tile.Width(); ++j) {
            map.PutPixel(x_offset + j, y_offset + i, tile.GetPixel(j, tile.Height() - i - 1));
            char_map[(y_offset + i) * map.Width() + x_offset + j] = c;
        }
    }
}

struct Point read_map(Image &map, u_char *char_map, const char *path, int w, int h) {

    std::ifstream input(path);
    Point start;
    u_char c;
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            input >> c;
            if (c =='@') {
                start.x = j * tileSize, start.y = i * tileSize;
                place_tile(map, char_map, '@', TUNNEL, j * tileSize, i * tileSize);
            } else if (c == 'x' || c == 'Q') {
                place_tile(map, char_map, c, TUNNEL, j * tileSize, i * tileSize);
            } else if (c == '#') {
                place_tile(map, char_map, c, WALL, j * tileSize, i * tileSize);
            } else if (c == 'T') {
                place_tile(map, char_map, c, TRAP[0], j * tileSize, i * tileSize);
                traps.push_back(Trap_parameters{-2, j * tileSize, i * tileSize});
            } else if (c == 'E'){
                monsters.push_back(Monster(3 + rand() % 3, Point{j * tileSize, i * tileSize}));
                place_tile(map, char_map, c, FLOOR, j * tileSize, i * tileSize);
            } else {
                place_tile(map, char_map, c, FLOOR, j * tileSize, i * tileSize);
            }
        }
    }
    return start;
}

void draw_moved_map(Image &map, Image& screen, Point &coords) {
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            int true_x = WINDOW_WIDTH / 2 - j + coords.x, true_y = WINDOW_HEIGHT / 2 - i + coords.y;
            if (true_x >= 0 && true_x < map.Width() && true_y >= 0 && true_y < map.Height())
                screen.PutPixel(WINDOW_WIDTH - j - 1, WINDOW_HEIGHT - i - 1, map.GetPixel(true_x, true_y));
            else
                screen.PutPixel(WINDOW_WIDTH - j - 1, WINDOW_HEIGHT - i - 1, backgroundColor);
        }
    }
}

struct Point change_map(Image &map, u_char *char_map, const char *path, int w, int h) {
    monsters.clear();
    traps.clear();
    return read_map(map, char_map, path, w, h);
}

//HP status
void place_semifull_tile(Image &screen, const Image &tile, int x_offset, int y_offset) {
    Pixel pixel;
    for (int i = 0; i < tile.Height(); ++i) {
        for (int j = 0; j < tile.Width(); ++j) {
            if (WINDOW_WIDTH / 2 - x_offset + j < WINDOW_WIDTH && WINDOW_WIDTH / 2 - x_offset + j >= 0 
                && WINDOW_HEIGHT / 2 - y_offset + i < WINDOW_HEIGHT && WINDOW_HEIGHT / 2 - y_offset + i >= 0) {
                pixel = tile.GetPixel(j, tile.Height() - i - 1);
                if (pixel.a != 0) {
                    screen.PutPixel(WINDOW_WIDTH / 2 - x_offset + j, WINDOW_HEIGHT / 2 - y_offset + i, tile.GetPixel(j, tile.Height() - i - 1));
                }
            }
        }
    }
}

void draw_hp(Image &screen, double hp) {
    int full_hp = int(hp);
    int half_hp = (hp - int(hp)) * 2;
    int empty_hp = 3 - full_hp - half_hp;

    for (int i = 0; i < full_hp; ++i) {
        place_semifull_tile(screen, HEART[0], WINDOW_WIDTH / 2 - i * tileSize, -WINDOW_HEIGHT / 2 + tileSize);
    }
    for (int i = 0; i < half_hp; ++i) {
        place_semifull_tile(screen, HEART[1], WINDOW_WIDTH / 2 - (i + full_hp) * tileSize, -WINDOW_HEIGHT / 2 + tileSize);
    }
    for (int i = 0; i < empty_hp; ++i) {
        place_semifull_tile(screen, HEART[2], WINDOW_WIDTH / 2 - (i + full_hp + half_hp) * tileSize, -WINDOW_HEIGHT / 2 + tileSize);
    }
}

//next_level / end_game fucntions

int E_action(u_char *char_map, int w, Point coords) {
    coords.x += tileSize / 4;
    coords.y += tileSize / 4;
    if (char_map[coords.x + w * coords.y] == 'x') {
        return 1;
    } else if (char_map[coords.x + w * coords.y] == 'Q') {
        return 2;
    }
    return 0;
}

void blackout(Image &screen, Image &theme, int a) {
    for (int i = 0; i < WINDOW_HEIGHT; ++i) {
        for (int j = 0; j < WINDOW_WIDTH; ++j) {
            screen.AlphaMix(j, i, theme.GetPixel(j, WINDOW_HEIGHT - i - 1), a);
        }
    }
}