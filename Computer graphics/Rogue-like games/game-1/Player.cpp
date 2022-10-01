#include "Player.h"
#include "Texture.h"

bool Player::Moved() const {
    if(coords.x == old_coords.x && coords.y == old_coords.y)
        return false;
    else
        return true;
}

void Player::ProcessInput(MovementDir dir, std::vector<bool> &wall) {
    int move_dist = move_speed * 1;
    switch(dir) {
        case MovementDir::UP:
        if (!wall[0]) {
            old_coords.y = coords.y;
            coords.y += move_dist;
        }
        break;
        case MovementDir::DOWN:
        if (!wall[1]) {
            old_coords.y = coords.y;
            coords.y -= move_dist;
        }   
        break;
        case MovementDir::LEFT:
        if (!wall[2]) {
            old_coords.x = coords.x;
            coords.x -= move_dist;
        }
        break;
        case MovementDir::RIGHT:
        if (!wall[3]) {
            old_coords.x = coords.x;
            coords.x += move_dist;
        }
        break;
        default:
        break;
    }
}

struct Point Player::get_coords() const {
    return old_coords;
}
int comparator(int step, int value) {
    int x = 0;
    while (value >= x) {
        x += step;
    }
    return x / step - 1;
}
void Player::Next_level_offset(int h_map, int w_map) {
    if (coords.x < 0) {
        coords.x -= WINDOW_WIDTH_G - w_map;
    } else if (coords.x >= WINDOW_WIDTH_G) {
        coords.x += WINDOW_WIDTH_G - w_map;
    }
    if (coords.y < 0) {
        coords.y -= WINDOW_HEIGHT_G - h_map;
    } else if (coords.y >= WINDOW_HEIGHT_G) {
        coords.y += WINDOW_HEIGHT_G - h_map;
    }
}
void Player::Draw(Image &screen, int h_map, int w_map) {
    if(Moved()) {
        //for(int y = old_coords.y; y <= old_coords.y + tileSize; ++y) {
        //    for(int x = old_coords.x; x <= old_coords.x + tileSize; ++x) {
        //        screen.PutPixel(x, y, backgroundColor); 
        //    }
        //}
        bool is_left = false;
        if (old_coords.x >= coords.x) {
            is_left = true;
        }
        old_coords = coords;

        if (coords.x < 0) {
            coords.x -= WINDOW_WIDTH_G - w_map;
        } else if (coords.x >= WINDOW_WIDTH_G) {
            coords.x += WINDOW_WIDTH_G - w_map;
        }
        if (coords.y < 0) {
            coords.y -= WINDOW_HEIGHT_G - h_map;
        } else if (coords.y >= WINDOW_HEIGHT_G) {
            coords.y += WINDOW_HEIGHT_G - h_map;
        }
        
        int pos_mov = comparator(15, moving_cnt) + 3;
        moving_cnt = (moving_cnt + 1) % 60;

        for(int y = coords.y; y < coords.y + HERO[pos_mov].get_h(); ++y) {
            for(int x = coords.x; x < coords.x + HERO[pos_mov].get_w(); ++x) {
                if (HERO[pos_mov].not_empty_pix(x - coords.x, y - coords.y)) {
                    if (x >= 0 && x < WINDOW_WIDTH_G && y >=0 && y < WINDOW_HEIGHT_G)
                        if (is_left) {
                            screen.PutPixel(x, y, HERO[pos_mov].GetPixel(x - coords.x, y - coords.y));
                        } else {
                            screen.PutPixel(HERO[pos_mov].get_w() + 2 * coords.x - x, y, HERO[pos_mov].GetPixel(x - coords.x, y - coords.y));
                        }
                }
            }
        }
        //screen.PutPixel(coords.x, coords.y, Pixel{.g=255});
    } else {


        if (coords.x < 0) {
            coords.x -= WINDOW_WIDTH_G - w_map;
        } else if (coords.x >= WINDOW_WIDTH_G) {
            coords.x += WINDOW_WIDTH_G - w_map;
        }
        if (coords.y < 0) {
            coords.y -= WINDOW_HEIGHT_G - h_map;
        } else if (coords.y >= WINDOW_HEIGHT_G) {
            coords.y += WINDOW_HEIGHT_G - h_map;
        }

        int pos_idle = comparator(40, idle_cnt);
        idle_cnt = (idle_cnt + 1) % 60;

        for(int y = coords.y; y < coords.y + HERO[pos_idle].get_h(); ++y) {
            for(int x = coords.x; x < coords.x + HERO[pos_idle].get_w(); ++x) {
                if (HERO[pos_idle].not_empty_pix(x - coords.x, y - coords.y)) {
                    if (x >= 0 && x < WINDOW_WIDTH_G && y >=0 && y < WINDOW_HEIGHT_G)
                        screen.PutPixel(x, y, HERO[pos_idle].GetPixel(x - coords.x, y - coords.y));
                }
            }
        }
        //screen.PutPixel(coords.x, coords.y, Pixel{.g=255});
    }
}

void Player::change_coords(int x, int y) {
    coords.x = x;
    coords.y = y;
}

double min(double a, double b) {
    return a < b ? a : b;
}
double max(double a, double b) {
    return a < b ? b : a;
}

void Player::collect_reward(int id) {
    switch(id) {
        case(FLASK_ID):
        flasks = min(20, flasks + 3);
        break;
        case(HEART_ID):
        hp = min(hp + 1, 5);
        break;
        case(STAFF_ID):
        attack = 2;
        flasks = max(10, flasks);
        break;
        case (STAFF_S_ID):
        attack = 4;
        flasks = 20;
        break;
        case (MANA_ID):
        flask_cooldown = 1.5;
        break;
        case (HP_REGEN_ID):
        hp_regen = 0.5;
        break;
        default:
        break;
    }
}

Stats Player::get_player_info() {
    return Stats{.hp=hp, .flasks=flasks, .attack=attack, .flask_cd = flask_cooldown, .regen = hp_regen};
}

void Player::get_spike_damage(double time) {
    if (time - last_damage_time >= 5) {
        hp = max(0, hp - 0.5);
        last_damage_time = time;
        wounded_time = time;
    }
}
void Player::get_mob_damage(int dmg, double time) {
    if (time - last_damage_time >= 8) {
        hp = max(0, hp - dmg);
        last_damage_time = time;
        wounded_time = time;
    }
}
bool Player::draw_wounded(double time, Image &screen) {
    if (hp <= 0) {
        return true;
    }
    if (time - wounded_time < 5) {
        for(int y = coords.y; y < coords.y + WOUND.get_h(); ++y) {
            for(int x = coords.x; x < coords.x + WOUND.get_w(); ++x) {
                if (WOUND.not_empty_pix(x - coords.x, y - coords.y)) {
                    if (x >= 0 && x < WINDOW_WIDTH_G && y >=0 && y < WINDOW_HEIGHT_G)
                        screen.PutPixel(x, y, WOUND.GetPixel(x - coords.x, y - coords.y));
                }
            }
        }
    }
    return false;

}

int Player::get_flasks_and_hp (double time) {
    if (time - flask_time >= flask_cooldown) {
        flask_time = time;
        if (flasks < 10) {
            ++flasks;
        }
    }
    if (time - hp_regen_time >= 6) {
        hp_regen_time = time;
        hp = min(5, hp + hp_regen);
    }
    return flasks;
}