#include "Player.h"
#include <iostream>

bool Player::Moved() const
{
  if(coords.x == old_coords.x && coords.y == old_coords.y)
    return false;
  else
    return true;
}

void Player::ProcessInput(MovementDir dir, u_char *char_map, int w)
{
  int move_dist = move_speed * 1;
  switch(dir)
  {
    case MovementDir::UP:
      if (char_map[(old_coords.y + move_dist + tileSize) * w + old_coords.x + 4] != '#' && char_map[(old_coords.y + move_dist + tileSize) * w + old_coords.x + tileSize - 4] != '#') { 
          old_coords.y = coords.y;
          coords.y += move_dist;
      }
      break;
    case MovementDir::DOWN:
      if (char_map[(old_coords.y - move_dist) * w + old_coords.x + 4] != '#' && char_map[(old_coords.y - move_dist) * w + old_coords.x + tileSize - 4] != '#') { 
        old_coords.y = coords.y;
        coords.y -= move_dist;
      }
      break;
    case MovementDir::LEFT:
      if (char_map[(old_coords.y + 4) * w + old_coords.x - move_dist - 1] != '#' && char_map[(old_coords.y + tileSize - 4) * w + old_coords.x - move_dist - 1] != '#') {
        old_coords.x = coords.x;
        coords.x -= move_dist;
      }
      break;
    case MovementDir::RIGHT:
      if (char_map[(old_coords.y + 4) * w + old_coords.x + move_dist + tileSize] != '#' && char_map[(old_coords.y + tileSize - 4) * w + old_coords.x + move_dist + tileSize] != '#') {
        old_coords.x = coords.x;
        coords.x += move_dist;
      }
      break;
    default:
      break;
  }
}

void Player::Draw(Image &screen, double time)
{
  int ind;
  if (time - attack_time <= 0.3) {
    ind = 3;
  } else if(Moved()) {
    ind = 4 + int(time * 4) % 4;
  } else {
    ind = 8 + int(time * 4) % 4;
  }
  if(Moved()) {
    if (old_coords.x < coords.x) {
      last_side = false;
    } else {
      last_side = true;
    }
  }
  Pixel pixel;
  if (!last_side) {
    for(int y = WINDOW_HEIGHT / 2 + 3; y < WINDOW_HEIGHT / 2 + PLAYER[ind].Height() - 3; ++y)
    {
      for(int x = WINDOW_WIDTH / 2; x < WINDOW_WIDTH / 2 + PLAYER[ind].Width(); ++x)
      {
        pixel = PLAYER[ind].GetPixel(x - WINDOW_WIDTH / 2, PLAYER[ind].Height() - y + WINDOW_HEIGHT / 2);
        if (pixel.a + pixel.r + pixel.g + pixel.b != 0) {
          screen.PutPixel(x, y, pixel);
        }
      }
    }
  } else {
    for(int y =  WINDOW_HEIGHT / 2 + 3; y <  WINDOW_HEIGHT / 2 + PLAYER[ind].Height() - 3; ++y)
    {
      for(int x = WINDOW_WIDTH / 2; x < WINDOW_WIDTH / 2 + PLAYER[ind].Width(); ++x)
      {
        pixel = PLAYER[ind].GetPixel(PLAYER[ind].Width() - x - 1 + WINDOW_WIDTH / 2, PLAYER[ind].Height() - y - 1 + WINDOW_HEIGHT / 2);
        if (pixel.a != 0 && pixel.r + pixel.g + pixel.b != 0) {
          screen.PutPixel(x, y, pixel);
        }
      }
    }
  }
  old_coords = coords;

}