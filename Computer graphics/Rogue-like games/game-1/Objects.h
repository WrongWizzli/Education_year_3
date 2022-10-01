#ifndef OBJECTS_H
#define OBJECTS_H

#include "Image.h"
//#include "Texture.h"
#include <vector>

class Mob {
    public:
};
class Melee_Mob {
    int mob_skin;
    std::vector<Texture> skin;

    Melee_Mob(int skin_id = -1){}
};

#endif