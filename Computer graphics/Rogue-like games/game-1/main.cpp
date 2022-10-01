#include "common.h"
#include "Image.h"
#include "Player.h"
#include "Texture.h"

//***********************************************************
//!Global Variables >_<

double x_mouse_pos = -1, y_mouse_pos = -1;
bool first_frame = true;
bool end_game = false;
bool end_level = false;
bool lost = false;
double Time_loading = -1;

//***********************************************************
struct InputState
{
    bool keys[1024]{}; //массив состояний кнопок - нажата/не нажата
    GLfloat lastX = 400, lastY = 300; //исходное положение мыши
    bool firstMouse = true;
    bool captureMouse         = true;  // Мышка захвачена нашим приложением или нет?
    bool capturedMouseJustNow = false;
} static Input;


GLfloat deltaTime = 0.0f;
GLfloat lastFrame = 0.0f;


void OnKeyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mode) {
	switch (key)
	{
	case GLFW_KEY_ESCAPE:
		if (action == GLFW_PRESS)
			glfwSetWindowShouldClose(window, GL_TRUE);
		break;
    case GLFW_KEY_1:
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
    break;
    case GLFW_KEY_2:
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    break;
	default:
		if (action == GLFW_PRESS)
      Input.keys[key] = true;
		else if (action == GLFW_RELEASE)
      Input.keys[key] = false;
	}

}
void processPlayerMovement(Map &level, int &ind, Player &player, std::vector<bool> wall) {

    auto crd = player.get_coords();

    if (Input.keys[GLFW_KEY_W])
        player.ProcessInput(MovementDir::UP, wall);
    else if (Input.keys[GLFW_KEY_S])
        player.ProcessInput(MovementDir::DOWN, wall);
    if (Input.keys[GLFW_KEY_A])
        player.ProcessInput(MovementDir::LEFT, wall);
    else if (Input.keys[GLFW_KEY_D])
        player.ProcessInput(MovementDir::RIGHT, wall);
    if (Input.keys[GLFW_KEY_E]) {
        if(level.take_chest(crd.x, crd.y))
            player.collect_reward(give_chest_reward());
        else if (level.end_level(crd.x, crd.y)) {
            first_frame = true;
            end_level = true;
            Time_loading = glfwGetTime();
        } else if(level.end_game(crd.x, crd.y)) {
            Time_loading = glfwGetTime();
            end_game = true;
        }
    }   
}


void OnMouseButtonClicked(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE) {
        glfwGetCursorPos(window, &x_mouse_pos, &y_mouse_pos);
    }
}

void OnMouseMove(GLFWwindow* window, double xpos, double ypos)
{
    if (Input.firstMouse)
    {
        Input.lastX = float(xpos);
        Input.lastY = float(ypos);
        Input.firstMouse = false;
    }

    GLfloat xoffset = float(xpos) - Input.lastX;
    GLfloat yoffset = Input.lastY - float(ypos);

    Input.lastX = float(xpos);
    Input.lastY = float(ypos);
}


void OnMouseScroll(GLFWwindow* window, double xoffset, double yoffset)
{
  // ...
}


int initGL()
{
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
	{
		std::cout << "Failed to initialize OpenGL context" << std::endl;
		return -1;
	}

	std::cout << "Vendor: " << glGetString(GL_VENDOR) << std::endl;
	std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
	std::cout << "Version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

    std::cout << "Controls: "<< std::endl;

    std::cout << "Left mouse button click to attack in a certain direction"<< std::endl;
    std::cout << "W, A, S, D - movement  "<< std::endl;
    std::cout << "E - interact" << std::endl;
    std::cout << "press ESC to exit" << std::endl;

	return 0;
}




  int main(int argc, char** argv) {
    if(!glfwInit())
        return -1;

  //	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  //	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  //	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

    GLFWwindow*  window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "task1 base project", nullptr, nullptr);
    if (window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    glfwMakeContextCurrent(window); 

    glfwSetKeyCallback        (window, OnKeyboardPressed);  
    glfwSetCursorPosCallback  (window, OnMouseMove); 
    glfwSetMouseButtonCallback(window, OnMouseButtonClicked);
    glfwSetScrollCallback     (window, OnMouseScroll);

    if(initGL() != 0) 
       return -1;
    
    //Reset any OpenGL errors which could be present for some reason
    GLenum gl_error = glGetError();
    while (gl_error != GL_NO_ERROR)
       gl_error = glGetError();

    Image screenBuffer(WINDOW_WIDTH, WINDOW_HEIGHT, 4);

    glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);  GL_CHECK_ERRORS;
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f); GL_CHECK_ERRORS;


    //!-----------------------------------------------------------------------------------------------------------

    Map level[2];
    level[0] = Map("./resources/layer1.txt", 38, 56);
    level[1] = Map("./resources/layer2.txt", 36, 70);
    int ind = 0;
    Hud hud;
    auto ppos = level[ind].GetPos();
    Point starting_pos{.x = ppos.x, .y = ppos.y};
    Player player{starting_pos};
    player.Draw(screenBuffer, level[ind].h(), level[ind].w());
    auto crd = player.get_coords();
    Stats player_info;
    int dmg;
    int fadeout_k = 0;
    double fps = 0;
    double fps_time = 0;

    //!-----------------------------------------------------------------------------------------------------------

    //game loop

    while (!glfwWindowShouldClose(window)) {

        GLfloat currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;
        glfwPollEvents();
        if (end_game || lost) {
            if (currentFrame - Time_loading > 10) {
                break;
            } else if (currentFrame - Time_loading > 1.4) {
                draw_endgame_theme(screenBuffer, lost, -29.65 * (currentFrame - Time_loading) + 296.51);            //at 10 seconds we have 0 alpha of black pixel, at 1.4 sec we have 255 alpha
            }
            fade_in(screenBuffer, 10);
        } else if (end_level) {
            if (currentFrame - Time_loading > 1.4) {
                ind++;
                auto ppos = level[ind].GetPos();
                player.change_pos(ppos.x, ppos.y);
                crd.x = ppos.x, crd.y = ppos.y;
                player.Next_level_offset(level[ind].h(), level[ind].w());
                fadeout_k = 255;
                end_level = false;
            } else {
                fade_in(screenBuffer, 10);
            }
        } else {
            crd = player.get_coords();

            processPlayerMovement(level[ind], ind, player, level[ind].wallcheck(crd.x, crd.y));

            level[ind].draw_map(crd.x, crd.y, screenBuffer);

            dmg = level[ind].check_mobs(screenBuffer, crd.x, crd.y, lastFrame);
            if (dmg) {
                player.get_mob_damage(dmg, lastFrame);
            }
            player.Draw(screenBuffer, level[ind].h(), level[ind].w());
            if (!first_frame && level[ind].is_spike(crd.x, crd.y, lastFrame)) {
                player.get_spike_damage(lastFrame);
            }
            level[ind].cooldown_spike(lastFrame);
            level[ind].proceed_player_shot(screenBuffer);

            if (player.get_flasks_and_hp(lastFrame) && x_mouse_pos >=0 && y_mouse_pos >= 0) {
                level[ind].make_shot(x_mouse_pos, y_mouse_pos, crd.x, crd.y, player.get_attack());
                player.consume_flask();
            }
            if(lost = player.draw_wounded(lastFrame, screenBuffer)) {
                Time_loading = lastFrame;
            }
            player_info = player.get_player_info();
            hud.redraw_hud(screenBuffer, level[ind].get_part(), player_info);
            x_mouse_pos = y_mouse_pos = -1;

            first_frame = false;
            if (fadeout_k != 0) {
                fade_out(screenBuffer, fadeout_k);
                fadeout_k = max(fadeout_k - 10, 0);
            }
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); GL_CHECK_ERRORS;

        glDrawPixels (WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, screenBuffer.Data()); GL_CHECK_ERRORS;

        glfwSwapBuffers(window);
    }


    glfwTerminate();
    return 0;
}
