#include "common.h"
#include "Image.h"
#include "Player.h"
#include "tiles.h"
#define GLFW_DLL
#include <GLFW/glfw3.h>


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


void OnKeyboardPressed(GLFWwindow* window, int key, int scancode, int action, int mode)
{
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

int processPlayerMovement(Player &player, double time, u_char *char_map, int w)
{ 
  int status = 0;
  if (Input.keys[GLFW_KEY_W])
    player.ProcessInput(MovementDir::UP, char_map, w);
  else if (Input.keys[GLFW_KEY_S])
    player.ProcessInput(MovementDir::DOWN, char_map, w);
  if (Input.keys[GLFW_KEY_A])
    player.ProcessInput(MovementDir::LEFT, char_map, w);
  else if (Input.keys[GLFW_KEY_D])
    player.ProcessInput(MovementDir::RIGHT, char_map, w);
  if (Input.keys[GLFW_KEY_Q]) {
    player.set_attack_time(time);
    attack_monster(player.Coord());
  } else if (Input.keys[GLFW_KEY_E]) {
    status = E_action(char_map, w, player.Coord());
  }
  return status;
}

void OnMouseButtonClicked(GLFWwindow* window, int button, int action, int mods)
{
  if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE)
    Input.captureMouse = !Input.captureMouse;

  if (Input.captureMouse)
  {
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    Input.capturedMouseJustNow = true;
  }
  else
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

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
  std::cout << "press right mouse button to capture/release mouse cursor  "<< std::endl;
  std::cout << "W, A, S, D - movement  "<< std::endl;
  std::cout << "press ESC to exit" << std::endl;
  std::cout << "Q - Melee Attack" << std::endl;
  std::cout << "E - 'Use' Button" << std::endl;

	return 0;
}

int main(int argc, char** argv)
{
	if(!glfwInit())
    return -1;

//	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
//	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
//	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  GLFWwindow*  window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "task1 base project", nullptr, nullptr);
	if (window == nullptr)
	{
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

  Image next_lvl("./resources/swap_level.png");
  Image win_end("./resources/win.png");

  bool lost = false;
  Image lose_end("./resources/loss.png");

	Image screenBuffer(WINDOW_WIDTH, WINDOW_HEIGHT, 4);
  Image map(50 * tileSize, 32 * tileSize, 4);
  u_char char_map[50 * tileSize * 32 * tileSize] = {};

  int status = 0;

  struct Point map_start = read_map(map, char_map, "./resources/map1.txt", 50, 32);
  double conversion_delta = -1;
  int fade_out_steps = 0;
  const double BLACKOUT_TIME = 3;
  const double BLACKOUT_TIME_END = 5;
  const double BLACKOUT_PARAMETER = 20;
  const double BLACKOUT_PARAMETER_END = 10;

	Player player{map_start};

  struct Point coords;
  glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);  GL_CHECK_ERRORS;
  glClearColor(0.0f, 0.0f, 0.0f, 1.0f); GL_CHECK_ERRORS;

  //game loop
	while (!glfwWindowShouldClose(window))
	{
		GLfloat currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

    glfwPollEvents();

    if (!status) {

      status = processPlayerMovement(player, lastFrame, char_map, map.Width());

      coords = player.Coord();
      draw_moved_map(map, screenBuffer, coords);

      if (trap_location(coords, screenBuffer, lastFrame)) {
        player.get_damage(lastFrame);
        if (player.HP() <= 0) {
          lost = true;
          status = 2;
          conversion_delta = lastFrame;
        }
      }
      if (monster_location(coords, screenBuffer, lastFrame)) {
        player.get_damage(lastFrame);
        if (player.HP() <= 0) {
          lost = true;
          status = 2;
          conversion_delta = lastFrame;
        }
      }
      draw_hp(screenBuffer, player.HP());
      player.Draw(screenBuffer, lastFrame);
      
      if (fade_out_steps != 0) {
        blackout(screenBuffer, next_lvl, fade_out_steps);
        fade_out_steps -= 2;
      }
      if (status == 1) {
        map_start = change_map(map, char_map, "./resources/map2.txt", 50, 32);
        conversion_delta = lastFrame;
      } else if (status == 2) {
        conversion_delta = lastFrame;
      }
    } else if (status == 1) {
      if (lastFrame - conversion_delta < BLACKOUT_TIME) {
        blackout(screenBuffer, next_lvl, BLACKOUT_PARAMETER);
        player.set_Coord(map_start);
      } else {
        status = 0; 
        fade_out_steps = 254;
      }
    } else {
      if (lastFrame - conversion_delta < BLACKOUT_TIME_END) {
        if (lost) {
          blackout(screenBuffer, lose_end, BLACKOUT_PARAMETER_END);
        } else {
          blackout(screenBuffer, win_end, BLACKOUT_PARAMETER_END);
        }
      }
    }

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); GL_CHECK_ERRORS;
    glDrawPixels (WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, screenBuffer.Data()); GL_CHECK_ERRORS;

		glfwSwapBuffers(window);
	}

	glfwTerminate();
	return 0;
}
