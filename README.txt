Source code for my Vulkan project.

For setting up an environment in Visual studio, some dependencies need to be installed and set up following Vulkan Tutorial.
https://vulkan-tutorial.com/Development_environment

Dependencies:
	Vulkan SDK:
		https://vulkan.lunarg.com/
	GLFW:
		https://www.glfw.org/download.html
	GLM:
		https://github.com/g-truc/glm/releases
	stb image library:
		https://github.com/nothings/stb/blob/master/stb_image.h
	tinyobj object loader:
		https://github.com/tinyobjloader/tinyobjloader/blob/release/tiny_obj_loader.h

Shader compiler batch file (shaders/compile.bat) contains a path to local Vulkan installation that needs to be changed if wanting to recompile
the program's shaders.