#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
	vec2 glossAndMode;
	vec3 lightPos;
} ubo;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
layout(location = 3) in vec3 inNormals;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;
layout(location = 2) out vec3 fragNormals;
layout(location = 3) out vec3 fragPos;
layout(location = 4) out float gloss;
layout(location = 5) out mat4 camera;
layout(location = 9) out vec3 fragLight;
layout(location = 10) out float mode;

void main() {
	gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = inColor;
	fragNormals = inNormals;
    fragTexCoord = inTexCoord;
	camera = ubo.view;
	gloss = ubo.glossAndMode.x;
	fragLight = (vec4(ubo.lightPos, 1.0) * ubo.model).rgb;
	mode = ubo.glossAndMode.y;
}
