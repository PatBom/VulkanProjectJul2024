#version 450

layout(binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;
layout(location = 2) in vec3 fragNormals;
layout(location = 3) in vec3 fragPos;
layout(location = 4) in float gloss;
layout(location = 5) in mat4 camera;
layout(location = 9) in vec3 fragLight;
layout(location = 10) in float mode;

layout(location = 0) out vec4 outColor;

float specularPower = 1.0;

vec3 ambientColor = vec3(1.0, 1.0, 1.0);
vec3 specularColor = vec3(1.0, 1.0, 1.0);

vec3 lightColor = vec3(1.0, 1.0, 1.0);
float lightAttenuation = 1.0; 

void main() {

	vec3 viewDirection = normalize(vec3(camera[0].x, camera[1].y, camera[2].z) - fragPos);
	vec3 lightDirection = normalize(fragLight - fragPos);
	float distance = length(lightDirection);
	float attenuation = clamp(10.0/distance, 0.0, 1.0);
	vec3 reflection = reflect(-lightDirection, fragNormals);
	
	float lambertian = max(0.0, dot(fragNormals, lightDirection));
	float specAngle = max(0.0, dot(reflection, viewDirection));

	vec3 specular = pow(specAngle, gloss) * specularPower * specularColor;
	
	vec3 lightingModel = lambertian * texture(texSampler, fragTexCoord).rgb + specular;
	vec3 attenuationColor = attenuation * ambientColor;

	outColor = vec4(0.0);

	if(mode == 0.0) {
		outColor = vec4(lightingModel * attenuationColor, 1.0);
	}
	if(mode > 0.9) {
		outColor = texture(texSampler, fragTexCoord);
	}

	//Lambert lighting modes
	//vec3 lambertDiffuse = NdotL * texture(texSampler, fragTexCoord).rgb; //Lambert lighting model
	//vec3 lambertDiffuse = pow(NdotL * 0.5 + 0.5, 2.0) * texture(texSampler, fragTexCoord).rgb; //Half-Lambert lighting model
    //outColor = vec4(vec3(lambertDiffuse * lightAttenuation * lightColor), 1.0);
}
