#version 330 core
in vec3 FragPos;
in vec3 Normal;
uniform vec3 lightPos;
uniform vec3 viewPos;
uniform vec4 color;
out vec4 FragColor;
void main() {
    vec3 ambient = 0.4 * color.rgb;
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * color.rgb;
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    vec3 specular = 0.5 * spec * vec3(1.0);
    FragColor = vec4(ambient + diffuse + specular, color.a);
}
