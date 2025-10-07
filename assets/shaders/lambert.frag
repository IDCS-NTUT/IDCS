#version 330 core

in vec3 v_world_pos;
in vec3 v_normal;
in vec2 v_uv;

uniform vec3 u_camera_pos;
uniform vec3 u_light_dir;
uniform vec3 u_base_color;
uniform float u_lighting;

out vec4 frag_color;

void main() {
    vec3 base = u_base_color;
    if (u_lighting > 0.5) {
        vec3 normal = normalize(v_normal);
        vec3 light_dir = normalize(-u_light_dir);
        float diffuse = max(dot(normal, light_dir), 0.0);
        float ambient = 0.35;
        base = base * (ambient + diffuse * (1.0 - ambient));
    }
    frag_color = vec4(base, 1.0);
}

