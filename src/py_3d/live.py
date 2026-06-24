"""Interactive live presentation helpers for py_3d demos."""

from __future__ import annotations

from array import array
from dataclasses import dataclass, field
from math import asin, atan2, cos, degrees, radians, sin, sqrt, tan
from time import perf_counter, sleep
from typing import Iterable
from weakref import ref

from .assets import load_framework_icon_rgba
from .buffer import PixelBuffer
from .camera import Camera
from .color import Color
from . import draw
from .gpu import GPURenderer
from .hud import HUDAnimation, HUDImage, HUDRect, HUDText, LiveHUD
from .lights import Lamp, Sun
from .materials import Material
from .math3d import Vec3
from .overlays import FloatingTextBulletin, TextBulletin
from .primitives import Bowl, Box, Capsule, Line3, LineSet3, Mesh, ParticleFluidPuddle, ParticleFluidVolume, ParticleWaterSurface, Plane, Point3, Sphere, TransformedMesh, Triangle
from .render import RenderEngine, RenderSettings, _triangles_for
from .scene import Scene
from .window import PixelWindow, WindowEvent


VERTEX_SHADER = """
#version 330

in vec3 in_position;
in vec3 in_normal;
in vec3 in_color;
in vec3 in_emission;
in vec4 in_material;
in vec2 in_texcoord;
in float in_texture_index;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_focal;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;

out vec3 v_world_position;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_emission;
out vec4 v_material;
out vec2 v_texcoord;
out float v_texture_index;

void main() {
    vec3 relative = in_position - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float ndc_z = -1.0 + 2.0 * ((camera_z - u_near) / max(0.0001, u_far - u_near));

    gl_Position = vec4(camera_x * u_focal / u_aspect, camera_y * u_focal, ndc_z * camera_z, camera_z);
    v_world_position = in_position;
    v_normal = in_normal;
    v_color = in_color;
    v_emission = in_emission;
    v_material = in_material;
    v_texcoord = in_texcoord;
    v_texture_index = in_texture_index;
}
"""


FRAGMENT_SHADER = """
#version 330

const int MAX_LIGHTS = 8;

in vec3 v_world_position;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_emission;
in vec4 v_material;
in vec2 v_texcoord;
in float v_texture_index;

uniform vec3 u_camera_position;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform bool u_two_sided_lighting;
uniform int u_reflection_bounces;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform bool u_use_textures;
uniform int u_texture_count;
uniform sampler2DArray u_textures;

out vec4 frag_color;

vec3 reflectionProbe(vec3 direction) {
    vec3 ray = normalize(direction);
    float sky = max(0.0, ray.y);
    float floor_light = max(0.0, -ray.y);
    float horizon = max(0.0, 1.0 - abs(ray.y));
    float sun_dot = max(0.0, dot(ray, normalize(vec3(-0.35, 0.72, -0.58))));
    float sun_core = pow(sun_dot, 180.0);
    float sun_bloom = pow(sun_dot, 24.0);
    vec3 sky_color = vec3(0.045, 0.070, 0.110) + vec3(0.25, 0.38, 0.62) * sky;
    vec3 ground_color = vec3(0.060, 0.065, 0.060) + vec3(0.18, 0.15, 0.11) * floor_light;
    vec3 haze = vec3(0.20, 0.24, 0.26) * horizon;
    vec3 sun = vec3(2.20, 1.82, 1.18) * sun_core + vec3(0.46, 0.38, 0.24) * sun_bloom;
    return sky_color + ground_color + haze + sun;
}

vec3 traceReflection(vec3 direction, int bounces) {
    vec3 ray = normalize(direction);
    vec3 total = vec3(0.0);
    float energy = 1.0;
    float normalizer = 0.0;
    int steps = max(1, bounces);
    for (int index = 0; index < 6; ++index) {
        if (index >= steps) {
            break;
        }
        total += reflectionProbe(ray) * energy;
        normalizer += energy;
        if (ray.y < 0.0) {
            ray = normalize(vec3(ray.x * 0.82, -ray.y * 0.78 + 0.16, ray.z * 0.82));
        } else {
            ray = normalize(vec3(ray.x * 0.76 - ray.z * 0.18, ray.y * 0.54 - 0.12, ray.z * 0.76 + ray.x * 0.18));
        }
        energy *= 0.52;
    }
    return total / max(0.0001, normalizer);
}

float fresnelSchlick(float cos_theta, float f0) {
    return f0 + (1.0 - f0) * pow(1.0 - clamp(cos_theta, 0.0, 1.0), 5.0);
}

void main() {
    vec3 normal = normalize(v_normal);
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    if (u_two_sided_lighting && dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    vec3 diffuse_light = vec3(0.0);
    vec3 specular_light = vec3(0.0);
    float diffuse = v_material.x;
    float specular = v_material.y;
    float shininess = max(1.0, v_material.z);
    float reflectivity = clamp(v_material.w, 0.0, 1.0);
    diffuse_light += vec3(0.22, 0.26, 0.34) * max(0.0, normal.y) * u_bounce_light;
    diffuse_light += vec3(0.32, 0.22, 0.14) * max(0.0, -normal.y) * u_bounce_light;

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }

        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index] * attenuation;
        }

        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        if (specular > 0.0 && strength > 0.0) {
            vec3 halfway = normalize(light_direction + view_direction);
            float ndoth = max(0.0, dot(normal, halfway));
            specular_light += light_color * (pow(ndoth, shininess) + pow(ndoth, shininess * 3.8) * 0.32) * strength;
        }
    }

    vec3 base_color = v_color;
    int texture_index = int(v_texture_index + 0.5);
    if (u_use_textures && texture_index >= 0 && texture_index < u_texture_count) {
        vec2 uv = clamp(vec2(v_texcoord.x, 1.0 - v_texcoord.y), vec2(0.0), vec2(1.0));
        base_color *= texture(u_textures, vec3(uv, float(texture_index))).rgb;
    }

    vec3 color = v_emission + base_color * (u_ambient + diffuse_light * diffuse) + specular_light * specular;
    if (u_reflection_bounces > 0 && reflectivity > 0.0) {
        vec3 reflection_direction = reflect(-view_direction, normal);
        vec3 reflection_color = traceReflection(reflection_direction, u_reflection_bounces);
        float fresnel = fresnelSchlick(max(0.0, dot(view_direction, normal)), clamp(0.025 + reflectivity * 0.24, 0.02, 0.72));
        float reflection_mix = clamp(reflectivity * (0.28 + min(float(u_reflection_bounces), 5.0) * 0.10) + fresnel * 0.34, 0.0, 0.96);
        color = mix(color, reflection_color, reflection_mix);
        color += reflection_color * fresnel * reflectivity * 0.08;
    }
    if (u_tone_mapping) {
        color = color / (color + vec3(1.0));
    } else {
        color = clamp(color, 0.0, 1.0);
    }
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    frag_color = vec4(color, 1.0);
}
"""


PARTICLE_WATER_VERTEX_SHADER = """
#version 430

const int MAX_BUBBLES = 8;

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(std430, binding = 2) readonly buffer WaterParticles {
    Particle particles[];
};

in vec3 in_local;
in vec2 in_uv;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_focal;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;
uniform vec3 u_center;
uniform float u_radius;
uniform float u_level;
uniform float u_particle_base_y;
uniform float u_kernel_radius;
uniform float u_time;
uniform int u_particle_count;
uniform vec3 u_vortex_center;
uniform float u_vortex_strength;
uniform int u_bubble_count;
uniform vec4 u_bubbles[MAX_BUBBLES];

out vec3 v_world_position;
out vec3 v_normal;
out vec2 v_uv;
out float v_edge;
out float v_motion;
out float v_density;
out float v_foam;
out float v_vortex;

float particleOffset(vec2 local, out float motion_out, out float density_out) {
    vec2 sample_xz = u_center.xz + local;
    float kernel2 = max(0.000001, u_kernel_radius * u_kernel_radius);
    float total = 0.0;
    float displacement = 0.0;
    float motion = 0.0;
    for (int index = 0; index < u_particle_count; ++index) {
        Particle particle = particles[index];
        vec2 delta = particle.position_mass.xz - sample_xz;
        float distance2 = dot(delta, delta);
        if (distance2 > kernel2) {
            continue;
        }
        float weight = max(0.0, 1.0 - distance2 / kernel2);
        weight *= weight;
        total += weight;
        displacement += weight * ((particle.position_mass.y - u_particle_base_y) * 0.36 + particle.velocity_pad.y * 0.026);
        motion += weight * length(particle.velocity_pad.xz);
    }
    if (total <= 0.000001) {
        motion_out = 0.0;
        density_out = 0.0;
        return 0.0;
    }
    motion_out = motion / total;
    density_out = total;
    return displacement / total * 0.34;
}

float vortexOffset(vec2 world_xz, out float vortex_out) {
    vec2 delta = world_xz - u_vortex_center.xz;
    float distance = length(delta);
    float core_radius = max(0.07, u_radius * 0.20);
    float shoulder_radius = max(0.12, u_radius * 0.38);
    float core = exp(-(distance * distance) / max(0.0001, core_radius * core_radius));
    float shoulder = exp(-pow((distance - shoulder_radius) / max(0.0001, shoulder_radius * 0.55), 2.0));
    float angle = atan(delta.y, delta.x);
    float spiral = sin(angle * 4.0 + distance * 22.0 - u_time * 8.5) * exp(-distance / max(0.001, u_radius * 0.88));
    vortex_out = clamp(core * u_vortex_strength, 0.0, 1.0);
    return (-0.072 * core + 0.016 * shoulder + 0.010 * spiral) * u_vortex_strength;
}

float bubbleOffset(vec2 world_xz, out float foam_out) {
    float height = 0.0;
    float foam = 0.0;
    for (int index = 0; index < MAX_BUBBLES; ++index) {
        if (index >= u_bubble_count) {
            break;
        }
        vec4 bubble = u_bubbles[index];
        float radius = max(0.001, bubble.z);
        float phase = clamp(bubble.w, 0.0, 1.0);
        float distance = length(world_xz - bubble.xy);
        float rising = 1.0 - smoothstep(0.001, 0.02, phase);
        float dome = exp(-(distance * distance) / max(0.0001, radius * radius * 1.8));
        height += dome * radius * 0.42 * rising;

        float pop_energy = smoothstep(0.0, 0.14, phase) * (1.0 - smoothstep(0.72, 1.0, phase));
        float ring_center = mix(radius * 0.72, radius * 3.9, phase);
        float ring_width = mix(radius * 0.50, radius * 0.24, phase);
        float ring = exp(-pow((distance - ring_center) / max(0.0001, ring_width), 2.0));
        float pucker = exp(-(distance * distance) / max(0.0001, radius * radius * 0.95));
        height += (ring * radius * 0.34 - pucker * radius * 0.18) * pop_energy;
        foam = max(foam, ring * pop_energy);
    }
    foam_out = clamp(foam, 0.0, 1.0);
    return height;
}

float surfaceHeightWithMetrics(vec2 local, out float motion_out, out float density_out, out float foam_out, out float vortex_out) {
    float radial = length(local) / max(0.001, u_radius);
    float edge = clamp((radial - 0.88) / 0.12, 0.0, 1.0);
    vec2 world_xz = u_center.xz + local;
    float bubble_foam = 0.0;
    float vortex_amount = 0.0;
    float wind_ripple = sin(u_time * 6.28318530718 * 1.45 + local.x * 6.4 + local.y * 2.7) * 0.010;
    float cross_ripple = sin(u_time * 6.28318530718 * 2.0 + local.x * 9.0) * max(0.0, 1.0 - abs(local.y) / max(0.001, u_radius)) * 0.006;
    float micro_ripple = sin(u_time * 17.0 + local.x * 27.0 + local.y * 19.0) * 0.0025;
    float height = u_level + particleOffset(local, motion_out, density_out);
    height += vortexOffset(world_xz, vortex_amount);
    height += bubbleOffset(world_xz, bubble_foam);
    height += (wind_ripple + cross_ripple + micro_ripple) * max(0.0, 1.0 - radial * 0.32);
    float edge_flatten = edge * edge;
    foam_out = bubble_foam;
    vortex_out = vortex_amount;
    return u_level + (height - u_level) * (1.0 - edge_flatten * 0.86) + edge_flatten * 0.003;
}

float surfaceHeight(vec2 local) {
    float motion_out = 0.0;
    float density_out = 0.0;
    float foam_out = 0.0;
    float vortex_out = 0.0;
    return surfaceHeightWithMetrics(local, motion_out, density_out, foam_out, vortex_out);
}

void main() {
    vec2 local = in_local.xy;
    float motion = 0.0;
    float density = 0.0;
    float foam = 0.0;
    float vortex = 0.0;
    float height = surfaceHeightWithMetrics(local, motion, density, foam, vortex);

    vec3 world = vec3(u_center.x + local.x, height, u_center.z + local.y);
    vec3 relative = world - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float ndc_z = -1.0 + 2.0 * ((camera_z - u_near) / max(0.0001, u_far - u_near));

    gl_Position = vec4(camera_x * u_focal / u_aspect, camera_y * u_focal, ndc_z * camera_z, camera_z);
    v_world_position = world;
    v_normal = vec3(0.0, 1.0, 0.0);
    v_uv = in_uv;
    v_edge = in_local.z;
    v_motion = motion;
    v_density = clamp(density * 0.22, 0.0, 1.0);
    v_foam = foam;
    v_vortex = vortex;
}
"""


PARTICLE_WATER_FRAGMENT_SHADER = """
#version 430

const int MAX_LIGHTS = 8;

in vec3 v_world_position;
in vec3 v_normal;
in vec2 v_uv;
in float v_edge;
in float v_motion;
in float v_density;
in float v_foam;
in float v_vortex;

uniform vec3 u_camera_position;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform int u_reflection_bounces;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform vec3 u_material_color;
uniform vec4 u_material;
uniform float u_alpha;

out vec4 frag_color;

vec3 reflectionProbe(vec3 direction) {
    vec3 ray = normalize(direction);
    float sky = max(0.0, ray.y);
    float floor_light = max(0.0, -ray.y);
    float horizon = max(0.0, 1.0 - abs(ray.y));
    float sun_dot = max(0.0, dot(ray, normalize(vec3(-0.35, 0.72, -0.58))));
    float sun_core = pow(sun_dot, 220.0);
    float sun_bloom = pow(sun_dot, 28.0);
    vec3 sky_color = vec3(0.040, 0.065, 0.115) + vec3(0.26, 0.40, 0.66) * sky;
    vec3 ground_color = vec3(0.055, 0.065, 0.060) + vec3(0.16, 0.15, 0.12) * floor_light;
    vec3 haze = vec3(0.18, 0.23, 0.27) * horizon;
    vec3 sun = vec3(2.40, 1.95, 1.22) * sun_core + vec3(0.54, 0.43, 0.26) * sun_bloom;
    return sky_color + ground_color + haze + sun;
}

vec3 traceReflection(vec3 direction, int bounces) {
    vec3 ray = normalize(direction);
    vec3 total = vec3(0.0);
    float energy = 1.0;
    float normalizer = 0.0;
    int steps = max(1, bounces);
    for (int index = 0; index < 6; ++index) {
        if (index >= steps) {
            break;
        }
        total += reflectionProbe(ray) * energy;
        normalizer += energy;
        ray = normalize(vec3(ray.x * 0.78 - ray.z * 0.13, abs(ray.y) * 0.58 + 0.05, ray.z * 0.78 + ray.x * 0.13));
        energy *= 0.52;
    }
    return total / max(0.0001, normalizer);
}

void main() {
    vec3 normal = normalize(cross(dFdx(v_world_position), dFdy(v_world_position)));
    if (length(normal) < 0.001) {
        normal = normalize(v_normal);
    }
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    float diffuse = u_material.x;
    float specular = u_material.y;
    float shininess = max(1.0, u_material.z);
    float reflectivity = clamp(u_material.w, 0.0, 1.0);
    vec3 diffuse_light = vec3(0.0);
    vec3 specular_light = vec3(0.0);
    diffuse_light += vec3(0.22, 0.26, 0.34) * max(0.0, normal.y) * u_bounce_light;
    diffuse_light += vec3(0.22, 0.15, 0.09) * max(0.0, -normal.y) * u_bounce_light;

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }
        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index] * attenuation;
        }
        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        vec3 halfway = normalize(light_direction + view_direction);
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess) * strength;
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess * 2.8) * strength * (0.55 + v_motion * 0.18);
    }

    float fresnel = pow(1.0 - max(0.0, dot(view_direction, normal)), 5.0);
    float vortex_caustic = v_vortex * pow(max(0.0, sin((v_uv.x * 62.0 - v_uv.y * 47.0 + v_motion * 1.2) * 6.28318530718)), 5.0) * 0.18;
    float caustic = pow(max(0.0, sin((v_uv.x * 34.0 + v_uv.y * 21.0 + v_motion * 1.8) * 6.28318530718)), 8.0) * 0.12 + vortex_caustic;
    vec3 absorption = mix(vec3(0.70, 0.95, 1.0), vec3(0.42, 0.84, 0.94), clamp(v_edge, 0.0, 1.0));
    vec3 base_color = u_material_color * absorption + caustic;
    vec3 color = base_color * (u_ambient + diffuse_light * diffuse) + specular_light * specular;
    int water_bounces = max(2, u_reflection_bounces);
    vec3 reflection_color = traceReflection(reflect(-view_direction, normal), water_bounces);
    vec3 refraction_direction = refract(-view_direction, normal, 0.75);
    if (length(refraction_direction) < 0.001) {
        refraction_direction = normalize(-view_direction + normal * 0.35);
    }
    vec3 refraction_color = traceReflection(refraction_direction, water_bounces) * vec3(0.58, 0.88, 1.0);
    float transmission = clamp(0.18 + (1.0 - fresnel) * 0.24 + v_density * 0.10, 0.0, 0.48);
    color = mix(color, refraction_color, transmission);
    color = mix(color, reflection_color, clamp(reflectivity * 0.54 + fresnel * 0.66, 0.0, 0.96));
    color += reflection_color * pow(max(0.0, dot(reflect(-view_direction, normal), normalize(vec3(-0.35, 0.72, -0.58)))), 96.0) * (0.22 + reflectivity * 0.38);
    color = mix(color, vec3(0.84, 0.96, 1.0), clamp(v_foam * 0.66, 0.0, 0.66));

    if (u_tone_mapping) {
        color = color / (color + vec3(1.0));
    } else {
        color = clamp(color, 0.0, 1.0);
    }
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    float density_alpha = smoothstep(0.04, 0.28, v_density);
    frag_color = vec4(color, clamp(u_alpha + fresnel * 0.20 + v_foam * 0.18, 0.32, 0.88) * density_alpha);
}
"""


PARTICLE_FLUID_PUDDLE_VERTEX_SHADER = """
#version 430

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(std430, binding = 2) readonly buffer FluidParticles {
    Particle particles[];
};

in vec2 in_xz;
in vec2 in_uv;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_focal;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;
uniform float u_floor_y;
uniform float u_kernel_radius;
uniform float u_time;
uniform int u_particle_count;

out vec3 v_world_position;
out vec2 v_uv;
out float v_density;
out float v_motion;
out float v_edge;

float puddleHeight(vec2 sample_xz, out float density_out, out float motion_out) {
    float kernel2 = max(0.000001, u_kernel_radius * u_kernel_radius);
    float density = 0.0;
    float weighted_y = 0.0;
    float motion = 0.0;
    for (int index = 0; index < u_particle_count; ++index) {
        Particle particle = particles[index];
        float height_from_floor = particle.position_mass.y - u_floor_y;
        float floor_proximity = 1.0 - smoothstep(0.045, 0.36, height_from_floor);
        if (floor_proximity <= 0.001) {
            continue;
        }
        vec2 delta = particle.position_mass.xz - sample_xz;
        vec2 component_distance = abs(delta);
        if (component_distance.x > u_kernel_radius || component_distance.y > u_kernel_radius) {
            continue;
        }
        float distance2 = dot(delta, delta);
        if (distance2 > kernel2) {
            continue;
        }
        float weight = max(0.0, 1.0 - distance2 / kernel2);
        weight = weight * weight * floor_proximity;
        density += weight;
        weighted_y += weight * particle.position_mass.y;
        motion += weight * length(particle.velocity_pad.xz);
    }
    if (density <= 0.000001) {
        density_out = 0.0;
        motion_out = 0.0;
        return u_floor_y + 0.006;
    }
    float mean_y = weighted_y / density;
    density_out = density;
    motion_out = motion / density;
    float thickness = clamp(density * 0.010, 0.008, 0.070);
    float lifted = clamp((mean_y - u_floor_y) * 0.12, 0.0, 0.055);
    float ripple = 0.00045 * sin(u_time * 5.3 + sample_xz.x * 13.0 + sample_xz.y * 7.0) * smoothstep(0.55, 1.65, density);
    return u_floor_y + thickness + lifted + ripple;
}

void main() {
    float density = 0.0;
    float motion = 0.0;
    float height = puddleHeight(in_xz, density, motion);
    vec3 world = vec3(in_xz.x, height, in_xz.y);
    vec3 relative = world - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float ndc_z = -1.0 + 2.0 * ((camera_z - u_near) / max(0.0001, u_far - u_near));
    gl_Position = vec4(camera_x * u_focal / u_aspect, camera_y * u_focal, ndc_z * camera_z, camera_z);
    v_world_position = world;
    v_uv = in_uv;
    v_density = density;
    v_motion = motion;
    v_edge = min(min(in_uv.x, 1.0 - in_uv.x), min(in_uv.y, 1.0 - in_uv.y));
}
"""


PARTICLE_FLUID_PUDDLE_FRAGMENT_SHADER = """
#version 430

const int MAX_LIGHTS = 8;

in vec3 v_world_position;
in vec2 v_uv;
in float v_density;
in float v_motion;
in float v_edge;

uniform vec3 u_camera_position;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform int u_reflection_bounces;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform vec3 u_material_color;
uniform vec4 u_material;
uniform float u_alpha;
uniform float u_density_threshold;

out vec4 frag_color;

vec3 reflectionProbe(vec3 direction) {
    vec3 ray = normalize(direction);
    float sky = max(0.0, ray.y);
    float floor_light = max(0.0, -ray.y);
    float horizon = max(0.0, 1.0 - abs(ray.y));
    float sun_dot = max(0.0, dot(ray, normalize(vec3(-0.35, 0.72, -0.58))));
    float sun_core = pow(sun_dot, 210.0);
    float sun_bloom = pow(sun_dot, 27.0);
    vec3 sky_color = vec3(0.040, 0.065, 0.115) + vec3(0.25, 0.39, 0.65) * sky;
    vec3 ground_color = vec3(0.055, 0.065, 0.060) + vec3(0.17, 0.16, 0.12) * floor_light;
    vec3 haze = vec3(0.18, 0.23, 0.27) * horizon;
    vec3 sun = vec3(2.34, 1.90, 1.20) * sun_core + vec3(0.52, 0.42, 0.25) * sun_bloom;
    return sky_color + ground_color + haze + sun;
}

vec3 traceReflection(vec3 direction, int bounces) {
    vec3 ray = normalize(direction);
    vec3 total = vec3(0.0);
    float energy = 1.0;
    float normalizer = 0.0;
    int steps = max(1, bounces);
    for (int index = 0; index < 5; ++index) {
        if (index >= steps) {
            break;
        }
        total += reflectionProbe(ray) * energy;
        normalizer += energy;
        ray = normalize(vec3(ray.x * 0.78 - ray.z * 0.13, abs(ray.y) * 0.58 + 0.05, ray.z * 0.78 + ray.x * 0.13));
        energy *= 0.52;
    }
    return total / max(0.0001, normalizer);
}

void main() {
    float density_alpha = smoothstep(u_density_threshold, u_density_threshold * 2.9 + 0.001, v_density);
    float boundary_alpha = smoothstep(0.000, 0.045, v_edge);
    if (density_alpha * boundary_alpha <= 0.006) {
        discard;
    }
    vec3 normal = vec3(0.0, 1.0, 0.0);
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    float diffuse = u_material.x;
    float specular = u_material.y;
    float shininess = max(1.0, u_material.z);
    float reflectivity = clamp(u_material.w, 0.0, 1.0);
    vec3 diffuse_light = vec3(u_ambient);
    diffuse_light += vec3(0.22, 0.26, 0.34) * max(0.0, normal.y) * u_bounce_light;
    diffuse_light += vec3(0.20, 0.15, 0.10) * max(0.0, -normal.y) * u_bounce_light;
    vec3 specular_light = vec3(0.0);

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }
        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index] * attenuation;
        }
        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        vec3 halfway = normalize(light_direction + view_direction);
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess) * strength;
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess * 2.8) * strength * (0.44 + v_motion * 0.14);
    }

    float fresnel = pow(1.0 - max(0.0, dot(view_direction, normal)), 5.0);
    float caustic = smoothstep(u_density_threshold * 1.4, u_density_threshold * 4.0 + 0.001, v_density) * 0.018;
    vec3 absorption = mix(vec3(0.70, 0.95, 1.0), vec3(0.42, 0.84, 0.94), clamp(v_density * 0.12, 0.0, 1.0));
    vec3 color = (u_material_color * absorption + caustic) * diffuse_light * diffuse + specular_light * specular;
    int bounces = max(2, u_reflection_bounces);
    vec3 reflection_color = traceReflection(reflect(-view_direction, normal), bounces);
    vec3 refraction_direction = refract(-view_direction, normal, 0.75);
    if (length(refraction_direction) < 0.001) {
        refraction_direction = normalize(-view_direction + normal * 0.35);
    }
    vec3 refraction_color = traceReflection(refraction_direction, bounces) * vec3(0.58, 0.88, 1.0);
    color = mix(color, refraction_color, 0.18 + (1.0 - fresnel) * 0.20);
    color = mix(color, reflection_color, clamp(reflectivity * 0.50 + fresnel * 0.62, 0.0, 0.95));

    if (u_tone_mapping) {
        color = color / (color + vec3(1.0));
    } else {
        color = clamp(color, 0.0, 1.0);
    }
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    frag_color = vec4(color, clamp(u_alpha + fresnel * 0.18, 0.26, 0.84) * density_alpha * boundary_alpha);
}
"""


PARTICLE_FLUID_VOLUME_VERTEX_SHADER = """
#version 430

struct Particle {
    vec4 position_mass;
    vec4 velocity_pad;
};

layout(std430, binding = 2) readonly buffer FluidParticles {
    Particle particles[];
};

in vec2 in_corner;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_focal;
uniform float u_aspect;
uniform float u_near;
uniform float u_far;
uniform float u_particle_radius;
uniform float u_min_y;
uniform int u_particle_count;

out vec2 v_corner;
out vec3 v_world_position;
out vec3 v_particle_center;
out vec3 v_velocity;
out float v_active;

void main() {
    int index = gl_InstanceID;
    Particle particle = particles[index];
    vec3 center = particle.position_mass.xyz;
    v_active = center.y >= u_min_y ? 1.0 : 0.0;
    vec3 world = center + (u_camera_right * in_corner.x + u_camera_up * in_corner.y) * u_particle_radius;
    vec3 relative = world - u_camera_position;
    float camera_x = dot(relative, u_camera_right);
    float camera_y = dot(relative, u_camera_up);
    float camera_z = dot(relative, u_camera_forward);
    float ndc_z = -1.0 + 2.0 * ((camera_z - u_near) / max(0.0001, u_far - u_near));
    gl_Position = vec4(camera_x * u_focal / u_aspect, camera_y * u_focal, ndc_z * camera_z, camera_z);
    v_corner = in_corner;
    v_world_position = world;
    v_particle_center = center;
    v_velocity = particle.velocity_pad.xyz;
}
"""


PARTICLE_FLUID_VOLUME_FRAGMENT_SHADER = """
#version 430

const int MAX_LIGHTS = 8;

in vec2 v_corner;
in vec3 v_world_position;
in vec3 v_particle_center;
in vec3 v_velocity;
in float v_active;

uniform vec3 u_camera_position;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform vec3 u_camera_forward;
uniform float u_ambient;
uniform float u_gamma;
uniform float u_light_wrap;
uniform float u_bounce_light;
uniform bool u_tone_mapping;
uniform int u_reflection_bounces;
uniform int u_light_count;
uniform int u_light_kind[MAX_LIGHTS];
uniform vec3 u_light_vector[MAX_LIGHTS];
uniform vec3 u_light_color[MAX_LIGHTS];
uniform float u_light_intensity[MAX_LIGHTS];
uniform vec3 u_material_color;
uniform vec4 u_material;
uniform float u_alpha;

out vec4 frag_color;

vec3 reflectionProbe(vec3 direction) {
    vec3 ray = normalize(direction);
    float sky = max(0.0, ray.y);
    float floor_light = max(0.0, -ray.y);
    float horizon = max(0.0, 1.0 - abs(ray.y));
    float sun_dot = max(0.0, dot(ray, normalize(vec3(-0.35, 0.72, -0.58))));
    float sun_core = pow(sun_dot, 190.0);
    float sun_bloom = pow(sun_dot, 26.0);
    vec3 sky_color = vec3(0.040, 0.065, 0.115) + vec3(0.25, 0.39, 0.65) * sky;
    vec3 ground_color = vec3(0.055, 0.065, 0.060) + vec3(0.18, 0.16, 0.12) * floor_light;
    vec3 haze = vec3(0.19, 0.24, 0.28) * horizon;
    vec3 sun = vec3(2.18, 1.78, 1.12) * sun_core + vec3(0.48, 0.39, 0.24) * sun_bloom;
    return sky_color + ground_color + haze + sun;
}

vec3 traceReflection(vec3 direction, int bounces) {
    vec3 ray = normalize(direction);
    vec3 total = vec3(0.0);
    float energy = 1.0;
    float normalizer = 0.0;
    int steps = max(1, bounces);
    for (int index = 0; index < 5; ++index) {
        if (index >= steps) {
            break;
        }
        total += reflectionProbe(ray) * energy;
        normalizer += energy;
        ray = normalize(vec3(ray.x * 0.76 - ray.z * 0.12, abs(ray.y) * 0.60 + 0.04, ray.z * 0.76 + ray.x * 0.12));
        energy *= 0.52;
    }
    return total / max(0.0001, normalizer);
}

void main() {
    if (v_active < 0.5) {
        discard;
    }
    float radius2 = dot(v_corner, v_corner);
    if (radius2 > 1.0) {
        discard;
    }
    float cap = sqrt(max(0.0, 1.0 - radius2));
    vec3 normal = normalize(u_camera_right * v_corner.x + u_camera_up * v_corner.y + u_camera_forward * cap);
    vec3 view_direction = normalize(u_camera_position - v_world_position);
    if (dot(normal, view_direction) < 0.0) {
        normal = -normal;
    }

    float diffuse = u_material.x;
    float specular = u_material.y;
    float shininess = max(1.0, u_material.z);
    float reflectivity = clamp(u_material.w, 0.0, 1.0);
    vec3 diffuse_light = vec3(u_ambient);
    diffuse_light += vec3(0.20, 0.26, 0.34) * max(0.0, normal.y) * u_bounce_light;
    diffuse_light += vec3(0.18, 0.14, 0.10) * max(0.0, -normal.y) * u_bounce_light;
    vec3 specular_light = vec3(0.0);

    for (int index = 0; index < MAX_LIGHTS; ++index) {
        if (index >= u_light_count) {
            break;
        }
        vec3 light_direction;
        float strength;
        if (u_light_kind[index] == 0) {
            light_direction = normalize(u_light_vector[index]);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index];
        } else {
            vec3 offset = u_light_vector[index] - v_world_position;
            float distance = max(0.0001, length(offset));
            light_direction = offset / distance;
            float attenuation = 1.0 / (1.0 + distance * distance);
            float facing = dot(normal, light_direction);
            float response = u_light_wrap > 0.0 ? max(0.0, (facing + u_light_wrap) / (1.0 + u_light_wrap)) : max(0.0, facing);
            strength = response * u_light_intensity[index] * attenuation;
        }
        vec3 light_color = u_light_color[index];
        diffuse_light += light_color * strength;
        vec3 halfway = normalize(light_direction + view_direction);
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess) * strength;
        specular_light += light_color * pow(max(0.0, dot(normal, halfway)), shininess * 2.6) * strength * 0.42;
    }

    float fresnel = pow(1.0 - max(0.0, dot(view_direction, normal)), 5.0);
    vec3 absorption = mix(vec3(0.64, 0.92, 1.0), vec3(0.38, 0.80, 0.94), clamp(length(v_velocity) * 0.42, 0.0, 1.0));
    vec3 base_color = u_material_color * absorption;
    vec3 color = base_color * diffuse_light * diffuse + specular_light * specular;
    int bounces = max(2, u_reflection_bounces);
    vec3 reflection_color = traceReflection(reflect(-view_direction, normal), bounces);
    vec3 refraction_direction = refract(-view_direction, normal, 0.75);
    if (length(refraction_direction) < 0.001) {
        refraction_direction = normalize(-view_direction + normal * 0.35);
    }
    vec3 refraction_color = traceReflection(refraction_direction, bounces) * vec3(0.58, 0.88, 1.0);
    color = mix(color, refraction_color, 0.18 + (1.0 - fresnel) * 0.24);
    color = mix(color, reflection_color, clamp(reflectivity * 0.42 + fresnel * 0.58, 0.0, 0.94));

    if (u_tone_mapping) {
        color = color / (color + vec3(1.0));
    } else {
        color = clamp(color, 0.0, 1.0);
    }
    float gamma = max(0.01, u_gamma);
    if (abs(gamma - 1.0) > 0.0001) {
        color = pow(color, vec3(1.0 / gamma));
    }
    float edge_alpha = 1.0 - smoothstep(0.22, 1.0, radius2);
    frag_color = vec4(color, clamp(u_alpha + fresnel * 0.28, 0.20, 0.78) * edge_alpha);
}
"""


OVERLAY_VERTEX_SHADER = """
#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texcoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texcoord = in_texcoord;
}
"""


OVERLAY_FRAGMENT_SHADER = """
#version 330

uniform sampler2D u_texture;

in vec2 v_texcoord;

out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, v_texcoord);
}
"""


_FLOATS_PER_VERTEX = 19
_VERTEX_FORMAT = "3f 3f 3f 3f 4f 2f 1f"
_VERTEX_ATTRIBUTES = ("in_position", "in_normal", "in_color", "in_emission", "in_material", "in_texcoord", "in_texture_index")
_MAX_TEXTURES = 8
_OVERLAY_VERTEX_FORMAT = "2f 2f"
_OVERLAY_VERTEX_ATTRIBUTES = ("in_position", "in_texcoord")
_WATER_VERTEX_FORMAT = "3f 2f"
_WATER_VERTEX_ATTRIBUTES = ("in_local", "in_uv")
_FLUID_VOLUME_VERTEX_FORMAT = "2f"
_FLUID_VOLUME_VERTEX_ATTRIBUTES = ("in_corner",)
_MENU_BUTTON_ACTIONS = {"apply", "done", "cancel", "resume", "quit"}
_MENU_GROUP_ORDER = ("Graphics", "Sky", "World", "Physics", "Camera", "Demo", "Main")
_VERTEX_STRIDE_BYTES = _FLOATS_PER_VERTEX * 4


@dataclass(frozen=True)
class LiveFrameStats:
    """Timing and payload counters for the most recent live frame."""

    build_seconds: float
    draw_seconds: float
    triangle_vertices: int
    line_vertices: int

    @property
    def total_seconds(self) -> float:
        return self.build_seconds + self.draw_seconds

    @property
    def approx_fps(self) -> float:
        return 1.0 / self.total_seconds if self.total_seconds > 0.0 else 0.0


@dataclass
class LiveFlyCamera:
    """Mouse-look camera for direct live navigation."""

    position: Vec3
    yaw_degrees: float = 0.0
    pitch_degrees: float = 0.0
    fov_degrees: float = 60.0
    speed: float = 2.8
    mouse_sensitivity: float = 0.12
    render_position: Vec3 | None = None
    render_yaw_degrees: float | None = None
    render_pitch_degrees: float | None = None

    @classmethod
    def looking_at(
        cls,
        position: Vec3 | tuple[float, float, float],
        target: Vec3 | tuple[float, float, float],
        *,
        fov_degrees: float = 60.0,
        speed: float = 2.8,
        mouse_sensitivity: float = 0.12,
    ) -> "LiveFlyCamera":
        eye = position if isinstance(position, Vec3) else Vec3(*position)
        target_vec = target if isinstance(target, Vec3) else Vec3(*target)
        forward = (target_vec - eye).normalized(Vec3(0.0, 0.0, 1.0))
        return cls(
            position=eye,
            yaw_degrees=degrees(atan2(forward.x, forward.z)),
            pitch_degrees=degrees(asin(max(-1.0, min(1.0, forward.y)))),
            fov_degrees=fov_degrees,
            speed=speed,
            mouse_sensitivity=mouse_sensitivity,
        )

    def _forward_for(self, yaw_degrees: float, pitch_degrees: float) -> Vec3:
        yaw = radians(yaw_degrees)
        pitch = radians(pitch_degrees)
        return Vec3(sin(yaw) * cos(pitch), sin(pitch), cos(yaw) * cos(pitch)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def forward(self) -> Vec3:
        return self._forward_for(self.yaw_degrees, self.pitch_degrees)

    @property
    def flat_forward(self) -> Vec3:
        yaw = radians(self.yaw_degrees)
        return Vec3(sin(yaw), 0.0, cos(yaw)).normalized(Vec3(0.0, 0.0, 1.0))

    @property
    def right(self) -> Vec3:
        forward = self.flat_forward
        return Vec3(forward.z, 0.0, -forward.x)

    def look(self, relative_x: float, relative_y: float) -> None:
        self.yaw_degrees += relative_x * self.mouse_sensitivity
        self.pitch_degrees = max(-86.0, min(86.0, self.pitch_degrees - relative_y * self.mouse_sensitivity))

    def move(self, keys: set[str], dt: float) -> None:
        if dt <= 0.0:
            return
        direction = Vec3(0.0, 0.0, 0.0)
        if "w" in keys:
            direction = direction + self.flat_forward
        if "s" in keys:
            direction = direction - self.flat_forward
        if "d" in keys:
            direction = direction + self.right
        if "a" in keys:
            direction = direction - self.right
        if "space" in keys or "shift" in keys:
            direction = direction + Vec3(0.0, 1.0, 0.0)
        if "ctrl" in keys:
            direction = direction - Vec3(0.0, 1.0, 0.0)
        if direction.length_squared() > 0.0:
            self.position = self.position + direction.normalized() * (self.speed * dt)

    def camera(self) -> Camera:
        return Camera(position=self.position, target=self.position + self.forward, fov_degrees=self.fov_degrees)

    def smoothed_camera(self, dt: float, *, responsiveness: float = 22.0) -> Camera:
        """Return a camera eased toward the current control state for rendering."""

        if self.render_position is None or self.render_yaw_degrees is None or self.render_pitch_degrees is None:
            self.render_position = self.position
            self.render_yaw_degrees = self.yaw_degrees
            self.render_pitch_degrees = self.pitch_degrees
            return self.camera()

        alpha = max(0.0, min(1.0, dt * responsiveness))
        self.render_position = self.render_position + (self.position - self.render_position) * alpha
        self.render_yaw_degrees = _lerp_degrees(self.render_yaw_degrees, self.yaw_degrees, alpha)
        self.render_pitch_degrees = self.render_pitch_degrees + (self.pitch_degrees - self.render_pitch_degrees) * alpha
        forward = self._forward_for(self.render_yaw_degrees, self.render_pitch_degrees)
        return Camera(position=self.render_position, target=self.render_position + forward, fov_degrees=self.fov_degrees)

    def reset_smoothing(self) -> None:
        self.render_position = self.position
        self.render_yaw_degrees = self.yaw_degrees
        self.render_pitch_degrees = self.pitch_degrees


@dataclass(frozen=True)
class LiveMenuOption:
    """One decision in a live-rendering escape menu."""

    action: str
    label: str
    detail: str = ""
    group: str = ""
    enabled: bool = True


@dataclass(frozen=True)
class LiveMenuTheme:
    """Simple color theme for live menus."""

    panel: tuple[int, int, int, int] = (5, 5, 5, 240)
    border: tuple[int, int, int, int] = (86, 86, 86, 255)
    text: tuple[int, int, int] = (238, 238, 238)
    muted_text: tuple[int, int, int] = (152, 152, 152)
    row: tuple[int, int, int, int] = (12, 12, 12, 236)
    row_selected: tuple[int, int, int, int] = (38, 38, 38, 244)
    button: tuple[int, int, int, int] = (28, 28, 28, 244)
    button_selected: tuple[int, int, int, int] = (58, 58, 58, 250)
    button_border: tuple[int, int, int, int] = (96, 96, 96, 255)
    active_border: tuple[int, int, int, int] = (210, 210, 210, 255)


@dataclass
class LiveMenu:
    """Shared mouse-and-key menu state for live OpenGL demos."""

    title: str = "py_3d"
    options: tuple[LiveMenuOption, ...] = (
        LiveMenuOption("resume", "Resume"),
        LiveMenuOption("quit", "Quit"),
    )
    selected_index: int = 0
    visible: bool = False
    hitboxes: tuple[tuple[int, int, int, int, int], ...] = ()
    tab_hitboxes: tuple[tuple[int, int, int, int, str], ...] = ()
    active_group: str | None = None
    scroll_offsets: dict[str, int] = field(default_factory=dict)
    background_blur: bool = False
    theme: LiveMenuTheme = field(default_factory=LiveMenuTheme)

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("live menu requires at least one option")
        self.selected_index %= len(self.options)

    def open(self) -> None:
        self.visible = True

    def close(self) -> None:
        self.visible = False
        self.hitboxes = ()
        self.tab_hitboxes = ()

    def toggle(self) -> None:
        self.visible = not self.visible

    def move(self, amount: int) -> None:
        indexes = self.visible_option_indexes()
        if not indexes:
            self.selected_index = (self.selected_index + amount) % len(self.options)
            return
        if self.selected_index not in indexes:
            self.selected_index = indexes[0]
            return
        local_index = indexes.index(self.selected_index)
        self.selected_index = indexes[(local_index + amount) % len(indexes)]
        self.active_group = self.group_for(self.options[self.selected_index])

    def selected_action(self) -> str:
        return self.options[self.selected_index].action

    def selected_enabled(self) -> bool:
        return self.options[self.selected_index].enabled

    def group_for(self, option: LiveMenuOption) -> str:
        if option.group:
            return option.group
        action = option.action
        if action.startswith("sky_") or action.startswith("time_") or action.startswith("sun_"):
            return "Sky"
        if action.startswith(("quality", "poly", "reflection", "texture", "tone", "smooth")) or action == "toggle_render":
            return "Graphics"
        if action.startswith(("wind", "blade")) or action in {"pause", "reset"}:
            return "Physics"
        if "camera" in action:
            return "Camera"
        return "Demo"

    def groups(self) -> tuple[str, ...]:
        groups = {self.group_for(option) for option in self.options if option.action not in _MENU_BUTTON_ACTIONS}
        return tuple(group for group in _MENU_GROUP_ORDER if group in groups) + tuple(sorted(groups - set(_MENU_GROUP_ORDER)))

    def current_group(self) -> str:
        groups = self.groups()
        if not groups:
            return "Main"
        if self.active_group in groups:
            return self.active_group
        selected = self.options[self.selected_index]
        selected_group = self.group_for(selected)
        if selected.action not in _MENU_BUTTON_ACTIONS and selected_group in groups:
            self.active_group = selected_group
            return selected_group
        self.active_group = groups[0]
        return groups[0]

    def set_group(self, group: str) -> None:
        groups = self.groups()
        if group not in groups:
            return
        self.active_group = group
        indexes = self.visible_option_indexes()
        if indexes and self.selected_index not in indexes:
            self.selected_index = indexes[0]

    def cycle_group(self, amount: int) -> None:
        groups = self.groups()
        if not groups:
            return
        current = self.current_group()
        index = groups.index(current)
        self.set_group(groups[(index + amount) % len(groups)])

    def visible_option_indexes(self) -> list[int]:
        group = self.current_group()
        return [index for index, option in enumerate(self.options) if option.action not in _MENU_BUTTON_ACTIONS and self.group_for(option) == group]

    def scroll(self, amount: int) -> None:
        group = self.current_group()
        current = self.scroll_offsets.get(group, 0)
        max_offset = max(0, len(self.visible_option_indexes()) - 1)
        self.scroll_offsets[group] = max(0, min(max_offset, current + amount))

    def has_action(self, action: str) -> bool:
        return any(option.action == action and option.enabled for option in self.options)

    def cancel_action(self) -> str:
        if self.has_action("cancel"):
            return "cancel"
        if self.has_action("done"):
            return "done"
        return "resume"

    def set_hitboxes(self, hitboxes: Iterable[tuple[int, int, int, int, int]]) -> None:
        self.hitboxes = tuple(hitboxes)

    def set_tab_hitboxes(self, hitboxes: Iterable[tuple[int, int, int, int, str]]) -> None:
        self.tab_hitboxes = tuple(hitboxes)

    def index_at(self, position: tuple[int, int]) -> int | None:
        x, y = position
        for left, top, width, height, index in self.hitboxes:
            if left <= x <= left + width and top <= y <= top + height:
                return index
        return None

    def tab_at(self, position: tuple[int, int]) -> str | None:
        x, y = position
        for left, top, width, height, group in self.tab_hitboxes:
            if left <= x <= left + width and top <= y <= top + height:
                return group
        return None

    def handle_pointer_event(self, event) -> str | None:
        if not self.visible:
            return None
        if event.kind == "motion":
            tab = self.tab_at(event.pos)
            if tab is not None:
                return "handled"
            index = self.index_at(event.pos)
            if index is not None:
                return "handled"
            return "handled"
        if event.kind == "wheel":
            self.scroll(-event.y)
            return "navigate"
        if event.kind == "button" and event.button == 1:
            tab = self.tab_at(event.pos)
            if tab is not None:
                self.set_group(tab)
                return "navigate"
            index = self.index_at(event.pos)
            if index is not None:
                self.selected_index = index
                self.active_group = self.group_for(self.options[index])
                if self.options[index].enabled:
                    return self.selected_action()
                return "handled"
            return "handled"
        return None

    def handle_key_name(self, key: str) -> str | None:
        if not self.visible:
            if key == "escape":
                self.open()
                return "opened"
            return None
        if key == "escape":
            return self.cancel_action()
        if key in ("right", "tab"):
            self.cycle_group(1)
            return "navigate"
        if key == "left":
            self.cycle_group(-1)
            return "navigate"
        if key in ("down", "s"):
            self.move(1)
            return "navigate"
        if key in ("up", "w"):
            self.move(-1)
            return "navigate"
        if key in ("pageup", "q"):
            self.scroll(-1)
            return "navigate"
        if key in ("pagedown", "e"):
            self.scroll(1)
            return "navigate"
        if key in ("return", "enter", "kp_enter", "space"):
            if self.selected_enabled():
                return self.selected_action()
            return "handled"
        return "handled"


def render_live_menu_surface(menu: LiveMenu, width: int, height: int):
    """Return a py_3d pixel menu surface plus its screen position."""

    theme = menu.theme
    panel_width = min(width - 36, max(500, int(width * 0.46)))
    panel_height = min(height - 36, max(330, int(height * 0.58)))
    panel_width = max(320, panel_width)
    panel_height = max(260, panel_height)
    panel_left = max(12, (width - panel_width) // 2)
    panel_top = max(12, (height - panel_height) // 2)
    surface = PixelBuffer.new(panel_width, panel_height, theme.panel[:3])

    draw.rect(surface, (0, 0), (panel_width, panel_height), theme.border[:3])
    draw.text(surface, (20, 14), draw.fit_text(menu.title, panel_width - 40, scale=2), theme.text, scale=2)
    draw.line(surface, (20, 45), (panel_width - 20, 45), theme.border[:3])

    groups = menu.groups()
    current_group = menu.current_group()
    tab_hitboxes: list[tuple[int, int, int, int, str]] = []
    tab_x = 20
    tab_y = 56
    for group in groups:
        text_color = theme.text if group == current_group else theme.muted_text
        group_label = draw.fit_text(group, max(42, panel_width - tab_x - 28))
        text_width, _text_height = draw.text_size(group_label)
        tab_width = max(64, text_width + 16)
        if tab_x + tab_width > panel_width - 20:
            break
        tab_rect = (tab_x, tab_y, tab_width, 24)
        if group == current_group:
            draw.rect(surface, (tab_x, tab_y), (tab_width, 24), theme.row_selected[:3], fill=True)
            draw.line(surface, (tab_x + 8, tab_y + 22), (tab_x + tab_width - 8, tab_y + 22), theme.active_border[:3])
        else:
            draw.rect(surface, (tab_x, tab_y), (tab_width, 24), theme.button[:3])
        draw.text(surface, (tab_x + (tab_width - text_width) // 2, tab_y + 8), group_label, text_color)
        tab_hitboxes.append((panel_left + tab_x, panel_top + tab_y, tab_width, 24, group))
        tab_x += tab_width + 6

    option_indexes = menu.visible_option_indexes()
    display_rows = _menu_display_rows(menu, option_indexes)
    option_area_top = 92
    footer_height = 52
    option_area_height = max(80, panel_height - option_area_top - footer_height - 10)
    option_height = 38
    option_width = panel_width - 40
    capacity = max(1, option_area_height // option_height)
    max_scroll = max(0, len(display_rows) - capacity)
    offset = max(0, min(menu.scroll_offsets.get(current_group, 0), max_scroll))
    menu.scroll_offsets[current_group] = offset
    visible_rows = display_rows[offset : offset + capacity]

    hitboxes: list[tuple[int, int, int, int, int]] = []
    for local, row in enumerate(visible_rows):
        x = 20
        y = option_area_top + local * option_height
        selected = menu.selected_index in row.indexes
        if selected:
            draw.rect(surface, (x, y), (option_width, option_height), theme.row_selected[:3], fill=True)
            draw.rect(surface, (x, y), (3, option_height), theme.active_border[:3], fill=True)
        else:
            draw.rect(surface, (x, y), (option_width, option_height), theme.row[:3], fill=True)
        draw.line(surface, (x, y + option_height - 1), (x + option_width, y + option_height - 1), (42, 42, 42))
        detail_x = x + max(190, int(option_width * 0.45))
        button_left = x + option_width
        if row.down_index is not None and row.up_index is not None:
            button_size = 26
            button_left = x + option_width - button_size * 2 - 10
        label_max = max(40, detail_x - x - 24)
        detail_max = max(40, button_left - detail_x - 10)
        row_enabled = any(menu.options[index].enabled for index in row.indexes)
        label_color = theme.text if row_enabled else _disabled_menu_color(theme)
        detail_color = theme.muted_text if row_enabled else _disabled_menu_color(theme)
        draw.text(surface, (x + 12, y + 14), draw.fit_text(row.label, label_max), label_color)
        if row.detail:
            draw.text(surface, (detail_x, y + 14), draw.fit_text(row.detail, detail_max), detail_color)
        if row.down_index is not None and row.up_index is not None:
            button_size = 26
            minus_rect = (x + option_width - button_size * 2 - 10, y + 6, button_size, button_size)
            plus_rect = (x + option_width - button_size - 6, y + 6, button_size, button_size)
            _draw_menu_small_button(surface, "-", minus_rect, row.down_index == menu.selected_index, theme, enabled=menu.options[row.down_index].enabled)
            _draw_menu_small_button(surface, "+", plus_rect, row.up_index == menu.selected_index, theme, enabled=menu.options[row.up_index].enabled)
            hitboxes.append((panel_left + minus_rect[0], panel_top + minus_rect[1], minus_rect[2], minus_rect[3], row.down_index))
            hitboxes.append((panel_left + plus_rect[0], panel_top + plus_rect[1], plus_rect[2], plus_rect[3], row.up_index))
            body_index = row.up_index if menu.options[row.up_index].enabled else row.down_index
            hitboxes.append((panel_left + x, panel_top + y, option_width - button_size * 2 - 18, option_height, body_index))
        else:
            hitboxes.append((panel_left + x, panel_top + y, option_width, option_height, row.indexes[0]))

    if max_scroll > 0:
        marker = f"{offset + 1}-{min(len(display_rows), offset + capacity)} / {len(display_rows)}"
        draw.text(surface, (20, panel_height - footer_height - 12), marker, theme.muted_text)

    footer_options = tuple((index, option) for index, option in enumerate(menu.options) if option.action in _MENU_BUTTON_ACTIONS)
    footer_y = panel_height - 40
    footer_x = 20
    draw.line(surface, (20, footer_y - 10), (panel_width - 20, footer_y - 10), theme.border[:3])
    for option_index, option in footer_options:
        label = draw.fit_text(option.label, max(1, panel_width - footer_x - 34))
        text_width, _text_height = draw.text_size(label)
        button_width = max(82, text_width + 22)
        if footer_x + button_width > panel_width - 22:
            break
        selected = option_index == menu.selected_index
        enabled = option.enabled
        fill = theme.button_selected if selected and enabled else theme.button
        border = theme.active_border if selected and enabled else theme.button_border
        text_color = theme.text if enabled else _disabled_menu_color(theme)
        draw.rect(surface, (footer_x, footer_y), (button_width, 30), fill[:3], fill=True)
        draw.rect(surface, (footer_x, footer_y), (button_width, 30), border[:3])
        draw.text(surface, (footer_x + (button_width - text_width) // 2, footer_y + 11), label, text_color)
        hitboxes.append((panel_left + footer_x, panel_top + footer_y, button_width, 30, option_index))
        footer_x += button_width + 6

    menu.set_hitboxes(hitboxes)
    menu.set_tab_hitboxes(tab_hitboxes)
    return PixelBuffer.from_rgb_bytes(surface.width, surface.height, bytearray(surface.to_rgb_bytes())), panel_left, panel_top


@dataclass(frozen=True)
class _MenuDisplayRow:
    indexes: tuple[int, ...]
    label: str
    detail: str = ""
    down_index: int | None = None
    up_index: int | None = None


def _menu_display_rows(menu: LiveMenu, option_indexes: list[int]) -> list[_MenuDisplayRow]:
    rows: list[_MenuDisplayRow] = []
    consumed: set[int] = set()
    index_lookup = {menu.options[index].action: index for index in option_indexes}
    for index in option_indexes:
        if index in consumed:
            continue
        option = menu.options[index]
        pair = _paired_action(option.action)
        if pair is not None:
            base, direction = pair
            opposite_action = f"{base}_{'down' if direction == 'up' else 'up'}"
            opposite_index = index_lookup.get(opposite_action)
            if opposite_index is not None:
                down_index = index if direction == "down" else opposite_index
                up_index = index if direction == "up" else opposite_index
                consumed.update({down_index, up_index})
                down = menu.options[down_index]
                up = menu.options[up_index]
                rows.append(
                    _MenuDisplayRow(
                        indexes=(down_index, up_index),
                        label=_paired_label(base, down, up),
                        detail=up.detail or down.detail,
                        down_index=down_index,
                        up_index=up_index,
                    )
                )
                continue
        consumed.add(index)
        rows.append(_MenuDisplayRow(indexes=(index,), label=option.label, detail=option.detail))
    return rows


def _paired_action(action: str) -> tuple[str, str] | None:
    if action.endswith("_up"):
        return action[:-3], "up"
    if action.endswith("_down"):
        return action[:-5], "down"
    return None


def _paired_label(base: str, down: LiveMenuOption, up: LiveMenuOption) -> str:
    labels = {
        "poly": "Polygons",
        "reflection": "Reflections",
        "reflections": "Reflections",
        "texture": "Texture",
        "gamma": "Gamma",
        "wind": "Wind",
        "blade": "Swirl",
        "look_smoothing": "Look Smoothing",
        "radius": "View Distance",
        "tree_lod": "Tree LOD",
        "sun": "Sun",
        "sky_time": "Sky Time",
        "sky_sun": "Sun",
    }
    if base in labels:
        return labels[base]
    stripped = _strip_direction_label(up.label) or _strip_direction_label(down.label)
    if stripped:
        return stripped
    return base.replace("_", " ").title()


def _strip_direction_label(label: str) -> str:
    result = label.strip()
    for suffix in (" +", " -", "+", "-"):
        if result.endswith(suffix):
            result = result[: -len(suffix)].strip()
    for prefix in ("More ", "Less ", "Fewer "):
        if result.startswith(prefix):
            result = result[len(prefix) :].strip()
    return result


def _disabled_menu_color(theme: LiveMenuTheme) -> tuple[int, int, int]:
    return tuple(max(42, int(channel * 0.62)) for channel in theme.muted_text)


def _draw_menu_small_button(surface: PixelBuffer, label: str, rect: tuple[int, int, int, int], selected: bool, theme: LiveMenuTheme, *, enabled: bool = True) -> None:
    fill = theme.button_selected if selected and enabled else theme.button
    border = theme.active_border if selected and enabled else theme.button_border
    text_color = theme.text if enabled else _disabled_menu_color(theme)
    x, y, width, height = rect
    draw.rect(surface, (x, y), (width, height), fill[:3], fill=True)
    draw.rect(surface, (x, y), (width, height), border[:3])
    text_width, text_height = draw.text_size(label)
    draw.text(surface, (x + (width - text_width) // 2, y + (height - text_height) // 2), label, text_color)


@dataclass(frozen=True)
class _Template:
    vertices: tuple[tuple[float, float, float, float, float, float, float, float], ...]


@dataclass
class _ResidentObjectEntry:
    key: tuple
    fingerprint: tuple
    triangle_bytes: bytes
    line_bytes: bytes
    triangle_vertices: int
    line_vertices: int
    triangle_first: int = 0
    line_first: int = 0


class LiveSceneBatchBuilder:
    """Build GPU vertex payloads from py_3d scenes with generated-mesh caches."""

    def __init__(self) -> None:
        self._sphere_templates: dict[tuple[float, int | None, int, int, bool], _Template] = {}
        self._bowl_templates: dict[tuple[float, float, float, int | None, int, int, bool], _Template] = {}
        self._capsule_templates: dict[tuple[float, float, int, int, bool], _Template] = {}
        self._box_templates: dict[tuple[float, float, float, bool], _Template] = {}
        self._mesh_templates: dict[tuple[int, bool], tuple[object, tuple[_Template, Material] | None]] = {}
        self._transformed_mesh_templates: dict[tuple[int, bool], tuple[object, tuple[tuple[_Template, Material, bool], ...]]] = {}
        self.active_textures: list[PixelBuffer] = []
        self._frame_texture_slots: dict[int, int] = {}

    def begin_frame(self, textures: Iterable[PixelBuffer] = ()) -> None:
        self.active_textures = list(textures)[:_MAX_TEXTURES]
        self._frame_texture_slots = {id(texture): index for index, texture in enumerate(self.active_textures)}

    def build(self, scene: Scene, settings: RenderSettings) -> tuple[bytes, bytes, int, int]:
        self.begin_frame()
        triangle_data = array("f")
        line_data = array("f")

        for obj in scene.objects:
            self._append_object(triangle_data, line_data, obj, settings)

        triangle_vertices = len(triangle_data) // _FLOATS_PER_VERTEX
        line_vertices = len(line_data) // _FLOATS_PER_VERTEX
        return triangle_data.tobytes(), line_data.tobytes(), triangle_vertices, line_vertices

    def build_object(self, obj, settings: RenderSettings) -> tuple[bytes, bytes, int, int]:
        triangle_data = array("f")
        line_data = array("f")
        self._append_object(triangle_data, line_data, obj, settings)
        triangle_vertices = len(triangle_data) // _FLOATS_PER_VERTEX
        line_vertices = len(line_data) // _FLOATS_PER_VERTEX
        return triangle_data.tobytes(), line_data.tobytes(), triangle_vertices, line_vertices

    def _append_object(self, triangle_data: array, line_data: array, obj, settings: RenderSettings) -> None:
        if isinstance(obj, Point3):
            return
        if isinstance(obj, ParticleFluidPuddle):
            return
        if isinstance(obj, ParticleFluidVolume):
            return
        if isinstance(obj, ParticleWaterSurface):
            return
        if isinstance(obj, Line3):
            self._append_line(line_data, obj.start, obj.end, obj.material)
            return
        if isinstance(obj, LineSet3):
            self._append_line_set(line_data, obj)
            return

        if settings.wireframe:
            for triangle in _triangles_for(obj, settings):
                self._append_triangle_wireframe(line_data, triangle)
            return

        if isinstance(obj, Sphere):
            self._append_sphere(triangle_data, obj, settings)
        elif isinstance(obj, Bowl):
            self._append_bowl(triangle_data, obj, settings)
        elif isinstance(obj, Capsule):
            self._append_capsule(triangle_data, obj, settings)
        elif isinstance(obj, Box):
            self._append_box(triangle_data, obj, settings)
        elif isinstance(obj, TransformedMesh):
            self._append_transformed_mesh(triangle_data, obj, settings)
        elif isinstance(obj, Mesh):
            self._append_mesh(triangle_data, obj, settings)
        elif isinstance(obj, (Plane, Triangle)):
            self._append_triangles(triangle_data, _triangles_for(obj, settings), settings)
        else:
            self._append_triangles(triangle_data, _triangles_for(obj, settings), settings)

    def _append_sphere(self, payload: array, sphere: Sphere, settings: RenderSettings) -> None:
        key = (
            sphere.radius,
            id(sphere.perturbation) if sphere.perturbation is not None else None,
            settings.sphere_segments,
            settings.sphere_rings,
            settings.smooth_shading,
        )
        template = self._sphere_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Sphere((0.0, 0.0, 0.0), sphere.radius, Material(), sphere.perturbation).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                ),
                smooth_shading=settings.smooth_shading,
            )
            self._sphere_templates[key] = template
        _append_template(payload, template, sphere.material, sphere.center, sphere.rotation, self.texture_index_for(sphere.material))

    def _append_bowl(self, payload: array, bowl: Bowl, settings: RenderSettings) -> None:
        key = (
            bowl.radius,
            bowl.depth,
            bowl.thickness,
            id(bowl.perturbation) if bowl.perturbation is not None else None,
            settings.sphere_segments,
            settings.sphere_rings,
            settings.smooth_shading,
        )
        template = self._bowl_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Bowl(
                    (0.0, 0.0, 0.0),
                    bowl.radius,
                    Material(),
                    bowl.depth,
                    bowl.perturbation,
                    bowl.thickness,
                ).to_triangles(segments=settings.sphere_segments, rings=settings.sphere_rings),
                smooth_shading=settings.smooth_shading,
            )
            self._bowl_templates[key] = template
        _append_template(payload, template, bowl.material, bowl.center, Vec3(0.0, 0.0, 0.0), self.texture_index_for(bowl.material))

    def _append_capsule(self, payload: array, capsule: Capsule, settings: RenderSettings) -> None:
        key = (capsule.radius, capsule.height, settings.sphere_segments, settings.sphere_rings, settings.smooth_shading)
        template = self._capsule_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Capsule((0.0, 0.0, 0.0), capsule.radius, capsule.height, Material()).to_triangles(
                    segments=settings.sphere_segments,
                    rings=settings.sphere_rings,
                ),
                smooth_shading=settings.smooth_shading,
            )
            self._capsule_templates[key] = template
        _append_template(payload, template, capsule.material, capsule.center, Vec3(0.0, 0.0, 0.0), self.texture_index_for(capsule.material))

    def _append_box(self, payload: array, box: Box, settings: RenderSettings) -> None:
        key = (box.size.x, box.size.y, box.size.z, settings.smooth_shading)
        template = self._box_templates.get(key)
        if template is None:
            template = _template_from_triangles(
                Box((0.0, 0.0, 0.0), box.size, Material()).to_triangles(),
                smooth_shading=settings.smooth_shading,
            )
            self._box_templates[key] = template
        _append_template(payload, template, box.material, box.center, Vec3(0.0, 0.0, 0.0), -1)

    def _append_mesh(self, payload: array, mesh: Mesh, settings: RenderSettings) -> None:
        key = (id(mesh), settings.smooth_shading)
        cached_entry = self._mesh_templates.get(key)
        cached: tuple[_Template, Material] | None
        if cached_entry is not None and cached_entry[0]() is mesh:
            cached = cached_entry[1]
        else:
            if cached_entry is not None:
                self._mesh_templates.pop(key, None)
            triangles = mesh.to_triangles()
            cached = self._template_for_homogeneous_mesh(triangles, settings)
            self._remember_mesh_template(key, mesh, cached)
        if cached is None:
            self._append_triangles(payload, mesh.to_triangles(), settings)
            return
        template, material = cached
        self._append_template_with_texture_policy(payload, template, material)

    def _append_transformed_mesh(self, payload: array, instance: TransformedMesh, settings: RenderSettings) -> None:
        for template, material, use_texture in self._transformed_mesh_groups(instance.mesh, settings):
            texture_index = self.texture_index_for(material) if use_texture else -1
            _append_template(payload, template, material, instance.center, instance.rotation, texture_index)

    def _transformed_mesh_groups(
        self,
        mesh: Mesh,
        settings: RenderSettings,
    ) -> tuple[tuple[_Template, Material, bool], ...]:
        key = (id(mesh), settings.smooth_shading)
        cached_entry = self._transformed_mesh_templates.get(key)
        if cached_entry is not None and cached_entry[0]() is mesh:
            return cached_entry[1]
        if cached_entry is not None:
            self._transformed_mesh_templates.pop(key, None)

        groups: list[tuple[Material, list[Triangle]]] = []
        for triangle in mesh.triangles:
            for material, material_triangles in groups:
                if triangle.material == material:
                    material_triangles.append(triangle)
                    break
            else:
                groups.append((triangle.material, [triangle]))

        templates: list[tuple[_Template, Material, bool]] = []
        for material, triangles in groups:
            use_texture = material.texture is not None and all(triangle.has_texture_coordinates() for triangle in triangles)
            templates.append((_template_from_triangles(triangles, smooth_shading=settings.smooth_shading), material, use_texture))
        cached = tuple(templates)
        self._remember_transformed_mesh_templates(key, mesh, cached)
        return cached

    def _remember_transformed_mesh_templates(
        self,
        key: tuple[int, bool],
        mesh: Mesh,
        cached: tuple[tuple[_Template, Material, bool], ...],
    ) -> None:
        try:
            mesh_ref = ref(mesh)
        except TypeError:
            return
        if len(self._transformed_mesh_templates) > 128:
            for stale_key, (cached_ref, _cached_template) in tuple(self._transformed_mesh_templates.items()):
                if cached_ref() is None:
                    self._transformed_mesh_templates.pop(stale_key, None)
            if len(self._transformed_mesh_templates) > 128:
                self._transformed_mesh_templates.clear()
        self._transformed_mesh_templates[key] = (mesh_ref, cached)

    def _remember_mesh_template(
        self,
        key: tuple[int, bool],
        mesh: Mesh,
        cached: tuple[_Template, Material] | None,
    ) -> None:
        try:
            mesh_ref = ref(mesh)
        except TypeError:
            return
        if len(self._mesh_templates) > 128:
            for stale_key, (cached_ref, _cached_template) in tuple(self._mesh_templates.items()):
                if cached_ref() is None:
                    self._mesh_templates.pop(stale_key, None)
            if len(self._mesh_templates) > 128:
                self._mesh_templates.clear()
        self._mesh_templates[key] = (mesh_ref, cached)

    def _template_for_homogeneous_mesh(
        self,
        triangles: tuple[Triangle, ...],
        settings: RenderSettings,
    ) -> tuple[_Template, Material] | None:
        if not triangles:
            return None
        material = triangles[0].material
        if any(triangle.material != material for triangle in triangles):
            return None
        if material.texture is not None and any(not triangle.has_texture_coordinates() for triangle in triangles):
            return None
        return (_template_from_triangles(triangles, smooth_shading=settings.smooth_shading), material)

    def _append_template_with_texture_policy(self, payload: array, template: _Template, material: Material) -> None:
        texture_index = self.texture_index_for(material)
        _append_template(payload, template, material, Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), texture_index)

    def _append_triangles(self, payload: array, triangles: Iterable[Triangle], settings: RenderSettings) -> None:
        for triangle in triangles:
            smooth = settings.smooth_shading and triangle.has_vertex_normals()
            if smooth:
                assert triangle.normal_a is not None and triangle.normal_b is not None and triangle.normal_c is not None
                normal_a = (triangle.normal_a.x, triangle.normal_a.y, triangle.normal_a.z)
                normal_b = (triangle.normal_b.x, triangle.normal_b.y, triangle.normal_b.z)
                normal_c = (triangle.normal_c.x, triangle.normal_c.y, triangle.normal_c.z)
            else:
                normal = _triangle_normal_components(triangle)
                normal_a = normal_b = normal_c = normal
            texture_index = self.texture_index_for(triangle.material) if triangle.has_texture_coordinates() else -1
            _append_vertex_components(payload, triangle.a.x, triangle.a.y, triangle.a.z, *normal_a, triangle.material, triangle.uv_a, texture_index=texture_index)
            _append_vertex_components(payload, triangle.b.x, triangle.b.y, triangle.b.z, *normal_b, triangle.material, triangle.uv_b, texture_index=texture_index)
            _append_vertex_components(payload, triangle.c.x, triangle.c.y, triangle.c.z, *normal_c, triangle.material, triangle.uv_c, texture_index=texture_index)

    def _append_triangle_wireframe(self, payload: array, triangle: Triangle) -> None:
        self._append_line(payload, triangle.a, triangle.b, triangle.material)
        self._append_line(payload, triangle.b, triangle.c, triangle.material)
        self._append_line(payload, triangle.c, triangle.a, triangle.material)

    def _append_line(self, payload: array, start: Vec3, end: Vec3, material: Material) -> None:
        normal = Vec3(0.0, 1.0, 0.0)
        _append_vertex(payload, start, normal, material, None, force_emissive=True)
        _append_vertex(payload, end, normal, material, None, force_emissive=True)

    def _append_line_set(self, payload: array, line_set: LineSet3) -> None:
        material = line_set.material
        color = material.color.to_floats()
        emission = material.emission.to_floats()
        material_values = (
            min(1.0, color[0] + emission[0]),
            min(1.0, color[1] + emission[1]),
            min(1.0, color[2] + emission[2]),
            0.0,
            0.0,
            1.0,
            0.0,
        )
        tail = (
            0.0,
            1.0,
            0.0,
            color[0],
            color[1],
            color[2],
            *material_values,
            0.0,
            0.0,
            -1.0,
        )
        for start, end in line_set.segments:
            payload.extend((start.x, start.y, start.z, *tail))
            payload.extend((end.x, end.y, end.z, *tail))

    def texture_index_for(self, material: Material) -> int:
        texture = material.texture
        if texture is None:
            return -1
        key = id(texture)
        existing = self._frame_texture_slots.get(key)
        if existing is not None:
            return existing
        if len(self.active_textures) >= _MAX_TEXTURES:
            return -1
        index = len(self.active_textures)
        self._frame_texture_slots[key] = index
        self.active_textures.append(texture)
        return index


class _FrameClock:
    def __init__(self) -> None:
        self._last = perf_counter()

    def tick(self, fps: int | float = 0) -> float:
        target = 1.0 / float(fps) if fps and fps > 0 else 0.0
        now = perf_counter()
        elapsed = now - self._last
        if target > elapsed:
            sleep(target - elapsed)
            now = perf_counter()
            elapsed = now - self._last
        self._last = now
        return elapsed * 1000.0


class _PixelLiveRenderer:
    """Render py_3d scenes into the built-in live pixel window."""

    backend = "pixel"

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str = "py_3d",
        vsync: bool = True,
        resizable: bool = True,
    ) -> None:
        self.window = PixelWindow(width, height, title=title, fit_window=True)
        self.engine = RenderEngine(GPURenderer(allow_cpu_fallback=True))
        self.stats = LiveFrameStats(0.0, 0.0, 0, 0)
        self.mouse_captured = False
        self.menu = LiveMenu()
        self.hud = LiveHUD()
        self.show_crosshair = True
        self._closed = False
        self._last_frame_size = (max(1, int(width)), max(1, int(height)))
        self._started = perf_counter()
        self._menu_surface_cache_key = None
        self._menu_surface_cache: tuple[PixelBuffer, int, int] | None = None
        self._menu_blur_cache: PixelBuffer | None = None
        self._menu_blur_cache_at = 0.0
        self._menu_was_visible = self.menu.visible

    @property
    def size(self) -> tuple[int, int]:
        return self._last_frame_size

    def set_title(self, title: str) -> None:
        self.window.set_title(title)

    def resize(self, width: int, height: int) -> None:
        self._last_frame_size = (max(1, int(width)), max(1, int(height)))
        self._menu_surface_cache_key = None
        self._menu_surface_cache = None
        self._menu_blur_cache = None

    def handle_resize_event(self, event) -> bool:
        if getattr(event, "kind", "") != "resize":
            return False
        size = getattr(event, "size", (0, 0)) or getattr(event, "pos", (0, 0))
        self.resize(size[0], size[1])
        return True

    def frame_clock(self):
        return _FrameClock()

    def ticks(self) -> int:
        return int((perf_counter() - self._started) * 1000.0)

    def events(self):
        if self._closed or self.window.closed:
            return [WindowEvent("quit")]
        try:
            raw_events = self.window.poll_events()
        except Exception:
            self._closed = True
            return [WindowEvent("quit")]
        events = tuple(self._map_window_event(event) for event in raw_events)
        if any(event.kind == "quit" for event in events):
            self._closed = True
        return events

    def event_key(self, event) -> str:
        return event.key

    def event_mouse_rel(self, event) -> tuple[int, int]:
        return event.rel

    def event_mouse_button(self, event) -> int:
        return event.button

    def event_mouse_wheel_y(self, event) -> int:
        return event.y

    def is_quit_event(self, event) -> bool:
        return event.kind == "quit"

    def is_menu_pointer_event(self, event) -> bool:
        return event.kind in {"motion", "button", "wheel"}

    def is_mouse_button_down_event(self, event, button: int | None = None) -> bool:
        if event.kind != "button":
            return False
        return button is None or event.button == button

    def is_mouse_motion_event(self, event) -> bool:
        return event.kind == "motion"

    def is_mouse_wheel_event(self, event) -> bool:
        return event.kind == "wheel"

    def is_key_down_event(self, event) -> bool:
        return event.kind == "key_down"

    def is_key_up_event(self, event) -> bool:
        return event.kind == "key_up"

    def handle_menu_mouse_event(self, event) -> str | None:
        return self.menu.handle_pointer_event(event)

    def handle_menu_key(self, key: str) -> str | None:
        action = self.menu.handle_key_name(key)
        if action == "opened":
            self.set_mouse_captured(False)
        return action

    def key_matches(self, key: str, *names: str) -> bool:
        normalized = self._normalize_key(key)
        return any(normalized == self._normalize_key(name) for name in names)

    def set_mouse_captured(self, captured: bool) -> None:
        self.mouse_captured = bool(captured)

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> LiveFrameStats:
        build_start = perf_counter()
        frame = self.engine.render(scene, camera, settings)
        build_seconds = perf_counter() - build_start
        self._last_frame_size = (frame.width, frame.height)

        draw_start = perf_counter()
        if self.hud.visible:
            self._draw_hud(frame)
        if self.show_crosshair and not self.menu.visible:
            self._draw_crosshair(frame)
        if self.menu.visible != self._menu_was_visible:
            self._menu_blur_cache = None
            self._menu_was_visible = self.menu.visible
        if self.menu.visible:
            if self.menu.background_blur:
                self._apply_menu_blur(frame)
            self._draw_menu(frame)
        self.window.show(frame)
        draw_seconds = perf_counter() - draw_start

        self.stats = LiveFrameStats(build_seconds, draw_seconds, 0, 0)
        return self.stats

    def close(self) -> None:
        self._closed = True
        try:
            self.window.close()
        except Exception:
            pass

    def _map_window_event(self, event: WindowEvent) -> WindowEvent:
        if event.kind not in {"motion", "button"}:
            return event
        frame_width, frame_height = self._last_frame_size
        window_width, window_height = self.window.size
        pos = self._map_position(event.pos[0], event.pos[1])
        rel = (
            int(event.rel[0] * frame_width / max(1, window_width)),
            int(event.rel[1] * frame_height / max(1, window_height)),
        )
        return WindowEvent(event.kind, key=event.key, pos=pos, rel=rel, button=event.button, y=event.y, size=event.size)

    def _map_position(self, x: int, y: int) -> tuple[int, int]:
        frame_width, frame_height = self._last_frame_size
        window_width, window_height = self.window.size
        return (
            max(0, min(frame_width - 1, int(x * frame_width / max(1, window_width)))),
            max(0, min(frame_height - 1, int(y * frame_height / max(1, window_height)))),
        )

    @staticmethod
    def _normalize_key(key: str) -> str:
        cleaned = key.lower().replace("-", "_")
        aliases = {
            "esc": "escape",
            "return": "return",
            "enter": "return",
            "kp_enter": "return",
            "shift_l": "lshift",
            "shift_r": "rshift",
            "control_l": "lctrl",
            "control_r": "rctrl",
            "ctrl_l": "lctrl",
            "ctrl_r": "rctrl",
            "bracketleft": "leftbracket",
            "bracketright": "rightbracket",
            "prior": "pageup",
            "next": "pagedown",
        }
        return aliases.get(cleaned, cleaned)

    def _draw_menu(self, frame: PixelBuffer) -> None:
        key = self._menu_cache_key(frame.width, frame.height)
        if key == self._menu_surface_cache_key and self._menu_surface_cache is not None:
            menu, left, top = self._menu_surface_cache
        else:
            menu, left, top = render_live_menu_surface(self.menu, frame.width, frame.height)
            self._menu_surface_cache_key = key
            self._menu_surface_cache = (menu, left, top)
        _blit_buffer(frame, menu, left, top)

    def _menu_cache_key(self, width: int, height: int):
        menu = self.menu
        options = tuple((option.action, option.label, option.detail, option.group, option.enabled) for option in menu.options)
        scroll = tuple(sorted(menu.scroll_offsets.items()))
        theme = (
            menu.theme.panel,
            menu.theme.border,
            menu.theme.text,
            menu.theme.muted_text,
            menu.theme.row,
            menu.theme.row_selected,
            menu.theme.button,
            menu.theme.button_selected,
            menu.theme.button_border,
            menu.theme.active_border,
        )
        return (width, height, menu.visible, menu.title, menu.selected_index, menu.active_group, options, scroll, theme)

    def _apply_menu_blur(self, frame: PixelBuffer) -> None:
        now = perf_counter()
        cache = self._menu_blur_cache
        if (
            cache is not None
            and cache.width == frame.width
            and cache.height == frame.height
        ):
            frame.blit(cache, 0, 0)
            return
        self._blur_buffer(frame)
        self._menu_blur_cache = PixelBuffer.from_rgb_bytes(frame.width, frame.height, bytearray(frame.to_rgb_bytes()))
        self._menu_blur_cache_at = now

    def _draw_crosshair(self, frame: PixelBuffer) -> None:
        center_x = frame.width // 2
        center_y = frame.height // 2
        color = Color(235, 244, 255)
        for offset in range(-8, 9):
            if abs(offset) <= 1:
                continue
            frame.set_pixel(center_x + offset, center_y, color)
            frame.set_pixel(center_x, center_y + offset, color)
        for y in range(center_y - 1, center_y + 2):
            for x in range(center_x - 1, center_x + 2):
                frame.set_pixel(x, y, color)

    def _draw_hud(self, frame: PixelBuffer) -> None:
        seconds = perf_counter() - self._started
        for element in self.hud.elements:
            if isinstance(element, HUDRect):
                _blend_rect(frame, element.position, element.size, element.color, element.alpha)
            elif isinstance(element, HUDText):
                if element.background is not None:
                    text_width, text_height = draw.text_size(element.text, scale=element.scale)
                    _blend_rect(
                        frame,
                        element.position,
                        (text_width + element.padding * 2, text_height + element.padding * 2),
                        element.background,
                        element.alpha,
                    )
                draw.text(
                    frame,
                    (element.position[0] + element.padding, element.position[1] + element.padding),
                    element.text,
                    element.color,
                    scale=element.scale,
                )
            elif isinstance(element, HUDImage):
                image = element.image if element.scale <= 1 else element.image.resized_nearest(element.image.width * element.scale, element.image.height * element.scale)
                _blit_buffer(frame, image, element.position[0], element.position[1], alpha=element.alpha)
            elif isinstance(element, HUDAnimation):
                image = element.frame_at(seconds)
                if image is None:
                    continue
                prepared = image if element.scale <= 1 else image.resized_nearest(image.width * element.scale, image.height * element.scale)
                _blit_buffer(frame, prepared, element.position[0], element.position[1], alpha=element.alpha)

    def _blur_buffer(self, frame: PixelBuffer) -> None:
        if _blur_buffer_numpy(frame, radius=3):
            return
        source = frame.copy()
        radius = 3
        for y in range(frame.height):
            for x in range(frame.width):
                red = green = blue = count = 0
                for yy in range(max(0, y - radius), min(frame.height, y + radius + 1)):
                    for xx in range(max(0, x - radius), min(frame.width, x + radius + 1)):
                        pixel = source.get_pixel(xx, yy)
                        red += pixel.r
                        green += pixel.g
                        blue += pixel.b
                        count += 1
                frame.set_pixel(x, y, (red // count, green // count, blue // count))


class _GLFWModernGLLiveRenderer:
    """Direct GLFW/ModernGL live renderer with a GPU-backed swapchain."""

    backend = "opengl"

    def __init__(
        self,
        width: int,
        height: int,
        *,
        title: str = "py_3d",
        vsync: bool = True,
        resizable: bool = True,
    ) -> None:
        try:
            import glfw
            import moderngl
        except Exception as exc:
            raise RuntimeError("ModernGL live rendering requires moderngl and glfw") from exc

        if not glfw.init():
            raise RuntimeError("GLFW could not initialize an OpenGL window")

        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE if resizable else glfw.FALSE)
        self._window = None
        for major, minor in ((4, 3), (3, 3)):
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, major)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, minor)
            self._window = glfw.create_window(max(1, int(width)), max(1, int(height)), str(title), None, None)
            if self._window:
                break
        if not self._window:
            raise RuntimeError("GLFW could not create an OpenGL window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1 if vsync else 0)
        self._glfw = glfw
        self._moderngl = moderngl
        self._set_window_icon()
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.depth_func = "<"
        self.scene_program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.overlay_program = self.ctx.program(vertex_shader=OVERLAY_VERTEX_SHADER, fragment_shader=OVERLAY_FRAGMENT_SHADER)
        self.water_program = None
        self.fluid_puddle_program = None
        self.fluid_volume_program = None
        if getattr(self.ctx, "version_code", 0) >= 430:
            try:
                self.water_program = self.ctx.program(vertex_shader=PARTICLE_WATER_VERTEX_SHADER, fragment_shader=PARTICLE_WATER_FRAGMENT_SHADER)
            except Exception:
                self.water_program = None
            try:
                self.fluid_puddle_program = self.ctx.program(vertex_shader=PARTICLE_FLUID_PUDDLE_VERTEX_SHADER, fragment_shader=PARTICLE_FLUID_PUDDLE_FRAGMENT_SHADER)
            except Exception:
                self.fluid_puddle_program = None
            try:
                self.fluid_volume_program = self.ctx.program(vertex_shader=PARTICLE_FLUID_VOLUME_VERTEX_SHADER, fragment_shader=PARTICLE_FLUID_VOLUME_FRAGMENT_SHADER)
            except Exception:
                self.fluid_volume_program = None
        self._set_uniform(self.scene_program, "u_textures", 0)
        self._set_uniform(self.overlay_program, "u_texture", 0)

        self._triangle_buffer = self.ctx.buffer(reserve=4, dynamic=True)
        self._line_buffer = self.ctx.buffer(reserve=4, dynamic=True)
        self._overlay_buffer = self.ctx.buffer(reserve=6 * 4 * 4, dynamic=True)
        self._triangle_capacity = 4
        self._line_capacity = 4
        self._resident_entries: dict[tuple[int, str], _ResidentObjectEntry] = {}
        self._resident_layout_signature: tuple | None = None
        self._resident_texture_key: tuple | None = None
        self._resident_triangle_vertices = 0
        self._resident_line_vertices = 0
        self._triangle_vao = self.ctx.vertex_array(
            self.scene_program,
            [(self._triangle_buffer, _VERTEX_FORMAT, *_VERTEX_ATTRIBUTES)],
        )
        self._line_vao = self.ctx.vertex_array(
            self.scene_program,
            [(self._line_buffer, _VERTEX_FORMAT, *_VERTEX_ATTRIBUTES)],
        )
        self._overlay_vao = self.ctx.vertex_array(
            self.overlay_program,
            [(self._overlay_buffer, _OVERLAY_VERTEX_FORMAT, *_OVERLAY_VERTEX_ATTRIBUTES)],
        )
        self._water_surface_vaos: dict[tuple[str, float], tuple[object, object, int]] = {}
        self._fluid_puddle_vaos: dict[tuple[str, tuple[float, float, float, float]], tuple[object, object, int]] = {}
        self._fluid_volume_buffer = None
        self._fluid_volume_vao = None
        self.builder = LiveSceneBatchBuilder()
        self.stats = LiveFrameStats(0.0, 0.0, 0, 0)
        self.mouse_captured = False
        self._mouse_capture_requested = False
        self._pointer_activated = False
        self._window_focused = bool(glfw.get_window_attrib(self._window, glfw.FOCUSED))
        self._window_iconified = bool(glfw.get_window_attrib(self._window, glfw.ICONIFIED))
        self.menu = LiveMenu()
        self.hud = LiveHUD()
        self.show_crosshair = True
        self._closed = False
        self._started = perf_counter()
        self._events: list[WindowEvent] = []
        self._last_mouse: tuple[int, int] | None = None
        self._last_frame_size = self._framebuffer_size()
        self._window_size = self._safe_window_size()
        self._scene_texture = None
        self._scene_texture_key = None
        self._scene_texture_size = 0
        self._texture_rgba_cache: dict[tuple[int, int, int, int, int], bytes] = {}
        self._surface_rgba_cache: dict[tuple, tuple[int, int, bytes]] = {}
        self._overlay_textures: dict[str, object] = {}
        self._overlay_sizes: dict[str, tuple[int, int]] = {}
        self._overlay_signatures: dict[str, tuple[int, int, object]] = {}
        self._active_overlay_slots: set[str] = set()
        self._menu_surface_cache_key = None
        self._menu_surface_cache: tuple[PixelBuffer, int, int, bytes] | None = None
        self._menu_was_visible = self.menu.visible
        self._install_callbacks()

    @property
    def size(self) -> tuple[int, int]:
        return self._framebuffer_size()

    def set_title(self, title: str) -> None:
        self._glfw.set_window_title(self._window, str(title))

    def _set_window_icon(self) -> None:
        set_window_icon = getattr(self._glfw, "set_window_icon", None)
        if set_window_icon is None:
            return
        try:
            icon = load_framework_icon_rgba()
            set_window_icon(self._window, 1, icon)
        except Exception:
            return

    def resize(self, width: int, height: int) -> None:
        self._glfw.set_window_size(self._window, max(1, int(width)), max(1, int(height)))
        self._last_frame_size = self._framebuffer_size()
        self._invalidate_surface_caches()

    def handle_resize_event(self, event) -> bool:
        if getattr(event, "kind", "") != "resize":
            return False
        self._last_frame_size = self._framebuffer_size()
        self._window_size = self._safe_window_size()
        self._invalidate_surface_caches()
        return True

    def frame_clock(self):
        return _FrameClock()

    def ticks(self) -> int:
        return int((perf_counter() - self._started) * 1000.0)

    def events(self):
        if self._closed:
            return [WindowEvent("quit")]
        self._glfw.poll_events()
        events = tuple(self._map_window_event(event) for event in self._events)
        self._events.clear()
        if self._glfw.window_should_close(self._window):
            self.release_mouse_capture()
            events = events + (WindowEvent("quit"),)
        if any(event.kind == "quit" for event in events):
            self._closed = True
        return events

    def event_key(self, event) -> str:
        return event.key

    def event_mouse_rel(self, event) -> tuple[int, int]:
        return event.rel

    def event_mouse_button(self, event) -> int:
        return event.button

    def event_mouse_wheel_y(self, event) -> int:
        return event.y

    def is_quit_event(self, event) -> bool:
        return event.kind == "quit"

    def is_menu_pointer_event(self, event) -> bool:
        return event.kind in {"motion", "button", "wheel"}

    def is_mouse_button_down_event(self, event, button: int | None = None) -> bool:
        if event.kind != "button":
            return False
        return button is None or event.button == button

    def is_mouse_motion_event(self, event) -> bool:
        return event.kind == "motion"

    def is_mouse_wheel_event(self, event) -> bool:
        return event.kind == "wheel"

    def is_key_down_event(self, event) -> bool:
        return event.kind == "key_down"

    def is_key_up_event(self, event) -> bool:
        return event.kind == "key_up"

    def handle_menu_mouse_event(self, event) -> str | None:
        return self.menu.handle_pointer_event(event)

    def handle_menu_key(self, key: str) -> str | None:
        action = self.menu.handle_key_name(key)
        if action == "opened":
            self.set_mouse_captured(False)
        return action

    def key_matches(self, key: str, *names: str) -> bool:
        normalized = self._normalize_key(key)
        return any(normalized == self._normalize_key(name) for name in names)

    def set_mouse_captured(self, captured: bool) -> None:
        self._mouse_capture_requested = bool(captured)
        self._sync_mouse_capture()

    def release_mouse_capture(self) -> None:
        self._mouse_capture_requested = False
        self._apply_mouse_capture(False)

    def _sync_mouse_capture(self) -> None:
        self._apply_mouse_capture(self._can_capture_mouse())

    def _can_capture_mouse(self) -> bool:
        return (
            self._mouse_capture_requested
            and self._pointer_activated
            and self._window_focused
            and not self._window_iconified
            and not self.menu.visible
        )

    def _apply_mouse_capture(self, captured: bool) -> None:
        active = bool(captured)
        if self.mouse_captured == active:
            return
        self.mouse_captured = active
        mode = self._glfw.CURSOR_DISABLED if active else self._glfw.CURSOR_NORMAL
        self._glfw.set_input_mode(self._window, self._glfw.CURSOR, mode)
        self._last_mouse = None

    def render(self, scene: Scene, camera: Camera, settings: RenderSettings) -> LiveFrameStats:
        build_start = perf_counter()
        triangle_vertices, line_vertices = self._update_resident_scene_buffers(scene, settings)
        build_seconds = perf_counter() - build_start

        draw_start = perf_counter()
        width, height = self._framebuffer_size()
        self._last_frame_size = (width, height)
        self.ctx.viewport = (0, 0, width, height)
        background = Color.from_value(settings.background).to_floats()
        self.ctx.clear(background[0], background[1], background[2], 1.0, depth=1.0)
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.disable(self._moderngl.BLEND)
        if settings.cull_backfaces:
            self.ctx.enable(self._moderngl.CULL_FACE)
        else:
            self.ctx.disable(self._moderngl.CULL_FACE)

        self._apply_scene_uniforms(scene, camera, settings, width, height)
        self._upload_scene_textures(settings)
        if triangle_vertices > 0:
            self._triangle_vao.render(mode=self._moderngl.TRIANGLES, vertices=triangle_vertices)
        if line_vertices > 0:
            self._line_vao.render(mode=self._moderngl.LINES, vertices=line_vertices)
        self._draw_particle_water_surfaces(scene, camera, settings, width, height)
        self._draw_particle_fluid_puddles(scene, camera, settings, width, height)
        self._draw_particle_fluid_volumes(scene, camera, settings, width, height)

        self.ctx.disable(self._moderngl.DEPTH_TEST)
        self.ctx.enable(self._moderngl.BLEND)
        self.ctx.blend_func = self._moderngl.SRC_ALPHA, self._moderngl.ONE_MINUS_SRC_ALPHA
        self._begin_overlays()
        self._draw_scene_bulletins(scene, camera, width, height)
        if self.hud.visible:
            self._draw_hud(width, height)
        if self.show_crosshair and not self.menu.visible:
            self._draw_crosshair(width, height)
        if self.menu.visible != self._menu_was_visible:
            self._menu_was_visible = self.menu.visible
            self._invalidate_surface_caches()
            if self.menu.visible:
                self._apply_mouse_capture(False)
            else:
                self._sync_mouse_capture()
        if self.menu.visible:
            if self.menu.background_blur:
                self._draw_overlay_solid(
                    "menu_dim",
                    0,
                    0,
                    width,
                    height,
                    (0, 0, 0),
                    0.22,
                )
            self._draw_menu(width, height)
        self._end_overlays()
        self._glfw.swap_buffers(self._window)
        draw_seconds = perf_counter() - draw_start

        self.stats = LiveFrameStats(build_seconds, draw_seconds, triangle_vertices, line_vertices)
        return self.stats

    def _draw_particle_water_surfaces(self, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        if self.water_program is None or settings.wireframe:
            return
        surfaces = [obj for obj in scene.objects if isinstance(obj, ParticleWaterSurface)]
        if not surfaces:
            return
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.enable(self._moderngl.BLEND)
        self.ctx.blend_func = self._moderngl.SRC_ALPHA, self._moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(self._moderngl.CULL_FACE)
        depth_mask_supported = True
        try:
            self.ctx.depth_mask = False
        except Exception:
            depth_mask_supported = False
        try:
            for surface in surfaces:
                binder = getattr(surface.world, "bind_particle_buffer", None)
                if not callable(binder) or not binder(2):
                    continue
                buffer, vao, vertices = self._particle_water_surface_vao(surface)
                del buffer
                self._apply_particle_water_uniforms(surface, scene, camera, settings, width, height)
                vao.render(mode=self._moderngl.TRIANGLES, vertices=vertices)
        finally:
            if depth_mask_supported:
                try:
                    self.ctx.depth_mask = True
                except Exception:
                    pass

    def _particle_water_surface_vao(self, surface: ParticleWaterSurface):
        if self.water_program is None:
            raise RuntimeError("particle water shader is unavailable")
        key = (surface.quality, round(surface.radius, 5))
        cached = self._water_surface_vaos.get(key)
        if cached is not None:
            return cached
        vertices = _particle_water_surface_vertices(surface.quality, surface.radius)
        buffer = self.ctx.buffer(array("f", vertices).tobytes())
        vao = self.ctx.vertex_array(
            self.water_program,
            [(buffer, _WATER_VERTEX_FORMAT, *_WATER_VERTEX_ATTRIBUTES)],
        )
        cached = (buffer, vao, len(vertices) // 5)
        self._water_surface_vaos[key] = cached
        return cached

    def _apply_particle_water_uniforms(self, surface: ParticleWaterSurface, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        assert self.water_program is not None
        right, true_up, forward = camera.basis()
        program = self.water_program
        material = surface.material
        color = Color.from_value(material.color).to_floats()
        particle_count = len(getattr(surface.world, "particles", ()))
        particle_base_y = surface.particle_base_y if surface.particle_base_y is not None else surface.center.y
        rest_distance = float(getattr(surface.world, "rest_distance", surface.radius * 0.03))
        kernel_radius = surface.kernel_radius if surface.kernel_radius is not None else max(rest_distance * 3.4, surface.radius * 0.095)
        self._set_uniform(program, "u_camera_position", camera.position.as_tuple())
        self._set_uniform(program, "u_camera_right", right.as_tuple())
        self._set_uniform(program, "u_camera_up", true_up.as_tuple())
        self._set_uniform(program, "u_camera_forward", forward.as_tuple())
        self._set_uniform(program, "u_focal", 1.0 / tan(radians(camera.fov_degrees) / 2.0))
        self._set_uniform(program, "u_aspect", width / max(1, height))
        self._set_uniform(program, "u_near", camera.near)
        self._set_uniform(program, "u_far", camera.far)
        self._set_uniform(program, "u_center", surface.center.as_tuple())
        self._set_uniform(program, "u_radius", float(surface.radius))
        self._set_uniform(program, "u_level", float(surface.center.y))
        self._set_uniform(program, "u_particle_base_y", float(particle_base_y))
        self._set_uniform(program, "u_kernel_radius", float(kernel_radius))
        self._set_uniform(program, "u_time", float(surface.time))
        self._set_uniform(program, "u_particle_count", int(particle_count))
        vortex_center = surface.vortex_center if surface.vortex_center is not None else surface.center
        self._set_uniform(program, "u_vortex_center", vortex_center.as_tuple())
        self._set_uniform(program, "u_vortex_strength", float(surface.vortex_strength))
        bubbles = tuple(surface.bubbles[:8])
        bubble_payload: list[float] = []
        for index in range(8):
            if index < len(bubbles):
                bubble_payload.extend(bubbles[index])
            else:
                bubble_payload.extend((0.0, 0.0, 0.0, 0.0))
        self._set_uniform(program, "u_bubble_count", len(bubbles))
        self._set_uniform(program, "u_bubbles", tuple(bubble_payload))
        self._set_uniform(program, "u_ambient", settings.ambient)
        self._set_uniform(program, "u_gamma", settings.gamma)
        self._set_uniform(program, "u_light_wrap", settings.light_wrap)
        self._set_uniform(program, "u_bounce_light", settings.bounce_light)
        self._set_uniform(program, "u_tone_mapping", bool(settings.tone_mapping))
        self._set_uniform(program, "u_reflection_bounces", max(2, int(settings.reflection_bounces)))
        self._set_uniform(program, "u_material_color", color)
        self._set_uniform(program, "u_material", (material.diffuse, material.specular, material.shininess, material.reflectivity))
        self._set_uniform(program, "u_alpha", max(0.25, min(0.82, 0.38 + material.light_transmission * 0.34)))
        self._apply_light_uniforms(scene, program)

    def _draw_particle_fluid_puddles(self, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        if self.fluid_puddle_program is None or settings.wireframe:
            return
        puddles = [obj for obj in scene.objects if isinstance(obj, ParticleFluidPuddle)]
        if not puddles:
            return
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.enable(self._moderngl.BLEND)
        self.ctx.blend_func = self._moderngl.SRC_ALPHA, self._moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(self._moderngl.CULL_FACE)
        depth_mask_supported = True
        try:
            self.ctx.depth_mask = False
        except Exception:
            depth_mask_supported = False
        try:
            for puddle in puddles:
                binder = getattr(puddle.world, "bind_particle_buffer", None)
                if not callable(binder) or not binder(2):
                    continue
                _buffer, vao, vertices = self._particle_fluid_puddle_vao(puddle)
                self._apply_particle_fluid_puddle_uniforms(puddle, scene, camera, settings, width, height)
                vao.render(mode=self._moderngl.TRIANGLES, vertices=vertices)
        finally:
            if depth_mask_supported:
                try:
                    self.ctx.depth_mask = True
                except Exception:
                    pass

    def _particle_fluid_puddle_vao(self, puddle: ParticleFluidPuddle):
        if self.fluid_puddle_program is None:
            raise RuntimeError("particle fluid puddle shader is unavailable")
        bounds_key = (
            round(puddle.bounds_min.x, 4),
            round(puddle.bounds_min.z, 4),
            round(puddle.bounds_max.x, 4),
            round(puddle.bounds_max.z, 4),
        )
        key = (puddle.quality, bounds_key)
        cached = self._fluid_puddle_vaos.get(key)
        if cached is not None:
            return cached
        vertices = _particle_fluid_puddle_vertices(puddle.quality, puddle.bounds_min, puddle.bounds_max)
        buffer = self.ctx.buffer(array("f", vertices).tobytes())
        vao = self.ctx.vertex_array(
            self.fluid_puddle_program,
            [(buffer, "2f 2f", "in_xz", "in_uv")],
        )
        cached = (buffer, vao, len(vertices) // 4)
        self._fluid_puddle_vaos[key] = cached
        return cached

    def _apply_particle_fluid_puddle_uniforms(self, puddle: ParticleFluidPuddle, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        assert self.fluid_puddle_program is not None
        right, true_up, forward = camera.basis()
        program = self.fluid_puddle_program
        material = puddle.material
        color = Color.from_value(material.color).to_floats()
        rest_distance = float(getattr(puddle.world, "rest_distance", 0.05))
        kernel_radius = puddle.kernel_radius if puddle.kernel_radius is not None else max(rest_distance * 2.9, 0.085)
        self._set_uniform(program, "u_camera_position", camera.position.as_tuple())
        self._set_uniform(program, "u_camera_right", right.as_tuple())
        self._set_uniform(program, "u_camera_up", true_up.as_tuple())
        self._set_uniform(program, "u_camera_forward", forward.as_tuple())
        self._set_uniform(program, "u_focal", 1.0 / tan(radians(camera.fov_degrees) / 2.0))
        self._set_uniform(program, "u_aspect", width / max(1, height))
        self._set_uniform(program, "u_near", camera.near)
        self._set_uniform(program, "u_far", camera.far)
        self._set_uniform(program, "u_floor_y", float(puddle.floor_y))
        self._set_uniform(program, "u_kernel_radius", float(kernel_radius))
        self._set_uniform(program, "u_time", float(puddle.time))
        self._set_uniform(program, "u_particle_count", len(getattr(puddle.world, "particles", ())))
        self._set_uniform(program, "u_ambient", settings.ambient)
        self._set_uniform(program, "u_gamma", settings.gamma)
        self._set_uniform(program, "u_light_wrap", settings.light_wrap)
        self._set_uniform(program, "u_bounce_light", settings.bounce_light)
        self._set_uniform(program, "u_tone_mapping", bool(settings.tone_mapping))
        self._set_uniform(program, "u_reflection_bounces", max(2, int(settings.reflection_bounces)))
        self._set_uniform(program, "u_material_color", color)
        self._set_uniform(program, "u_material", (material.diffuse, material.specular, material.shininess, material.reflectivity))
        self._set_uniform(program, "u_alpha", max(0.24, min(0.82, 0.34 + material.light_transmission * 0.30)))
        self._set_uniform(program, "u_density_threshold", float(puddle.density_threshold))
        self._apply_light_uniforms(scene, program)

    def _draw_particle_fluid_volumes(self, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        if self.fluid_volume_program is None or settings.wireframe:
            return
        volumes = [obj for obj in scene.objects if isinstance(obj, ParticleFluidVolume)]
        if not volumes:
            return
        self.ctx.enable(self._moderngl.DEPTH_TEST)
        self.ctx.enable(self._moderngl.BLEND)
        self.ctx.blend_func = self._moderngl.SRC_ALPHA, self._moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.disable(self._moderngl.CULL_FACE)
        depth_mask_supported = True
        try:
            self.ctx.depth_mask = False
        except Exception:
            depth_mask_supported = False
        try:
            vao = self._particle_fluid_volume_vao()
            for volume in volumes:
                particle_count = len(getattr(volume.world, "particles", ()))
                if particle_count <= 0:
                    continue
                binder = getattr(volume.world, "bind_particle_buffer", None)
                if not callable(binder) or not binder(2):
                    continue
                self._apply_particle_fluid_volume_uniforms(volume, scene, camera, settings, width, height)
                vao.render(mode=self._moderngl.TRIANGLES, vertices=6, instances=particle_count)
        finally:
            if depth_mask_supported:
                try:
                    self.ctx.depth_mask = True
                except Exception:
                    pass

    def _particle_fluid_volume_vao(self):
        if self.fluid_volume_program is None:
            raise RuntimeError("particle fluid volume shader is unavailable")
        if self._fluid_volume_vao is not None:
            return self._fluid_volume_vao
        vertices = array("f", (-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0))
        self._fluid_volume_buffer = self.ctx.buffer(vertices.tobytes())
        self._fluid_volume_vao = self.ctx.vertex_array(
            self.fluid_volume_program,
            [(self._fluid_volume_buffer, _FLUID_VOLUME_VERTEX_FORMAT, *_FLUID_VOLUME_VERTEX_ATTRIBUTES)],
        )
        return self._fluid_volume_vao

    def _apply_particle_fluid_volume_uniforms(self, volume: ParticleFluidVolume, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        assert self.fluid_volume_program is not None
        right, true_up, forward = camera.basis()
        program = self.fluid_volume_program
        material = volume.material
        color = Color.from_value(material.color).to_floats()
        rest_distance = float(getattr(volume.world, "rest_distance", 0.05))
        particle_radius = volume.particle_radius if volume.particle_radius is not None else rest_distance * 0.62
        min_y = volume.min_y if volume.min_y is not None else -1.0e9
        self._set_uniform(program, "u_camera_position", camera.position.as_tuple())
        self._set_uniform(program, "u_camera_right", right.as_tuple())
        self._set_uniform(program, "u_camera_up", true_up.as_tuple())
        self._set_uniform(program, "u_camera_forward", forward.as_tuple())
        self._set_uniform(program, "u_focal", 1.0 / tan(radians(camera.fov_degrees) / 2.0))
        self._set_uniform(program, "u_aspect", width / max(1, height))
        self._set_uniform(program, "u_near", camera.near)
        self._set_uniform(program, "u_far", camera.far)
        self._set_uniform(program, "u_particle_radius", float(particle_radius))
        self._set_uniform(program, "u_min_y", float(min_y))
        self._set_uniform(program, "u_particle_count", len(getattr(volume.world, "particles", ())))
        self._set_uniform(program, "u_ambient", settings.ambient)
        self._set_uniform(program, "u_gamma", settings.gamma)
        self._set_uniform(program, "u_light_wrap", settings.light_wrap)
        self._set_uniform(program, "u_bounce_light", settings.bounce_light)
        self._set_uniform(program, "u_tone_mapping", bool(settings.tone_mapping))
        self._set_uniform(program, "u_reflection_bounces", max(2, int(settings.reflection_bounces)))
        self._set_uniform(program, "u_material_color", color)
        self._set_uniform(program, "u_material", (material.diffuse, material.specular, material.shininess, material.reflectivity))
        self._set_uniform(program, "u_alpha", max(0.18, min(0.74, 0.22 + material.light_transmission * 0.34)))
        self._apply_light_uniforms(scene, program)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.release_mouse_capture()
        for texture in tuple(self._overlay_textures.values()):
            try:
                texture.release()
            except Exception:
                pass
        if self._scene_texture is not None:
            try:
                self._scene_texture.release()
            except Exception:
                pass
        for buffer, vao, _vertices in tuple(self._water_surface_vaos.values()):
            for resource in (vao, buffer):
                try:
                    resource.release()
                except Exception:
                    pass
        self._water_surface_vaos.clear()
        for buffer, vao, _vertices in tuple(self._fluid_puddle_vaos.values()):
            for resource in (vao, buffer):
                try:
                    resource.release()
                except Exception:
                    pass
        self._fluid_puddle_vaos.clear()
        for resource in (self._fluid_volume_vao, self._fluid_volume_buffer):
            if resource is None:
                continue
            try:
                resource.release()
            except Exception:
                pass
        self._fluid_volume_vao = None
        self._fluid_volume_buffer = None
        for resource in (
            self._triangle_vao,
            self._line_vao,
            self._overlay_vao,
            self._triangle_buffer,
            self._line_buffer,
            self._overlay_buffer,
            self.scene_program,
            self.overlay_program,
            self.water_program,
            self.fluid_puddle_program,
            self.fluid_volume_program,
        ):
            if resource is None:
                continue
            try:
                resource.release()
            except Exception:
                pass
        try:
            self._glfw.destroy_window(self._window)
        except Exception:
            pass

    def _install_callbacks(self) -> None:
        self._glfw.set_window_size_callback(self._window, self._on_window_size)
        self._glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_size)
        self._glfw.set_window_close_callback(self._window, self._on_close)
        self._glfw.set_window_focus_callback(self._window, self._on_window_focus)
        self._glfw.set_window_iconify_callback(self._window, self._on_window_iconify)
        self._glfw.set_key_callback(self._window, self._on_key)
        self._glfw.set_cursor_pos_callback(self._window, self._on_cursor_pos)
        self._glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        self._glfw.set_scroll_callback(self._window, self._on_scroll)

    def _on_window_size(self, _window, width: int, height: int) -> None:
        self._window_size = (max(1, int(width)), max(1, int(height)))
        self._events.append(WindowEvent("resize", size=self._window_size))

    def _on_framebuffer_size(self, _window, width: int, height: int) -> None:
        self._last_frame_size = (max(1, int(width)), max(1, int(height)))
        self._invalidate_surface_caches()

    def _on_close(self, _window) -> None:
        self.release_mouse_capture()
        self._events.append(WindowEvent("quit"))

    def _on_window_focus(self, _window, focused: int) -> None:
        self._window_focused = bool(focused)
        if not self._window_focused:
            self._pointer_activated = False
            self.release_mouse_capture()
            return
        self._sync_mouse_capture()

    def _on_window_iconify(self, _window, iconified: int) -> None:
        self._window_iconified = bool(iconified)
        if self._window_iconified:
            self._pointer_activated = False
            self.release_mouse_capture()
            return
        self._sync_mouse_capture()

    def _on_key(self, _window, key: int, scancode: int, action: int, _mods: int) -> None:
        if action == self._glfw.PRESS or action == self._glfw.REPEAT:
            self._events.append(WindowEvent("key_down", key=self._glfw_key_name(key, scancode)))
        elif action == self._glfw.RELEASE:
            self._events.append(WindowEvent("key_up", key=self._glfw_key_name(key, scancode)))

    def _on_cursor_pos(self, _window, x: float, y: float) -> None:
        pos = (int(x), int(y))
        last = self._last_mouse or pos
        rel = (pos[0] - last[0], pos[1] - last[1])
        self._last_mouse = pos
        self._events.append(WindowEvent("motion", pos=pos, rel=rel))

    def _on_mouse_button(self, _window, button: int, action: int, _mods: int) -> None:
        x, y = self._glfw.get_cursor_pos(self._window)
        event_button = self._mouse_button_number(button)
        if action == self._glfw.PRESS:
            self._pointer_activated = True
            self._sync_mouse_capture()
            self._events.append(WindowEvent("button", pos=(int(x), int(y)), button=event_button))
        elif action == self._glfw.RELEASE:
            self._events.append(WindowEvent("button_up", pos=(int(x), int(y)), button=event_button))

    def _on_scroll(self, _window, _x: float, y: float) -> None:
        self._events.append(WindowEvent("wheel", y=1 if y > 0 else -1 if y < 0 else 0))

    def _mouse_button_number(self, button: int) -> int:
        mapping = {
            self._glfw.MOUSE_BUTTON_LEFT: 1,
            self._glfw.MOUSE_BUTTON_MIDDLE: 2,
            self._glfw.MOUSE_BUTTON_RIGHT: 3,
            self._glfw.MOUSE_BUTTON_4: 4,
            self._glfw.MOUSE_BUTTON_5: 5,
        }
        return mapping.get(button, int(button) + 1)

    def _map_window_event(self, event: WindowEvent) -> WindowEvent:
        if event.kind not in {"motion", "button", "button_up"}:
            return event
        frame_width, frame_height = self._last_frame_size
        window_width, window_height = self._window_size
        pos = self._map_position(event.pos[0], event.pos[1])
        rel = (
            int(event.rel[0] * frame_width / max(1, window_width)),
            int(event.rel[1] * frame_height / max(1, window_height)),
        )
        return WindowEvent(event.kind, key=event.key, pos=pos, rel=rel, button=event.button, y=event.y, size=event.size)

    def _map_position(self, x: int, y: int) -> tuple[int, int]:
        frame_width, frame_height = self._last_frame_size
        window_width, window_height = self._window_size
        return (
            max(0, min(frame_width - 1, int(x * frame_width / max(1, window_width)))),
            max(0, min(frame_height - 1, int(y * frame_height / max(1, window_height)))),
        )

    def _framebuffer_size(self) -> tuple[int, int]:
        width, height = self._glfw.get_framebuffer_size(self._window)
        return (max(1, int(width)), max(1, int(height)))

    def _safe_window_size(self) -> tuple[int, int]:
        width, height = self._glfw.get_window_size(self._window)
        return (max(1, int(width)), max(1, int(height)))

    def _glfw_key_name(self, key: int, scancode: int) -> str:
        name = self._glfw.get_key_name(key, scancode)
        if name:
            return name.lower()
        mapping = {
            self._glfw.KEY_BACKSPACE: "backspace",
            self._glfw.KEY_TAB: "tab",
            self._glfw.KEY_ENTER: "return",
            self._glfw.KEY_ESCAPE: "escape",
            self._glfw.KEY_SPACE: "space",
            self._glfw.KEY_PAGE_UP: "pageup",
            self._glfw.KEY_PAGE_DOWN: "pagedown",
            self._glfw.KEY_LEFT: "left",
            self._glfw.KEY_UP: "up",
            self._glfw.KEY_RIGHT: "right",
            self._glfw.KEY_DOWN: "down",
            self._glfw.KEY_LEFT_SHIFT: "lshift",
            self._glfw.KEY_RIGHT_SHIFT: "rshift",
            self._glfw.KEY_LEFT_CONTROL: "lctrl",
            self._glfw.KEY_RIGHT_CONTROL: "rctrl",
            self._glfw.KEY_LEFT_ALT: "lalt",
            self._glfw.KEY_RIGHT_ALT: "ralt",
            self._glfw.KEY_LEFT_BRACKET: "leftbracket",
            self._glfw.KEY_RIGHT_BRACKET: "rightbracket",
        }
        return mapping.get(key, f"key_{key}")

    @staticmethod
    def _normalize_key(key: str) -> str:
        cleaned = key.lower().replace("-", "_")
        aliases = {
            "esc": "escape",
            "return": "return",
            "enter": "return",
            "kp_enter": "return",
            "shift_l": "lshift",
            "shift_r": "rshift",
            "control_l": "lctrl",
            "control_r": "rctrl",
            "ctrl_l": "lctrl",
            "ctrl_r": "rctrl",
            "left_shift": "lshift",
            "right_shift": "rshift",
            "left_control": "lctrl",
            "right_control": "rctrl",
            "bracketleft": "leftbracket",
            "bracketright": "rightbracket",
            "prior": "pageup",
            "next": "pagedown",
        }
        return aliases.get(cleaned, cleaned)

    def _apply_scene_uniforms(self, scene: Scene, camera: Camera, settings: RenderSettings, width: int, height: int) -> None:
        right, true_up, forward = camera.basis()
        self._set_uniform(self.scene_program, "u_camera_position", camera.position.as_tuple())
        self._set_uniform(self.scene_program, "u_camera_right", right.as_tuple())
        self._set_uniform(self.scene_program, "u_camera_up", true_up.as_tuple())
        self._set_uniform(self.scene_program, "u_camera_forward", forward.as_tuple())
        self._set_uniform(self.scene_program, "u_focal", 1.0 / tan(radians(camera.fov_degrees) / 2.0))
        self._set_uniform(self.scene_program, "u_aspect", width / max(1, height))
        self._set_uniform(self.scene_program, "u_near", camera.near)
        self._set_uniform(self.scene_program, "u_far", camera.far)
        self._set_uniform(self.scene_program, "u_ambient", settings.ambient)
        self._set_uniform(self.scene_program, "u_gamma", settings.gamma)
        self._set_uniform(self.scene_program, "u_light_wrap", settings.light_wrap)
        self._set_uniform(self.scene_program, "u_bounce_light", settings.bounce_light)
        self._set_uniform(self.scene_program, "u_tone_mapping", bool(settings.tone_mapping))
        self._set_uniform(self.scene_program, "u_two_sided_lighting", bool(settings.two_sided_lighting))
        self._set_uniform(self.scene_program, "u_reflection_bounces", int(settings.reflection_bounces))
        self._apply_light_uniforms(scene, self.scene_program)

    def _apply_light_uniforms(self, scene: Scene, program=None) -> None:
        if program is None:
            program = self.scene_program
        lights = []
        for light in scene.lights:
            if isinstance(light, Sun):
                sample = light.sample(Vec3(0.0, 0.0, 0.0))
                lights.append((0, sample.direction, sample.color.to_floats(), sample.intensity))
            elif isinstance(light, Lamp):
                lights.append((1, light.position, light.color.to_floats(), light.intensity))
            elif hasattr(light, "sample"):
                sample = light.sample(Vec3(0.0, 0.0, 0.0))
                lights.append((0, sample.direction, sample.color.to_floats(), sample.intensity))
            if len(lights) >= 8:
                break

        self._set_uniform(program, "u_light_count", len(lights))
        kinds: list[int] = []
        vectors: list[tuple[float, float, float]] = []
        colors: list[tuple[float, float, float]] = []
        intensities: list[float] = []
        for index in range(8):
            if index < len(lights):
                kind, vector, color, intensity = lights[index]
                vector_tuple = vector.as_tuple()
                color_tuple = color
            else:
                kind, vector_tuple, color_tuple, intensity = 0, (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), 0.0
            kinds.append(int(kind))
            vectors.append(vector_tuple)
            colors.append(color_tuple)
            intensities.append(float(intensity))
            self._set_uniform(program, f"u_light_kind[{index}]", int(kind))
            self._set_uniform(program, f"u_light_vector[{index}]", vector_tuple)
            self._set_uniform(program, f"u_light_color[{index}]", color_tuple)
            self._set_uniform(program, f"u_light_intensity[{index}]", float(intensity))
        self._set_uniform(program, "u_light_kind", tuple(kinds))
        self._set_uniform(program, "u_light_vector", tuple(vectors))
        self._set_uniform(program, "u_light_color", tuple(colors))
        self._set_uniform(program, "u_light_intensity", tuple(intensities))

    def _upload_scene_textures(self, settings: RenderSettings) -> None:
        textures = tuple(self.builder.active_textures[:_MAX_TEXTURES])
        if not textures:
            self._set_uniform(self.scene_program, "u_use_textures", False)
            self._set_uniform(self.scene_program, "u_texture_count", 0)
            return

        size = max(16, int(settings.texture_size))
        key = tuple((id(texture), id(texture.pixels), texture.width, texture.height, size) for texture in textures)
        if key != self._scene_texture_key or self._scene_texture is None or self._scene_texture_size != size:
            layers = [self._texture_layer_rgba(texture, size) for texture in textures]
            blank = bytes(size * size * 4)
            while len(layers) < _MAX_TEXTURES:
                layers.append(blank)
            payload = b"".join(layers)
            if self._scene_texture is None or self._scene_texture_size != size:
                if self._scene_texture is not None:
                    self._scene_texture.release()
                self._scene_texture = self.ctx.texture_array((size, size, _MAX_TEXTURES), 4, payload, alignment=1)
                try:
                    self._scene_texture.filter = (self._moderngl.LINEAR, self._moderngl.LINEAR)
                except Exception:
                    pass
                self._scene_texture_size = size
            else:
                self._scene_texture.write(payload, alignment=1)
            self._scene_texture_key = key
        self._scene_texture.use(location=0)
        self._set_uniform(self.scene_program, "u_use_textures", True)
        self._set_uniform(self.scene_program, "u_texture_count", len(textures))

    def _texture_layer_rgba(self, texture: PixelBuffer, size: int) -> bytes:
        key = (id(texture), id(texture.pixels), texture.width, texture.height, size)
        cached = self._texture_rgba_cache.get(key)
        if cached is not None:
            return cached
        prepared = texture if texture.width == size and texture.height == size else texture.resized_nearest(size, size)
        rgba = _pixelbuffer_rgba(prepared, 1.0)[2]
        flipped = _flip_rgba_rows(rgba, size, size)
        if len(self._texture_rgba_cache) > 64:
            self._texture_rgba_cache.clear()
        self._texture_rgba_cache[key] = flipped
        return flipped

    def _update_resident_scene_buffers(self, scene: Scene, settings: RenderSettings) -> tuple[int, int]:
        textures = _live_textures_for_scene(scene, settings)
        texture_key = tuple((id(texture), id(texture.pixels), texture.width, texture.height) for texture in textures)
        texture_changed = texture_key != self._resident_texture_key
        self.builder.begin_frame(textures)

        entries: list[_ResidentObjectEntry] = []
        next_entries: dict[tuple, _ResidentObjectEntry] = {}
        dirty_keys: set[tuple] = set()
        old_offsets = {key: (entry.triangle_first, entry.line_first) for key, entry in self._resident_entries.items()}
        for index, obj in enumerate(scene.objects):
            key = _resident_object_key(index, obj)
            fingerprint = _live_object_fingerprint(obj, settings)
            previous = self._resident_entries.get(key)
            if previous is None or previous.fingerprint != fingerprint or texture_changed:
                triangle_bytes, line_bytes, triangle_vertices, line_vertices = self.builder.build_object(obj, settings)
                entry = _ResidentObjectEntry(
                    key,
                    fingerprint,
                    triangle_bytes,
                    line_bytes,
                    triangle_vertices,
                    line_vertices,
                )
                dirty_keys.add(key)
            else:
                entry = previous
            entries.append(entry)
            next_entries[key] = entry

        triangle_first = 0
        line_first = 0
        for entry in entries:
            entry.triangle_first = triangle_first
            entry.line_first = line_first
            triangle_first += entry.triangle_vertices
            line_first += entry.line_vertices
            previous_offset = old_offsets.get(entry.key)
            if previous_offset is not None and previous_offset != (entry.triangle_first, entry.line_first):
                dirty_keys.add(entry.key)

        layout_signature = tuple((entry.key, entry.triangle_vertices, entry.line_vertices) for entry in entries)
        layout_changed = layout_signature != self._resident_layout_signature

        triangle_bytes_required = triangle_first * _VERTEX_STRIDE_BYTES
        line_bytes_required = line_first * _VERTEX_STRIDE_BYTES
        triangle_buffer_changed = self._ensure_buffer_capacity(self._triangle_buffer, "_triangle_capacity", triangle_bytes_required)
        line_buffer_changed = self._ensure_buffer_capacity(self._line_buffer, "_line_capacity", line_bytes_required)
        if triangle_buffer_changed or line_buffer_changed:
            dirty_keys = {entry.key for entry in entries}

        for entry in entries:
            if entry.key not in dirty_keys:
                continue
            if entry.triangle_vertices > 0:
                self._triangle_buffer.write(entry.triangle_bytes, offset=entry.triangle_first * _VERTEX_STRIDE_BYTES)
            if entry.line_vertices > 0:
                self._line_buffer.write(entry.line_bytes, offset=entry.line_first * _VERTEX_STRIDE_BYTES)

        stale_keys = set(self._resident_entries) - set(next_entries)
        if stale_keys:
            layout_changed = True

        self._resident_entries = next_entries
        self._resident_layout_signature = layout_signature
        self._resident_texture_key = texture_key
        self._resident_triangle_vertices = triangle_first
        self._resident_line_vertices = line_first
        return triangle_first, line_first

    def _ensure_buffer_capacity(self, buffer, capacity_name: str, required: int) -> bool:
        required = max(4, int(required))
        capacity = getattr(self, capacity_name)
        if required <= capacity:
            return False
        while capacity < required:
            capacity *= 2
        buffer.orphan(capacity)
        setattr(self, capacity_name, capacity)
        return True

    def _draw_scene_vertices(self, data: bytes, vertices: int, buffer, capacity_name: str, vao, mode: int) -> None:
        if not data or vertices <= 0:
            return
        capacity = getattr(self, capacity_name)
        required = len(data)
        if required > capacity:
            while capacity < required:
                capacity *= 2
            buffer.orphan(capacity)
            setattr(self, capacity_name, capacity)
        buffer.write(data)
        vao.render(mode=mode, vertices=vertices)

    def _begin_overlays(self) -> None:
        self._active_overlay_slots.clear()

    def _end_overlays(self) -> None:
        for slot in tuple(self._overlay_textures):
            if slot in self._active_overlay_slots:
                continue
            try:
                self._overlay_textures[slot].release()
            except Exception:
                pass
            self._overlay_textures.pop(slot, None)
            self._overlay_sizes.pop(slot, None)
            self._overlay_signatures.pop(slot, None)

    def _draw_scene_bulletins(self, scene: Scene, camera: Camera, width: int, height: int) -> None:
        for index, bulletin in enumerate(scene.bulletins):
            if isinstance(bulletin, TextBulletin):
                surface_width, surface_height, rgba = self._cached_bulletin_rgba(
                    bulletin.text,
                    bulletin.color,
                    bulletin.background,
                    bulletin.padding,
                    bulletin.scale,
                )
                self._draw_overlay_rgba(f"bulletin_{index}", bulletin.position[0], bulletin.position[1], surface_width, surface_height, rgba)
            elif isinstance(bulletin, FloatingTextBulletin):
                projected = camera.project(bulletin.position, width, height)
                if projected is None:
                    continue
                surface_width, surface_height, rgba = self._cached_bulletin_rgba(
                    bulletin.text,
                    bulletin.color,
                    bulletin.background,
                    bulletin.padding,
                    bulletin.scale,
                )
                left = int(projected.x + bulletin.screen_offset[0] - surface_width * bulletin.anchor[0])
                top = int(projected.y + bulletin.screen_offset[1] - surface_height * bulletin.anchor[1])
                self._draw_overlay_rgba(f"floating_bulletin_{index}", left, top, surface_width, surface_height, rgba)

    def _draw_hud(self, width: int, height: int) -> None:
        seconds = perf_counter() - self._started
        del width, height
        for index, element in enumerate(self.hud.elements):
            if isinstance(element, HUDRect):
                self._draw_overlay_solid(
                    f"hud_{index}",
                    element.position[0],
                    element.position[1],
                    max(1, int(element.size[0])),
                    max(1, int(element.size[1])),
                    element.color,
                    element.alpha,
                )
            elif isinstance(element, HUDText):
                surface_width, surface_height, rgba = self._cached_hud_text_rgba(element)
                self._draw_overlay_rgba(f"hud_{index}", element.position[0], element.position[1], surface_width, surface_height, rgba)
            elif isinstance(element, HUDImage):
                image = element.image if element.scale <= 1 else element.image.resized_nearest(element.image.width * element.scale, element.image.height * element.scale)
                surface_width, surface_height, rgba = _pixelbuffer_rgba(image, element.alpha)
                self._draw_overlay_rgba(f"hud_{index}", element.position[0], element.position[1], surface_width, surface_height, rgba)
            elif isinstance(element, HUDAnimation):
                image = element.frame_at(seconds)
                if image is None:
                    continue
                prepared = image if element.scale <= 1 else image.resized_nearest(image.width * element.scale, image.height * element.scale)
                surface_width, surface_height, rgba = _pixelbuffer_rgba(prepared, element.alpha)
                self._draw_overlay_rgba(f"hud_{index}", element.position[0], element.position[1], surface_width, surface_height, rgba)

    def _draw_crosshair(self, width: int, height: int) -> None:
        surface_width, surface_height, rgba = self._cached_crosshair_rgba(Color(235, 244, 255))
        self._draw_overlay_rgba(
            "crosshair",
            (width - surface_width) // 2,
            (height - surface_height) // 2,
            surface_width,
            surface_height,
            rgba,
        )

    def _draw_menu(self, width: int, height: int) -> None:
        key = self._menu_cache_key(width, height)
        if key == self._menu_surface_cache_key and self._menu_surface_cache is not None:
            menu, left, top, rgba = self._menu_surface_cache
        else:
            menu, left, top = render_live_menu_surface(self.menu, width, height)
            rgba = _pixelbuffer_rgba(menu, 1.0)[2]
            self._menu_surface_cache_key = key
            self._menu_surface_cache = (menu, left, top, rgba)
        self._draw_overlay_rgba("menu", left, top, menu.width, menu.height, rgba, upload_key=key)

    def _draw_overlay_solid(
        self,
        slot: str,
        left: int,
        top: int,
        width: int,
        height: int,
        color: Color | tuple[int, int, int],
        alpha: float,
    ) -> None:
        solid = Color.from_value(color)
        opacity = _alpha_byte(alpha)
        rgba = bytes((solid.r, solid.g, solid.b, opacity))
        self._draw_overlay_rgba(
            slot,
            left,
            top,
            1,
            1,
            rgba,
            quad_width=max(1, int(width)),
            quad_height=max(1, int(height)),
            upload_key=(solid, opacity),
        )

    def _draw_overlay_rgba(
        self,
        slot: str,
        left: int,
        top: int,
        width: int,
        height: int,
        rgba: bytes,
        *,
        quad_width: int | None = None,
        quad_height: int | None = None,
        upload_key: object | None = None,
    ) -> None:
        if width <= 0 or height <= 0:
            return
        texture = self._overlay_texture(slot, width, height, rgba, upload_key=upload_key)
        texture.use(location=0)
        frame_width, frame_height = self._last_frame_size
        vertices = _overlay_quad_vertices(
            left,
            top,
            quad_width if quad_width is not None else width,
            quad_height if quad_height is not None else height,
            frame_width,
            frame_height,
        )
        self._overlay_buffer.write(array("f", vertices).tobytes())
        self._overlay_vao.render(mode=self._moderngl.TRIANGLES, vertices=6)

    def _overlay_texture(self, slot: str, width: int, height: int, rgba: bytes, *, upload_key: object | None = None):
        self._active_overlay_slots.add(slot)
        texture = self._overlay_textures.get(slot)
        signature = (width, height, upload_key if upload_key is not None else id(rgba))
        if texture is not None and self._overlay_sizes.get(slot) == (width, height) and self._overlay_signatures.get(slot) == signature:
            return texture
        payload = _flip_rgba_rows(rgba, width, height)
        if texture is None or self._overlay_sizes.get(slot) != (width, height):
            if texture is not None:
                texture.release()
            texture = self.ctx.texture((width, height), 4, payload, alignment=1)
            try:
                texture.filter = (self._moderngl.NEAREST, self._moderngl.NEAREST)
            except Exception:
                pass
            self._overlay_textures[slot] = texture
            self._overlay_sizes[slot] = (width, height)
        else:
            texture.write(payload, alignment=1)
        self._overlay_signatures[slot] = signature
        return texture

    def _cached_bulletin_rgba(
        self,
        text: str,
        color: Color,
        background: Color | None,
        padding: int,
        scale: int,
    ) -> tuple[int, int, bytes]:
        key = ("bulletin", text, color, background, padding, scale)
        cached = self._surface_rgba_cache.get(key)
        if cached is not None:
            return cached
        value = _bulletin_rgba(text, color, background, padding, scale)
        self._remember_surface_rgba(key, value)
        return value

    def _cached_hud_text_rgba(self, element: HUDText) -> tuple[int, int, bytes]:
        key = (
            "hud_text",
            element.text,
            Color.from_value(element.color),
            Color.from_value(element.background) if element.background is not None else None,
            _alpha_byte(element.alpha),
            element.padding,
            element.scale,
        )
        cached = self._surface_rgba_cache.get(key)
        if cached is not None:
            return cached
        value = _hud_text_rgba(element)
        self._remember_surface_rgba(key, value)
        return value

    def _cached_crosshair_rgba(self, color: Color) -> tuple[int, int, bytes]:
        key = ("crosshair", color)
        cached = self._surface_rgba_cache.get(key)
        if cached is not None:
            return cached
        value = _crosshair_rgba(color)
        self._remember_surface_rgba(key, value)
        return value

    def _remember_surface_rgba(self, key: tuple, value: tuple[int, int, bytes]) -> None:
        if len(self._surface_rgba_cache) > 128:
            self._surface_rgba_cache.clear()
        self._surface_rgba_cache[key] = value

    def _menu_cache_key(self, width: int, height: int):
        menu = self.menu
        options = tuple((option.action, option.label, option.detail, option.group, option.enabled) for option in menu.options)
        scroll = tuple(sorted(menu.scroll_offsets.items()))
        theme = (
            menu.theme.panel,
            menu.theme.border,
            menu.theme.text,
            menu.theme.muted_text,
            menu.theme.row,
            menu.theme.row_selected,
            menu.theme.button,
            menu.theme.button_selected,
            menu.theme.button_border,
            menu.theme.active_border,
        )
        return (width, height, menu.visible, menu.title, menu.selected_index, menu.active_group, options, scroll, theme)

    def _invalidate_surface_caches(self) -> None:
        self._menu_surface_cache_key = None
        self._menu_surface_cache = None

    @staticmethod
    def _set_uniform(program, name: str, value) -> bool:
        for candidate in (name, f"{name}[0]" if not name.endswith("]") else name):
            try:
                program[candidate].value = value
                return True
            except Exception:
                continue
        return False


class ModernGLLiveRenderer:
    """Create the fastest available live renderer with a compatible API."""

    def __new__(
        cls,
        width: int,
        height: int,
        *,
        title: str = "py_3d",
        vsync: bool = True,
        resizable: bool = True,
        backend: str = "auto",
    ):
        backend_name = backend.lower()
        if backend_name in {"auto", "gl", "opengl", "moderngl", "gpu"}:
            try:
                return _GLFWModernGLLiveRenderer(width, height, title=title, vsync=vsync, resizable=resizable)
            except Exception:
                if backend_name != "auto":
                    raise
        if backend_name in {"auto", "pixel", "cpu", "native"}:
            return _PixelLiveRenderer(width, height, title=title, vsync=vsync, resizable=resizable)
        raise ValueError(f"unknown live renderer backend: {backend}")


def _overlay_quad_vertices(left: int, top: int, width: int, height: int, frame_width: int, frame_height: int) -> tuple[float, ...]:
    x0 = (float(left) / max(1, frame_width)) * 2.0 - 1.0
    x1 = (float(left + width) / max(1, frame_width)) * 2.0 - 1.0
    y0 = 1.0 - (float(top) / max(1, frame_height)) * 2.0
    y1 = 1.0 - (float(top + height) / max(1, frame_height)) * 2.0
    return (
        x0,
        y0,
        0.0,
        1.0,
        x1,
        y0,
        1.0,
        1.0,
        x1,
        y1,
        1.0,
        0.0,
        x0,
        y0,
        0.0,
        1.0,
        x1,
        y1,
        1.0,
        0.0,
        x0,
        y1,
        0.0,
        0.0,
    )


def _flip_rgba_rows(rgba: bytes, width: int, height: int) -> bytes:
    stride = width * 4
    if len(rgba) != stride * height:
        raise ValueError("RGBA payload dimensions do not match")
    return b"".join(rgba[y * stride : (y + 1) * stride] for y in range(height - 1, -1, -1))


def _particle_water_surface_vertices(quality: str, radius: float) -> tuple[float, ...]:
    rings, segments = {"fast": (12, 44), "balanced": (17, 60), "high": (24, 84), "ultra": (32, 112)}.get(quality, (17, 60))
    points: dict[tuple[int, int], tuple[float, float, float, float, float]] = {}
    center = (0.0, 0.0, 0.0, 0.5, 0.5)
    for ring in range(1, rings + 1):
        amount = ring / rings
        radial = radius * (amount ** 0.72)
        edge = max(0.0, (amount - 0.88) / 0.12)
        for segment in range(segments):
            angle = 2.0 * 3.141592653589793 * segment / segments
            x = cos(angle) * radial
            z = sin(angle) * radial
            points[(ring, segment)] = (x, z, edge, 0.5 + x / max(0.001, radius * 2.0), 0.5 + z / max(0.001, radius * 2.0))

    vertices: list[float] = []

    def append(vertex: tuple[float, float, float, float, float]) -> None:
        vertices.extend(vertex)

    for segment in range(segments):
        next_segment = (segment + 1) % segments
        append(center)
        append(points[(1, next_segment)])
        append(points[(1, segment)])
    for ring in range(1, rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = points[(ring, segment)]
            b = points[(ring, next_segment)]
            c = points[(ring + 1, next_segment)]
            d = points[(ring + 1, segment)]
            append(a)
            append(b)
            append(c)
            append(a)
            append(c)
            append(d)
    return tuple(vertices)


def _particle_fluid_puddle_vertices(quality: str, bounds_min: Vec3, bounds_max: Vec3) -> tuple[float, ...]:
    columns, rows = {"fast": (44, 28), "balanced": (64, 40), "high": (88, 56), "ultra": (116, 72)}.get(quality, (64, 40))
    points: dict[tuple[int, int], tuple[float, float, float, float]] = {}
    for row in range(rows + 1):
        v = row / rows
        z = bounds_min.z + (bounds_max.z - bounds_min.z) * v
        for column in range(columns + 1):
            u = column / columns
            x = bounds_min.x + (bounds_max.x - bounds_min.x) * u
            points[(column, row)] = (x, z, u, v)

    vertices: list[float] = []

    def append(vertex: tuple[float, float, float, float]) -> None:
        vertices.extend(vertex)

    for row in range(rows):
        for column in range(columns):
            a = points[(column, row)]
            b = points[(column + 1, row)]
            c = points[(column + 1, row + 1)]
            d = points[(column, row + 1)]
            append(a)
            append(b)
            append(c)
            append(a)
            append(c)
            append(d)
    return tuple(vertices)


def _live_textures_for_scene(scene: Scene, settings: RenderSettings) -> tuple[PixelBuffer, ...]:
    if settings.wireframe:
        return ()
    textures: list[PixelBuffer] = []
    seen: set[int] = set()
    for obj in scene.objects:
        for texture in _live_textures_for_object(obj, settings):
            key = id(texture)
            if key in seen:
                continue
            seen.add(key)
            textures.append(texture)
            if len(textures) >= _MAX_TEXTURES:
                return tuple(textures)
    return tuple(textures)


def _live_textures_for_object(obj, settings: RenderSettings) -> tuple[PixelBuffer, ...]:
    del settings
    if isinstance(obj, (Sphere, Bowl, Capsule)):
        return _material_texture_tuple(obj.material)
    if isinstance(obj, Plane):
        return _material_texture_tuple(obj.material)
    if isinstance(obj, Triangle):
        return _material_texture_tuple(obj.material) if obj.has_texture_coordinates() else ()
    if isinstance(obj, TransformedMesh):
        return _live_textures_for_mesh(obj.mesh)
    if isinstance(obj, Mesh):
        return _live_textures_for_mesh(obj)
    return ()


def _live_textures_for_mesh(mesh: Mesh) -> tuple[PixelBuffer, ...]:
    textures: list[PixelBuffer] = []
    seen: set[int] = set()
    for triangle in mesh.triangles:
        texture = triangle.material.texture if triangle.has_texture_coordinates() else None
        if texture is None or id(texture) in seen:
            continue
        seen.add(id(texture))
        textures.append(texture)
        if len(textures) >= _MAX_TEXTURES:
            break
    return tuple(textures)


def _material_texture_tuple(material: Material) -> tuple[PixelBuffer, ...]:
    return (material.texture,) if material.texture is not None else ()


def _resident_object_key(index: int, obj) -> tuple:
    if isinstance(obj, Mesh) and len(obj.triangles) > 16:
        return ("mesh_id", id(obj), type(obj).__name__)
    if isinstance(obj, LineSet3) and len(obj.segments) > 16:
        return ("line_set_id", id(obj), type(obj).__name__)
    return ("index", index, type(obj).__name__)


def _live_object_fingerprint(obj, settings: RenderSettings) -> tuple:
    geometry_settings = (
        bool(settings.wireframe),
        bool(settings.smooth_shading),
        settings.sphere_segments,
        settings.sphere_rings,
    )
    if isinstance(obj, Point3):
        return ("point", obj.position.as_tuple(), _material_fingerprint(obj.material), geometry_settings)
    if isinstance(obj, ParticleFluidPuddle):
        return (
            "particle_fluid_puddle",
            id(obj.world),
            obj.bounds_min.as_tuple(),
            obj.bounds_max.as_tuple(),
            obj.floor_y,
            obj.quality,
            obj.kernel_radius,
            obj.density_threshold,
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, ParticleFluidVolume):
        return (
            "particle_fluid_volume",
            id(obj.world),
            obj.particle_radius,
            obj.min_y,
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, ParticleWaterSurface):
        return (
            "particle_water_surface",
            id(obj.world),
            obj.center.as_tuple(),
            obj.radius,
            obj.quality,
            obj.particle_base_y,
            obj.kernel_radius,
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, Line3):
        return ("line", obj.start.as_tuple(), obj.end.as_tuple(), _material_fingerprint(obj.material), geometry_settings)
    if isinstance(obj, LineSet3):
        if len(obj.segments) <= 16:
            line_key = tuple((start.as_tuple(), end.as_tuple()) for start, end in obj.segments)
        else:
            line_key = (
                id(obj),
                len(obj.segments),
                (obj.segments[0][0].as_tuple(), obj.segments[0][1].as_tuple()) if obj.segments else None,
                (obj.segments[-1][0].as_tuple(), obj.segments[-1][1].as_tuple()) if obj.segments else None,
            )
        return ("line_set", line_key, _material_fingerprint(obj.material), geometry_settings)
    if isinstance(obj, Sphere):
        return (
            "sphere",
            obj.center.as_tuple(),
            obj.radius,
            id(obj.perturbation),
            obj.rotation.as_tuple(),
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, Bowl):
        return (
            "bowl",
            obj.center.as_tuple(),
            obj.radius,
            obj.depth,
            obj.thickness,
            id(obj.perturbation),
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, Capsule):
        return (
            "capsule",
            obj.center.as_tuple(),
            obj.radius,
            obj.height,
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, Box):
        return ("box", obj.center.as_tuple(), obj.size.as_tuple(), _material_fingerprint(obj.material), geometry_settings)
    if isinstance(obj, Plane):
        return (
            "plane",
            obj.point.as_tuple(),
            obj.normal.as_tuple(),
            obj.size,
            obj.thickness,
            _material_fingerprint(obj.material),
            geometry_settings,
        )
    if isinstance(obj, Triangle):
        return ("triangle", _triangle_fingerprint(obj), geometry_settings)
    if isinstance(obj, TransformedMesh):
        return (
            "transformed_mesh",
            id(obj.mesh),
            len(obj.mesh.triangles),
            _triangle_fingerprint(obj.mesh.triangles[0]) if obj.mesh.triangles else None,
            _triangle_fingerprint(obj.mesh.triangles[-1]) if obj.mesh.triangles else None,
            obj.center.as_tuple(),
            obj.rotation.as_tuple(),
            geometry_settings,
        )
    if isinstance(obj, Mesh):
        if len(obj.triangles) <= 16:
            mesh_key = tuple(_triangle_fingerprint(triangle) for triangle in obj.triangles)
        else:
            mesh_key = (
                id(obj),
                len(obj.triangles),
                _triangle_fingerprint(obj.triangles[0]) if obj.triangles else None,
                _triangle_fingerprint(obj.triangles[-1]) if obj.triangles else None,
            )
        return (
            "mesh",
            mesh_key,
            geometry_settings,
        )
    return ("object", type(obj).__name__, repr(obj), geometry_settings)


def _triangle_fingerprint(triangle: Triangle) -> tuple:
    return (
        triangle.a.as_tuple(),
        triangle.b.as_tuple(),
        triangle.c.as_tuple(),
        _material_fingerprint(triangle.material),
        triangle.uv_a,
        triangle.uv_b,
        triangle.uv_c,
        triangle.normal_a.as_tuple() if triangle.normal_a is not None else None,
        triangle.normal_b.as_tuple() if triangle.normal_b is not None else None,
        triangle.normal_c.as_tuple() if triangle.normal_c is not None else None,
    )


def _material_fingerprint(material: Material) -> tuple:
    texture = material.texture
    texture_key = None if texture is None else (id(texture), id(texture.pixels), texture.width, texture.height)
    return (
        material.color,
        material.absorption,
        material.diffuse,
        material.emission,
        texture_key,
        material.roughness,
        material.fuzziness,
        material.specular,
        material.shininess,
        material.reflectivity,
        material.light_transmission,
    )


def _blend_color(base: Color, overlay: Color | tuple[int, int, int], alpha: float) -> Color:
    top = Color.from_value(overlay)
    amount = max(0.0, min(1.0, float(alpha)))
    inverse = 1.0 - amount
    return Color(
        int(base.r * inverse + top.r * amount),
        int(base.g * inverse + top.g * amount),
        int(base.b * inverse + top.b * amount),
    )


def _blend_rect(buffer: PixelBuffer, position: tuple[int, int], size: tuple[int, int], color: Color | tuple[int, int, int], alpha: float) -> None:
    x, y = int(position[0]), int(position[1])
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    for yy in range(max(0, y), min(buffer.height, y + height)):
        for xx in range(max(0, x), min(buffer.width, x + width)):
            buffer.set_pixel(xx, yy, _blend_color(buffer.get_pixel(xx, yy), color, alpha))


def _blur_buffer_numpy(frame: PixelBuffer, *, radius: int) -> bool:
    try:
        import numpy
    except Exception:
        return False
    if radius <= 0:
        return True
    try:
        source = numpy.frombuffer(frame.to_rgb_bytes(), dtype=numpy.uint8).reshape((frame.height, frame.width, 3))
    except Exception:
        return False
    pixel_count = frame.width * frame.height
    sample_step = 1
    if pixel_count > 900_000:
        sample_step = 4
    elif pixel_count > 260_000:
        sample_step = 2
    if sample_step > 1:
        source = source[::sample_step, ::sample_step]
    source_height, source_width = source.shape[:2]
    window = radius * 2 + 1
    horizontal_source = numpy.pad(source, ((0, 0), (radius, radius), (0, 0)), mode="edge").astype(numpy.uint32)
    horizontal_sum = numpy.cumsum(horizontal_source, axis=1, dtype=numpy.uint32)
    horizontal_sum = numpy.pad(horizontal_sum, ((0, 0), (1, 0), (0, 0)), mode="constant")
    horizontal = horizontal_sum[:, window : window + source_width] - horizontal_sum[:, :source_width]
    vertical_source = numpy.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    vertical_sum = numpy.cumsum(vertical_source, axis=0, dtype=numpy.uint32)
    vertical_sum = numpy.pad(vertical_sum, ((1, 0), (0, 0), (0, 0)), mode="constant")
    blurred = ((vertical_sum[window : window + source_height] - vertical_sum[:source_height]) // (window * window)).astype(numpy.uint8)
    if sample_step > 1:
        blurred = blurred.repeat(sample_step, axis=0).repeat(sample_step, axis=1)[: frame.height, : frame.width]
    frame.pixels = PixelBuffer.from_rgb_bytes(frame.width, frame.height, bytearray(blurred.tobytes())).pixels
    return True


def _blit_buffer(target: PixelBuffer, source: PixelBuffer, left: int, top: int, *, alpha: float = 1.0) -> None:
    amount = max(0.0, min(1.0, float(alpha)))
    if amount >= 1.0:
        target.blit(source, left, top)
        return
    for y in range(source.height):
        target_y = top + y
        if target_y < 0 or target_y >= target.height:
            continue
        for x in range(source.width):
            target_x = left + x
            if target_x < 0 or target_x >= target.width:
                continue
            pixel = source.get_pixel(x, y)
            if amount >= 1.0:
                target.set_pixel(target_x, target_y, pixel)
            else:
                target.set_pixel(target_x, target_y, _blend_color(target.get_pixel(target_x, target_y), pixel, amount))


def _template_from_triangles(triangles: Iterable[Triangle], *, smooth_shading: bool = True) -> _Template:
    vertices: list[tuple[float, float, float, float, float, float, float, float]] = []
    for triangle in triangles:
        face_normal = _triangle_normal_components(triangle)
        if smooth_shading and triangle.has_vertex_normals():
            normals = (
                (triangle.normal_a.x, triangle.normal_a.y, triangle.normal_a.z),
                (triangle.normal_b.x, triangle.normal_b.y, triangle.normal_b.z),
                (triangle.normal_c.x, triangle.normal_c.y, triangle.normal_c.z),
            )
        else:
            normals = (face_normal, face_normal, face_normal)
        uvs = (
            triangle.uv_a or (0.0, 0.0),
            triangle.uv_b or (0.0, 0.0),
            triangle.uv_c or (0.0, 0.0),
        )
        for point, normal, uv in zip((triangle.a, triangle.b, triangle.c), normals, uvs):
            vertices.append((point.x, point.y, point.z, normal[0], normal[1], normal[2], uv[0], uv[1]))
    return _Template(tuple(vertices))


def _triangle_normal_components(triangle: Triangle) -> tuple[float, float, float]:
    ax, ay, az = triangle.a.x, triangle.a.y, triangle.a.z
    ux = triangle.b.x - ax
    uy = triangle.b.y - ay
    uz = triangle.b.z - az
    vx = triangle.c.x - ax
    vy = triangle.c.y - ay
    vz = triangle.c.z - az
    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx
    length = sqrt(nx * nx + ny * ny + nz * nz)
    if length <= 1e-12:
        return (0.0, 0.0, 0.0)
    inverse = 1.0 / length
    return (nx * inverse, ny * inverse, nz * inverse)


def _append_template(
    payload: array,
    template: _Template,
    material: Material,
    center: Vec3,
    rotation: Vec3,
    texture_index: int,
) -> None:
    use_rotation = rotation.length_squared() > 1e-12
    material_values = _material_values(material)
    base_color = material.color.to_floats()
    center_x, center_y, center_z = center.x, center.y, center.z
    if use_rotation:
        rotation_values = (
            cos(rotation.x),
            sin(rotation.x),
            cos(rotation.y),
            sin(rotation.y),
            cos(rotation.z),
            sin(rotation.z),
        )
    else:
        rotation_values = None

    for px, py, pz, nx, ny, nz, u, v in template.vertices:
        if rotation_values is not None:
            px, py, pz = _rotate_components(px, py, pz, rotation_values)
            nx, ny, nz = _rotate_components(nx, ny, nz, rotation_values)
        color = base_color if texture_index >= 0 or material.texture is None else _material_color_floats(material, (u, v))
        payload.extend(
            (
                center_x + px,
                center_y + py,
                center_z + pz,
                nx,
                ny,
                nz,
                color[0],
                color[1],
                color[2],
                *material_values,
                u,
                v,
                float(texture_index),
            )
        )


def _append_vertex(
    payload: array,
    position: Vec3,
    normal: Vec3,
    material: Material,
    uv: tuple[float, float] | None,
    *,
    force_emissive: bool = False,
    texture_index: int = -1,
) -> None:
    _append_vertex_components(
        payload,
        position.x,
        position.y,
        position.z,
        normal.x,
        normal.y,
        normal.z,
        material,
        uv,
        force_emissive=force_emissive,
        texture_index=texture_index,
    )


def _append_vertex_components(
    payload: array,
    position_x: float,
    position_y: float,
    position_z: float,
    normal_x: float,
    normal_y: float,
    normal_z: float,
    material: Material,
    uv: tuple[float, float] | None,
    *,
    force_emissive: bool = False,
    texture_index: int = -1,
) -> None:
    if force_emissive:
        texture_index = -1
    color = material.color.to_floats() if (texture_index >= 0 and uv is not None) or material.texture is None else _material_color_floats(material, uv)
    if force_emissive:
        emission = tuple(min(1.0, color[index] + material.emission.to_floats()[index]) for index in range(3))
        material_values = (emission[0], emission[1], emission[2], 0.0, 0.0, 1.0, 0.0)
    else:
        material_values = _material_values(material)
    payload.extend(
        (
            position_x,
            position_y,
            position_z,
            normal_x,
            normal_y,
            normal_z,
            color[0],
            color[1],
            color[2],
            *material_values,
            uv[0] if uv is not None else 0.0,
            uv[1] if uv is not None else 0.0,
            float(texture_index if uv is not None else -1),
        )
    )


def _bulletin_rgba(
    text: str,
    color: Color,
    background: Color | None,
    padding: int,
    scale: int,
) -> tuple[int, int, bytes]:
    text_width, text_height = draw.text_size(text, scale=scale)
    width = max(1, text_width + padding * 2)
    height = max(1, text_height + padding * 2)
    fill = background or Color(0, 0, 0)
    buffer = PixelBuffer.new(width, height, fill)
    draw.text(buffer, (padding, padding), text, color, scale=scale)
    payload = bytearray(width * height * 4)
    offset = 0
    for pixel in buffer.pixels:
        is_background = background is not None and pixel == background
        is_empty = background is None and pixel == fill
        alpha = 218 if is_background else (0 if is_empty else 255)
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = alpha
        offset += 4
    return width, height, bytes(payload)


def _crosshair_rgba(color: Color) -> tuple[int, int, bytes]:
    size = 19
    center = size // 2
    payload = bytearray(size * size * 4)
    for y in range(size):
        for x in range(size):
            horizontal = y == center and (2 <= x <= 7 or 11 <= x <= 16)
            vertical = x == center and (2 <= y <= 7 or 11 <= y <= 16)
            dot = abs(x - center) <= 1 and abs(y - center) <= 1
            if not (horizontal or vertical or dot):
                continue
            offset = (y * size + x) * 4
            payload[offset] = color.r
            payload[offset + 1] = color.g
            payload[offset + 2] = color.b
            payload[offset + 3] = 230 if dot else 190
    return size, size, bytes(payload)


def _solid_rgba(size: tuple[int, int], color: Color | tuple[int, int, int], alpha: float) -> tuple[int, int, bytes]:
    width = max(1, int(size[0]))
    height = max(1, int(size[1]))
    fill = Color.from_value(color)
    payload = bytearray(width * height * 4)
    opacity = _alpha_byte(alpha)
    for offset in range(0, len(payload), 4):
        payload[offset] = fill.r
        payload[offset + 1] = fill.g
        payload[offset + 2] = fill.b
        payload[offset + 3] = opacity
    return width, height, bytes(payload)


def _hud_text_rgba(element: HUDText) -> tuple[int, int, bytes]:
    text_width, text_height = draw.text_size(element.text, scale=element.scale)
    width = max(1, text_width + element.padding * 2)
    height = max(1, text_height + element.padding * 2)
    transparent_key = Color(0, 0, 0)
    fill = Color.from_value(element.background) if element.background is not None else transparent_key
    buffer = PixelBuffer.new(width, height, fill)
    draw.text(buffer, (element.padding, element.padding), element.text, element.color, scale=element.scale)
    background_alpha = _alpha_byte(element.alpha) if element.background is not None else 0
    foreground_alpha = _alpha_byte(element.alpha)
    payload = bytearray(width * height * 4)
    offset = 0
    for pixel in buffer.pixels:
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = background_alpha if pixel == fill else foreground_alpha
        offset += 4
    return width, height, bytes(payload)


def _pixelbuffer_rgba(buffer: PixelBuffer, alpha: float) -> tuple[int, int, bytes]:
    payload = bytearray(buffer.width * buffer.height * 4)
    opacity = _alpha_byte(alpha)
    offset = 0
    for pixel in buffer.pixels:
        payload[offset] = pixel.r
        payload[offset + 1] = pixel.g
        payload[offset + 2] = pixel.b
        payload[offset + 3] = opacity
        offset += 4
    return buffer.width, buffer.height, bytes(payload)


def _alpha_byte(alpha: float) -> int:
    return max(0, min(255, int(float(alpha) * 255)))


def _material_color(material: Material, uv: tuple[float, float] | None) -> Color:
    if material.texture is not None and uv is not None:
        return material.color_at(uv)
    return material.color


def _material_color_floats(material: Material, uv: tuple[float, float] | None) -> tuple[float, float, float]:
    color = _material_color(material, uv)
    return (color.r / 255.0, color.g / 255.0, color.b / 255.0)


def _material_values(material: Material) -> tuple[float, float, float, float, float, float, float]:
    emission = material.emission.to_floats()
    specular = material.specular * (1.0 - material.roughness * 0.85) + material.reflectivity * 0.5
    return (
        emission[0],
        emission[1],
        emission[2],
        material.diffuse,
        specular,
        material.shininess,
        material.reflectivity,
    )


def _lerp_degrees(start: float, end: float, amount: float) -> float:
    delta = (end - start + 180.0) % 360.0 - 180.0
    return start + delta * amount


def _rotate_components(
    x: float,
    y: float,
    z: float,
    rotation_values: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float]:
    cx, sx, cy, sy, cz, sz = rotation_values
    x1 = x
    y1 = y * cx - z * sx
    z1 = y * sx + z * cx
    x2 = x1 * cy + z1 * sy
    y2 = y1
    z2 = -x1 * sy + z1 * cy
    return (x2 * cz - y2 * sz, x2 * sz + y2 * cz, z2)


def _rotate_euler(value: Vec3, rotation: Vec3) -> Vec3:
    cx, sx = cos(rotation.x), sin(rotation.x)
    cy, sy = cos(rotation.y), sin(rotation.y)
    cz, sz = cos(rotation.z), sin(rotation.z)
    x_rotated = Vec3(value.x, value.y * cx - value.z * sx, value.y * sx + value.z * cx)
    y_rotated = Vec3(x_rotated.x * cy + x_rotated.z * sy, x_rotated.y, -x_rotated.x * sy + x_rotated.z * cy)
    return Vec3(y_rotated.x * cz - y_rotated.y * sz, y_rotated.x * sz + y_rotated.y * cz, y_rotated.z)
