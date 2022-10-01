float GLOBAL_LIGHTER = 1.0;
vec3 CAMERA_POS = vec3(-2., 1.5, -3.);
//LIGHT INFO
const vec3 LIGHTS_POS[3] = vec3[3](vec3(3., 1., -0.0), vec3(-2.0, 2., 2.), vec3(-2.0, 1.0, -3.0));
const vec3 LIGHTS_COLOR[3] = vec3[3](vec3(0.89, 0.85, 0.36), vec3(.0, 1.,1.), vec3(1., 1., 1.));
const float LIGHTS_RADIUS[3] = float[3](.5, 0.5, 0.5);

//Pyramid Block
const vec3 P_CNTR = vec3(.0, -0.3 + 10e-5, 0.0);
const float P_H = 2.;
const float P_BASE = 1.;

//Arrays of pyramid points and triangle points and pos offsets of camera
const vec3 P_P[5] = vec3[5](P_CNTR + vec3(0, P_H, 0),                   //0
                            P_CNTR + vec3(P_BASE, 0, -P_BASE),          //1
                            P_CNTR + vec3(-P_BASE, 0, -P_BASE),         //2
                            P_CNTR + vec3(P_BASE, 0, P_BASE),           //3
                            P_CNTR + vec3(-P_BASE, 0, P_BASE));         //4
                            
const vec3 P_T[6] = vec3[6](vec3(3, 1, 0), vec3(2, 3, 4), vec3(0, 4, 2), 
                            vec3(2, 1, 0), vec3(2, 1, 3), vec3(0, 3, 4));
                        
//End of Pyramid Block


const vec3 P_CNTR2 = vec3(.0, -0.3 + 10e-5, 0.0);
const float P_H2 = 1.9;
const float P_BASE2 = .9;

//Arrays of pyramid points and triangle points and pos offsets of camera
const vec3 P_P2[5] = vec3[5](P_CNTR2 + vec3(0, P_H2, 0),                   //0
                            P_CNTR2 + vec3(P_BASE2, 0, -P_BASE2),          //1
                            P_CNTR2 + vec3(-P_BASE2, 0, -P_BASE2),         //2
                            P_CNTR2 + vec3(P_BASE2, 0, P_BASE2),           //3
                            P_CNTR2 + vec3(-P_BASE2, 0, P_BASE2));         //4
                            
const vec3 P_T2[6] = vec3[6](vec3(3, 1, 0), vec3(2, 3, 4), vec3(0, 4, 2), 
                            vec3(2, 1, 0), vec3(2, 1, 3), vec3(0, 3, 4));  


//CYLINDER
const vec3 CYL = vec3(1.5, -1.0 + 10e-5, 0.7); // vec3(Radius, lowest y of cyl, height of cyl)

//Useful consts
const float INF = 10e10;
const float EPS = 10e-7;

const int EMISSION = 0;
const int DIFFUSION = 1;
const int REFLECTION = 2;
const int REFRACTION = 3;
const int FIRE = 4;
//End of useful consts


float rand(float frame) {
    return fract(sin( dot(vec3(frame), vec3(12.9898,78.233,45.5432) )) * 43758.5453);
}


float tracePlane(vec3 pos, vec3 dir, float y, float trace_dist, out vec3 normal) {
    float t = (y - pos.y) / dir.y;
    if (t <= 0.0) {
        return INF;
    }
    vec3 WorldPos = t * dir + pos;
    if (dot(WorldPos.xz, WorldPos.xz) < trace_dist) {
        normal = vec3(0, 1, 0);
        return t;
    }
    return INF;
}  
   
   
float traceSphere(vec3 pos, vec3 dir, float r, out vec3 normal) {
    float b = dot(pos, dir);
    float D = b * b - dot(pos, pos) + r * r;
    if (D < 0.0) {
        return INF;
    }
    float t = (-b-sqrt(D));
    if (t > 0.0) {
        normal = normalize(pos + t * dir);
        return t;
    }
    t = -b + sqrt(D);
    if (t < 0.0) {
        return INF;
    }
    normal = normalize(pos + t * dir);
    return t;
}


float traceCylinder(vec3 pos, vec3 dir, vec3 cyl, out vec3 normal) {
    float t = INF;
    float temp_t = (cyl.y - pos.y) / dir.y;
    if (temp_t <= 0.0) {
        return INF;
    } else if (temp_t < t) {
        vec3 P = temp_t * dir + pos;
        if (dot(P.xz, P.xz) <= cyl.x * cyl.x) {
            normal = vec3(0, -1, 0);
            t = temp_t;
        }
    }
    temp_t = (cyl.y + cyl.z - pos.y) / dir.y;
    if (temp_t > 0.0 && temp_t < t) {
        vec3 P = temp_t * dir + pos;
        if (dot(P.xz, P.xz) <= cyl.x * cyl.x) {
            normal = vec3(0, 1, 0);
            t = temp_t;
        }
    }
    float a = dot(dir.xz, dir.xz);
    float b = dot(dir.xz, pos.xz);
    float c = dot(pos.xz, pos.xz) - cyl.x * cyl.x;
    float D = b * b - a * c;
    if (D >= 0.0) {
        float sq_D = sqrt(D);
        temp_t = (-b - sq_D) / a;
        if (temp_t > 0.0 && temp_t < t) {
            vec3 P = dir * temp_t + pos;
            if (P.y >= cyl.y && P.y <= cyl.y + cyl.z) {
                normal = vec3(P.x, 0, P.z);
                t = temp_t;
            }
        }
        temp_t = (-b + sq_D) / a;
        if (temp_t > 0.0 && temp_t < t) {
            vec3 P = dir * temp_t + pos;
            if (P.y >= cyl.y && P.y <= cyl.y + cyl.z) {
                normal = vec3(P.x, 0, P.z);
                t = temp_t;
            }
        }
    }
    normal = normalize(normal);
    return t;
}


float traceTriangle(vec3 pos, vec3 dir, vec3 v0, vec3 v1, vec3 v2, out vec3 normal) {
    vec3 e1 = v1 - v0;
    vec3 e2 = v2 - v0;
    vec3 pvec = cross(dir, e2);
    float det = dot(e1, pvec);

    if (det < 1e-8 && det > -1e-8) {
        return INF;
    }

    float inv_det = 1. / det;
    vec3 tvec = pos - v0;
    float u = dot(tvec, pvec) * inv_det;
    if (u < 0. || u > 1.) {
        return INF;
    }

    vec3 qvec = cross(tvec, e1);
    float v = dot(dir, qvec) * inv_det;
    if (v < 0. || u + v > 1.) {
        return INF;
    }
    normal = normalize(cross(e1, e2));
    float D = dot(normal, v0);
    float t = (D - dot(normal, pos)) / dot(normal, dir);
    if (t < 0.0) {
        return INF;
    }
    return t;
}


float tracePyramid(vec3 pos, vec3 dir, out vec3 normal, out bool base, out float hits) {
    
    //out float hits: for light computing. 
    //out bool base: to understand if we move in air or in cylinder
    vec3 randVals =(vec3(rand(float(iFrame + 5)), rand(float(iFrame + 125)), rand(float(iFrame + 515))) - 0.5) * 0.05;
    base = false;
    float t_max = INF;
    vec3 t_normal;
    for (int i = 0; i < 6; ++i) {    //Actually tracing 6 triangles of pyramid
        float t = traceTriangle(pos, dir, P_P[int(P_T[i].x)] + randVals, P_P[int(P_T[i].y)] + randVals, P_P[int(P_T[i].z)] + randVals, t_normal);
        if (t < INF) {
            hits += 1.;
        }
        if (t < t_max) {
            t_max = t;
            normal = t_normal;
            if (i == 1 || i == 4) {
                base = true;
            } else {
                base = false;
            }
        }
    }
    return t_max;
}

float tracePyramid2(vec3 pos, vec3 dir, out vec3 normal, out bool base, out float hits) {
    
    //out float hits: for light computing. 
    //out bool base: to understand if we move in air or in cylinder
    vec3 randVals = (vec3(rand(float(iFrame + 5)), rand(float(iFrame + 125)), rand(float(iFrame + 515))) - 0.5) * 0.5;
    base = false;
    float t_max = INF;
    vec3 t_normal;
    for (int i = 0; i < 6; ++i) {    //Actually tracing 6 triangles of pyramid
        float t = traceTriangle(pos, dir, P_P2[int(P_T2[i].x)] + randVals, P_P2[int(P_T2[i].y)] + randVals, P_P2[int(P_T2[i].z)] + randVals, t_normal);
        if (t < INF) {
            hits += 1.;
        }
        if (t < t_max) {
            t_max = t;
            normal = t_normal;
            if (i == 1 || i == 4) {
                base = true;
            } else {
                base = false;
            }
        }
    }
    return t_max;
}


vec4 fire_color(vec4 colA, vec4 colB, vec2 fragCoord) {

    vec2 uv0 = ((fragCoord.xy - iResolution.xy * .5) / iResolution.y) * 2.0;
    float angle = 0.79;
    mat2 rot = mat2(
        sin(angle), -cos(angle),
        cos(angle), sin(angle)
    );
    vec2 uv1 = uv0 * rot;
    vec2 uv2 = rot * uv0;
    vec3 enlarge = 2. - fract(vec3(0., 0.333, 0.667) + 0.5);

    float r = dot(uv0,uv0);
    float p = (pow(r, 3.) + 0.3);
    uv0 *= p;
    uv1 *= p;
    uv2 *= p;
    float fire = dot(vec3(
        texture(iChannel3, uv0 * enlarge.x).x,
        texture(iChannel3, uv1 * enlarge.y).y,
        texture(iChannel3, uv2 * enlarge.z).z
    ), smoothstep(vec3(0.5), vec3(0.0), abs(fract(enlarge)-0.5)));
    return (mix(colA, colB, fire) - r*r * 1.75) *1.75;
    
}


float Occlusion_debuff(vec3 pos, vec3 target) {

    vec3 dir = target - pos;
    float dist = length(dir);
    dir /= dist;
    float c = 0.0;
    
    vec3 trNorm;
    bool base;
    float hits = 0.;
    vec3 cylNorm;
    float cylT = traceCylinder(pos, dir, CYL, cylNorm);
    if (cylT < dist) {
        return 0.;
    }
    
    float trT = tracePyramid(pos, dir, trNorm, base, hits);
    if (trT < dist) {
        return 1. * (6. - hits) / 6.;
    }
    
    return 1.;
}


vec3 ComputeLight(vec3 pos, vec3 normal, vec3 color, vec2 fragCoord) {
    vec3 total_color = vec3(0);
    for (int j = 0; j < 2; ++j) {
        vec3 LIGHT = LIGHTS_POS[j] - pos;
        float sqdist = dot(LIGHT,LIGHT);
        float att = 10.0 / sqdist;
        float debuff = Occlusion_debuff(pos, LIGHTS_POS[j] + LIGHTS_RADIUS[j] *2.* rand(float(iFrame)));
        total_color += color * max(0.0, dot(normal, normalize(LIGHT))) * att * LIGHTS_COLOR[j] * debuff;
    }
    vec3 randVals = vec3(rand(float(iFrame + 5)), rand(float(iFrame + 125)), rand(float(iFrame + 515)));
    vec3 LIGHT = vec3(0.0, 0.3, 0.0) + randVals * cos(iTime) * 0.3;
    float sqdist = dot(LIGHT - pos,LIGHT - pos);
    float att = 7.0 / sqdist;
    float debuff = Occlusion_debuff(pos, LIGHT);
    total_color += color *  max(0.0, dot(normal, normalize(LIGHT - pos))) * att * fire_color(vec4(1.0, 0.0, 0.0, 1.0), vec4(2.0, 1.5, 0.8, 1.0), fragCoord).rgb * 0.1 * debuff;
    return total_color + texture(iChannel1, normal).rgb * color * 0.1;
}


vec3 refraction(vec3 v, vec3 normal, float n1, float n2, out bool refl) {
    if (dot(v, normal) < 0.0) {
        normal = -normal;
    }
    float cosA = dot(v, normal) / length(v) / length(normal);
    float sinA = sqrt(1.0 - cosA * cosA);
    vec3 tang = normalize(v - cosA * normal);
    float sinB = sinA / n2 * n1;
    if (sinB > 1.) {
        refl = true;
        return reflect(v, normal);
    }
    float cosB = sqrt(1.0 - sinB * sinB);
    return sinB * tang + cosB * normal;
}


vec3 reflection(vec3 v, vec3 normal) {
    return v - 2.0 * dot(v, normal) * normal;
}


float pow2(float a) {
    return a * a;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec3 randVals = vec3(rand(float(iFrame + 5)), rand(float(iFrame + 125)), rand(float(iFrame + 515)));
    vec2 uv = (fragCoord - iResolution.xy * 0.5 + (randVals.xy - 0.5) * 2.0) / iResolution.y;
    
    vec3 front = normalize(-CAMERA_POS);
    vec3 up = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(front, up));
    up = normalize(cross(right, front));
    
    vec3 ViewVector = normalize(front + right * uv.x + up * uv.y);
    vec3 WorldPos;
    
    vec3 CurPos = CAMERA_POS;
    vec3 CurDir = ViewVector;
    
    float air_n = 1.0, glass_n = 1.5;
    float n1 = 1.0, n = 1.5;
    
    float glass_r = pow2(air_n - glass_n) / pow2(air_n + glass_n);
    
    vec3 ColorMult = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 40; ++i) {

        float t = INF;
        
        int material;
        vec3 color;
        vec4 colA, colB;
        
        vec3 normal, temp_normal;
        
        float curT = t;
        float PlaneT = tracePlane(CurPos, CurDir, -1., 100.0, temp_normal);
        if (t > PlaneT) {
        
            t = PlaneT;
            material = DIFFUSION;
            vec3 WorldPos = t * CurDir + CurPos;
            
            ColorMult *= vec3(1.0, 1.0, 1.0);
            color = texture(iChannel0, WorldPos.xz * 0.5).rgb;
            
            normal = temp_normal;
        }
        
        for (int j = 0; j < 2; ++j) {

            float lightT = traceSphere(CurPos - LIGHTS_POS[j] - randVals * 10e-2, CurDir, LIGHTS_RADIUS[j], temp_normal);
            if (lightT < t) {
                t = lightT;
                material = EMISSION;
                color = LIGHTS_COLOR[j];
                normal = temp_normal;
            }
        }
        
        bool base;
        float hits;
        
        float pyramidT = tracePyramid(CurPos, CurDir, temp_normal, base, hits);
        if (t > pyramidT) {
        
            t = pyramidT;
            if (randVals.x < 3. * glass_r) {
                material = REFLECTION;
            } else {
                material = REFRACTION;
            }
            
            //ColorMult *= vec3(0.67, 0.62, 0.86);
            //ColorMult *= vec3(0.9, 0.5, 0.5);
            ColorMult *= vec3(0.8, 0.9, 0.8);
            
            normal = temp_normal;
            if (base) {
                n = n1;
            }
        }
        
        vec3 sin_off = (randVals - 0.5);
        
        float sphT = traceSphere(CurPos - vec3(0.,0.4,0.) + sin_off / 3., CurDir, .3, temp_normal);
        if (sphT < t) {
        
            t = sphT;
            material = FIRE;
            colA = vec4(1.0, 0.0, 0.0, 1.0);
            colB = vec4(3.0, 2.25, 1.6, 1.0);
            color = vec3(1., 0.6, 1.);
            normal = temp_normal;
        }

        if (iFrame % 3 == 0 || iFrame % 4 == 0) {
            float pyramid2T = tracePyramid(CurPos -  - vec3(0.,0.4,0.) + sin_off / 3., CurDir, temp_normal, base, hits);
            if (t > pyramid2T) {

                t = pyramid2T;
                material = FIRE;

                colA = vec4(0.0, 0.0, 0.0, 1.0);
                colB = vec4(0.0, 0.5, 0.8, 1.0);
            
                normal = temp_normal;
            }
       }
        
        float cylT = traceCylinder(CurPos, CurDir, CYL, temp_normal);
        if (cylT < t) {
        
            t = cylT;
            material = DIFFUSION;
            
            vec3 WorldPos = t * CurDir + CurPos;
            ColorMult *= vec3(1.0, 1.0, 1.0);
            
            vec3 y_norm = vec3(0,1,0);
            if (temp_normal == y_norm || temp_normal == -y_norm) {
                color = texture(iChannel2, WorldPos.xz).rgb;
            } else {
                color = texture(iChannel2, WorldPos.yz).rgb;
            }
            normal = temp_normal;
        }
        
        if (t != INF) {

            WorldPos = CurPos + CurDir * t;

            if (material == EMISSION) {
            
               fragColor.rgb = color * ColorMult * GLOBAL_LIGHTER;
               break;
               
            } else if (material == DIFFUSION) {
            
                fragColor.rgb = ComputeLight(WorldPos, normal, color, fragCoord) * ColorMult * GLOBAL_LIGHTER;              
                break;
                
            } else if (material == REFLECTION) {
            
                CurDir = reflect(CurDir, normal);
                CurPos = WorldPos + CurDir * 10e-5;
                
            } else if (material == REFRACTION) {
            
                bool reflected = false;
                vec3 temp_cur = refraction(CurDir, normal, n1, n, reflected);
                CurDir = temp_cur;
                CurPos = WorldPos + CurDir * 10e-5;
                if (!reflected) {
                    n1 = n;
                }
                
            } else if (material == FIRE) {
            
                fragColor.rgb = fire_color(colA, colB, fragCoord).rgb * ColorMult;
                break;
                
            }
        } else {       
            fragColor.rgb = texture(iChannel1, CurDir).rgb * ColorMult * GLOBAL_LIGHTER;
        }
    }
}