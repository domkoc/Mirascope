//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Kocka Dominik
// Neptun : FIBRPN
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

struct Paraboloid;
struct Ellipsoid;
struct Cylinder;
struct Hyperboloid;

enum MaterialType { ROUGH, REFLECTIVE };

struct Material {
    vec3 ka, kd, ks;
    float  shininess;
    vec3 F0;
    MaterialType type;
    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = (_kd * M_PI);
        kd = (_kd);
        ks = (_ks);
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

const vec3 one(1, 1, 1);

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        F0 = (((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa));
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material* material;
    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;
    Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
    Material* material;
public:
    virtual Hit intersect(const Ray& ray) = 0;
};

const float epsilon = 0.0001f;

struct Plane : public Intersectable {
    vec3 P0, normal;
    bool counts = true;


    Plane(const vec3& _dot, const vec3& _normal, Material* _material, bool _counts) {
        P0 = _dot;
        normal = normalize(_normal);
        material = _material;
        counts = _counts;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;
        if (counts == false)
        {
            return hit;
        }
        float a = P0.x, b = P0.y, c = P0.z, A = normal.x, B = normal.y, C = normal.z;
        float sx = ray.start.x, sy = ray.start.y, sz = ray.start.z, dx = ray.dir.x, dy = ray.dir.y, dz = ray.dir.z;

        float denom = A * dx + B * dy + C * dz;
        float num = A * (a - sx) + B * (b - sy) + C * (c - sz);

        float t = num / denom;

        vec3 cut = ray.start + ray.dir * t;
        hit.t = t;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = normal;
        hit.material = material;
        return hit;
    }

    int wichSide(vec3 point) {
        float a = P0.x, b = P0.y, c = P0.z, A = normal.x, B = normal.y, C = normal.z;
        float x = point.x, y = point.y, z = point.z;

        float result = dot(normal, (point - P0));

        if (result<epsilon && result >(-1 * epsilon))
        {
            return 0;
        }
        if (result < 0)
        {
            return -1;
        }
        else if (result > 0)
        {
            return 1;
        }
    }
};

struct Paraboloid : public Intersectable {
    vec3 vertex;
    float a, b;
    Plane* p;

    Paraboloid(const vec3& _vertex, float _a, float _b, Plane* _p, Material* _material) {
        vertex = _vertex; a = _a; b = _b; p = _p; material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        float A = (ray.dir.x * ray.dir.x) / (a * a) + (ray.dir.z * ray.dir.z) / (b * b);
        float B = (2.0 * ray.dir.x * (ray.start.x - vertex.x)) / (a * a) + (2.0 * ray.dir.z * (ray.start.z - vertex.z)) / (b * b) + ray.dir.y;
        float C = ((ray.start.x - vertex.x) * (ray.start.x - vertex.x)) / (a * a) + ((ray.start.z - vertex.z) * (ray.start.z - vertex.z)) / (b * b) + (ray.start.y - vertex.y);

        float discr = B * B - 4.0f * A * C;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-B + sqrt_discr) / 2.0f / A;
        float t2 = (-B - sqrt_discr) / 2.0f / A;
        if (t1 <= 0) return hit;

        vec3 p1 = ray.start + ray.dir * t1;
        vec3 p2 = ray.start + ray.dir * t2;

        if ((p->wichSide(p2) > 0) && (p->wichSide(p1) < 0))
        {
            Hit tmp;
            tmp = p->intersect(ray);
            if (tmp.t < t1)
            {
                hit.t = tmp.t;
                hit.position = ray.start + ray.dir * hit.t;
                hit.normal = tmp.normal;
                hit.material = tmp.material;
                return hit;
            }
        }

        if (p->wichSide(p1) > 0)
        {
            t1 = -1;
        }
        if (p->wichSide(p2) > 0)
        {
            t2 = -1;
        }

        hit.t = (t2 > 0) ? t2 : t1;
        if (hit.t <= 0) return hit;
        hit.position = ray.start + ray.dir * hit.t;

        float nx = 2.0 * hit.position.x / a * a;
        float ny = 2.0 * hit.position.z / b * b;
        float nz = 1;
        vec3 N(nx, ny, nz);
        hit.normal = normalize(N);
        hit.material = material;

        return hit;
    }
};

struct Ellipsoid : public Intersectable {
    vec3 center;
    float a, b, c;
    Plane* plane;

    Ellipsoid(const vec3& _center, float _a, float _b, float _c, Plane* _plane, Material* _material) {
        center = _center; a = _a; b = _b; c = _c; plane = _plane; material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        float A = (ray.dir.x * ray.dir.x) / (a * a) + (ray.dir.y * ray.dir.y) / (b * b) + (ray.dir.z * ray.dir.z) / (c * c);
        float B = (2.0 * (ray.start.x - center.x) * ray.dir.x) / (a * a) + (2.0 * (ray.start.y - center.y) * ray.dir.y) / (b * b) + (2.0 * (ray.start.z - center.z) * ray.dir.z) / (c * c);
        float C = ((ray.start.x - center.x) * (ray.start.x - center.x)) / (a * a) + ((ray.start.y - center.y) * (ray.start.y - center.y)) / (b * b) + ((ray.start.z - center.z) * (ray.start.z - center.z)) / (c * c) - 1.0;


        float discr = B * B - 4.0f * A * C;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-B + sqrt_discr) / 2.0f / A;
        float t2 = (-B - sqrt_discr) / 2.0f / A;
        if (t1 <= 0) return hit;

        vec3 p1 = ray.start + ray.dir * t1;
        vec3 p2 = ray.start + ray.dir * t2;
        if (plane->wichSide(p1) > 0)
        {
            t1 = -1;
        }
        if (plane->wichSide(p2) > 0)
        {
            t2 = -1;
        }
        if (t1 <= 0) return hit;

        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;

        if (plane->wichSide(p2) > 0)
        {
            Hit tmp;
            tmp = plane->intersect(ray);
            if (tmp.t < hit.t)
            {
                hit.t = tmp.t;
                hit.position = ray.start + ray.dir * hit.t;
                hit.normal = tmp.normal;
                hit.material = tmp.material;
                return hit;
            }
        }

        float nx = 2.0 * hit.position.x / (a * a);
        float ny = 2.0 * hit.position.y / (b * b);
        float nz = 2.0 * hit.position.z / (c * c);
        vec3 N(nx, ny, nz);
        vec3 normalN = normalize(N) * -1;
        if (t2 <= 0)
        {
            vec3 normalN = normalize(N);
        }
        hit.normal = normalN;
        hit.material = material;
        return hit;
    }
};

struct Cylinder : public Intersectable {
    vec3 center;
    float a, b;
    Plane* panel1;
    Plane* panel2;

    Cylinder(const vec3& _center, float _a, float _b, Plane* _p1, Plane* _p2, Material* _material) {
        center = _center;
        a = _a;
        b = _b;
        panel1 = _p1;
        panel2 = _p2;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        float A = (ray.dir.x * ray.dir.x) / (a * a) + (ray.dir.z * ray.dir.z) / (b * b);
        float B = (2.0 * ray.dir.x * (ray.start.x - center.x)) / (a * a) + (2.0 * ray.dir.z * (ray.start.z - center.z)) / (b * b);
        float C = ((ray.start.x - center.x) * (ray.start.x - center.x)) / (a * a) + ((ray.start.z - center.z) * (ray.start.z - center.z)) / (b * b) - 1;


        float discr = B * B - 4.0f * A * C;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-B + sqrt_discr) / 2.0f / A;
        float t2 = (-B - sqrt_discr) / 2.0f / A;
        if (t1 <= 0) return hit;

        vec3 p1 = ray.start + ray.dir * t1;
        vec3 p2 = ray.start + ray.dir * t2;
        if ((panel1->wichSide(p1)) < 0 && (panel1->wichSide(p2) > 0))
        {
            Hit tmp;
            tmp = panel1->intersect(ray);
            if (tmp.t < t1)
            {
                hit.t = tmp.t;
                hit.position = ray.start + ray.dir * hit.t;
                hit.normal = tmp.normal;
                hit.material = tmp.material;
                return hit;
            }
        }
        if ((panel2->wichSide(p2) > 0) && (panel2->wichSide(p1) < 0))
        {
            Hit tmp;
            tmp = panel2->intersect(ray);
            if (tmp.t < t1)
            {
                hit.t = tmp.t;
                hit.position = ray.start + ray.dir * hit.t;
                hit.normal = tmp.normal;
                hit.material = tmp.material;
                return hit;
            }
        }
        if ((panel1->wichSide(p2)) > 0)
        {
            t1 = -1;
        }
        if ((panel2->wichSide(p2)) > 0)
        {
            t1 = -1;
        }
        if (t1 <= 0) return hit;

        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;

        float nx = 2.0 * hit.position.x / (a * a);
        float ny = 2.0 * hit.position.z / (b * b);
        float nz = 0;
        vec3 N(nx, ny, nz);
        hit.normal = normalize(N);
        hit.material = material;
        return hit;
    }
};

struct Hyperboloid : public Intersectable {
    vec3 center;
    float a, b, c;
    Plane* plane1;
    Plane* plane2;

    Hyperboloid(const vec3& _center, float _a, float _b, float _c, Plane* pl1, Plane* pl2, Material* _material) {
        center = _center;
        a = _a;
        b = _b;
        c = _c;
        plane1 = pl1;
        plane2 = pl2;
        material = _material;
    }

    Hit intersect(const Ray& ray) {
        Hit hit;

        float A = (ray.dir.x * ray.dir.x) / (a * a) + (ray.dir.z * ray.dir.z) / (b * b) - (ray.dir.y * ray.dir.y) / (c * c);
        float B = (2.0 * ray.dir.x * (ray.start.x - center.x)) / (a * a) + (2.0 * ray.dir.z * (ray.start.z - center.z)) / (b * b) - (2.0 * ray.dir.y * (ray.start.y - center.y)) / (c * c);
        float C = ((ray.start.x - center.x) * (ray.start.x - center.x)) / (a * a) + ((ray.start.z - center.z) * (ray.start.z - center.z)) / (b * b) - ((ray.start.y - center.y) * (ray.start.y - center.y)) / (c * c) - 1;


        float discr = B * B - 4.0f * A * C;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-B + sqrt_discr) / 2.0f / A;
        float t2 = (-B - sqrt_discr) / 2.0f / A;
        if (t1 <= 0) return hit;

        vec3 p1 = ray.start + ray.dir * t1;
        vec3 p2 = ray.start + ray.dir * t2;
        if ((plane1->wichSide(p1)) < 0 && (plane1->wichSide(p2) > 0))
        {
            Hit tmp;
            tmp = plane1->intersect(ray);
            if (tmp.t < t1)
            {
                if (tmp.t > 0)
                {
                    hit.t = tmp.t;
                    hit.position = ray.start + ray.dir * hit.t;
                    hit.normal = tmp.normal;
                    hit.material = material;
                    return hit;
                }
                else
                {
                    hit.t = t1;
                    hit.position = ray.start + ray.dir * hit.t;
                    float nx = 2.0 * hit.position.x / (a * a);
                    float ny = 2.0 * hit.position.z / (b * b);
                    float nz = -2.0 * hit.position.y / (c * c);
                    vec3 N(nx, ny, nz);
                    hit.normal = normalize(N) * -1;
                    hit.material = material;
                    return hit;
                }
            }
        }
        if ((plane2->wichSide(p2) > 0) && (plane2->wichSide(p1) < 0))
        {
            Hit tmp;
            tmp = plane2->intersect(ray);
            if (tmp.t < t1)
            {
                if (tmp.t > 0)
                {
                    hit.t = tmp.t;
                    hit.position = ray.start + ray.dir * hit.t;
                    hit.normal = tmp.normal;
                    hit.material = material;
                    return hit;
                }
                else
                {
                    hit.t = t1;
                    hit.position = ray.start + ray.dir * hit.t;
                    float nx = 2.0 * hit.position.x / (a * a);
                    float ny = 2.0 * hit.position.z / (b * b);
                    float nz = -2.0 * hit.position.y / (c * c);
                    vec3 N(nx, ny, nz);
                    hit.normal = normalize(N) * -1;
                    hit.material = material;
                    return hit;
                }
            }
        }
        if ((plane1->wichSide(p2)) > 0)
        {
            t1 = -1;
        }
        if ((plane2->wichSide(p2)) > 0)
        {
            t1 = -1;
        }
        if (t1 <= 0) return hit;

        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;

        float nx = 2.0 * hit.position.x / (a * a);
        float ny = 2.0 * hit.position.z / (b * b);
        float nz = -2.0 * hit.position.y / (c * c);
        vec3 N(nx, ny, nz);
        hit.normal = normalize(N);
        if (t2 <= 0)
        {
            hit.normal = normalize(N) * -1;
        }
        hit.material = material;
        return hit;
    }
};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float fov) {
        eye = _eye;
        lookat = _lookat;
        vec3 w = eye - lookat;
        float focus = length(w);
        right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
        up = normalize(cross(w, right)) * focus * tanf(fov / 2);
    }
    Ray getRay(int X, int Y) {
        vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
        return Ray(eye, dir);
    }
    //TODO: mozgás
    void Animate(float dt) {
        vec3 d = eye - lookat;
        eye = vec3(d.x * cos(dt) + d.z * sin(dt), d.y, -d.x * sin(dt) + d.z * cos(dt)) + lookat;
        set(eye, lookat, up, fov);
    }
};

struct Light {
    vec3 direction;
    vec3 Le;
    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};

float rnd() { return (float)rand() / RAND_MAX; }
float rnd2() { return (2.0 * (float)rand() / RAND_MAX) - 1; }

std::vector<vec3> controlPoints;

void countControlPoints(vec3 center, float r, int num) {
    vec3 tmp;
    for (int i = 0; i < num; i++)
    {
        tmp = vec3(center.x + r * rnd2(), center.y, center.z + r * rnd2());
        if (length(center - tmp) <= r)
        {
            controlPoints.push_back(tmp);
        }
    }
}

float deltaOmega(float r, int num, vec3 from, vec3 to, vec3 pointNormal) {
    float A = r * r * M_PI;
    float costheta = dot(normalize((to - from)), normalize(pointNormal * -1));
    float l = length(to - from);
    //return (A / num) * (costheta / (r * r));
    return A / num * costheta / (l * l);
}

class Scene {
    std::vector<Intersectable*> objects;
    std::vector<Intersectable*> objects2;
    std::vector<Light*> lights;
    Camera camera;
    vec3 La;
    Hyperboloid* mirror;

    vec3 Sun = vec3(0.5, 10, 1);

public:
    //TODO: itt
    void build() {
        vec3 eye = vec3(0, 0, 1.8), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.4, 0.4, 0.4);
        vec3 lightDirection(0, 1, 0), Le(2, 2, 2);

        vec3 kd(0.2f, 0.05f, 0.1f), ks(2, 2, 2);
        Material* material1 = new RoughMaterial(kd, ks, 50);

        vec3 kd2(0.1f, 0.1f, 0.8f), ks2(2, 2, 2);
        Material* material2 = new RoughMaterial(kd2, ks2, 150);

        vec3 kd3(0.1f, 0.6f, 0.1f), ks3(2, 2, 2);
        Material* material3 = new RoughMaterial(kd3, ks3, 50);

        Material* gold = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));
        Material* silver = new ReflectiveMaterial(vec3(0.14, 0.16, 0.13), vec3(4.1, 2.3, 3.1));

        vec3 mirrorHoleCenter(0, 0.4974, 0);

        Plane* ellipsoidPlane = new Plane(mirrorHoleCenter, vec3(0, 1, 0), material2, false);
        objects.push_back(new Ellipsoid(vec3(0, 0, 0), 2, 0.5, 2, ellipsoidPlane, material1));
        objects2.push_back(new Ellipsoid(vec3(0, 0, 0), 2, 0.5, 2, ellipsoidPlane, material1));

        Plane* upperEllipsPanel = new Plane(vec3(-0.55, 0.2, 0.1), vec3(0, 1, 0), material3, true);
        Plane* cylinderPlane2 = new Plane(vec3(-0.55, -0.5, 0.1), vec3(0, -1, 0), material3, true);
        objects.push_back(new Cylinder(vec3(-0.55, 0.2, 0.1), 0.05, 0.05, upperEllipsPanel, cylinderPlane2, material3));
        objects2.push_back(new Cylinder(vec3(-0.55, 0.2, 0.1), 0.05, 0.05, upperEllipsPanel, cylinderPlane2, material3));

        objects.push_back(new Ellipsoid(vec3(0.3, -0.35, 0.1), 0.15, 0.09, 0.18, ellipsoidPlane, material2));
        objects2.push_back(new Ellipsoid(vec3(0.3, -0.35, 0.1), 0.15, 0.09, 0.18, ellipsoidPlane, material2));

        Plane* paraboloidPlane = new Plane(vec3(-0.1, -0.5, -0.2), vec3(0, -1, 0), gold, true);
        objects.push_back(new Paraboloid(vec3(-0.1, 0.2, -0.2), 0.3, 0.3, paraboloidPlane, gold));
        objects2.push_back(new Paraboloid(vec3(-0.1, 0.2, -0.2), 0.3, 0.3, paraboloidPlane, gold));

        Plane* mirrorLowerPlane = new Plane(mirrorHoleCenter, vec3(0, -1, 0), material2, false);
        Plane* mirrorUpperPlane = new Plane(vec3(0, 0.7, 0), vec3(0, 1, 0), material1, false);
        mirror = new Hyperboloid(mirrorHoleCenter, 0.2, 0.2, 0.2, mirrorUpperPlane, mirrorLowerPlane, silver);
        objects.push_back(mirror);


        countControlPoints(mirrorHoleCenter, 0.2, 50);
    }

    void render(std::vector<vec4>& image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable* object : objects) {
            Hit hit = object->intersect(ray);
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
        }
        if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
        return bestHit;
    }

    bool shadowIntersect(Ray ray) {
        for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 mirrorTrace2(Ray ray, int depth = 0) {
        Hit hit = mirror->intersect(ray);
        vec3 LeSky(0, 1, 1);
        vec3 LeSun(17, 17, 0);
        //vec3 Sun = vec3(0.5, 1, 1);
        vec3 SunDir = normalize(Sun - ray.start);
        if (depth > 5 || hit.t < 0) return LeSky + LeSun * pow(dot(normalize(ray.dir), SunDir), 10);

        vec3 outRadiance(0, 0, 0);
        vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
        float cosa = -dot(ray.dir, hit.normal);
        vec3 one(1, 1, 1);
        vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
        outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        return outRadiance;
    }

    //TODO: itt
    vec3 trace(Ray ray, int depth = 0) {
        Hit hit = firstIntersect(ray);
        vec3 LeSky(0, 1, 1);
        vec3 LeSun(17, 17, 0);
        vec3 SunDir = normalize(Sun - ray.start);

        if (depth > 5) return La;

        if (hit.t < 0) return LeSky + LeSun * pow(dot(normalize(ray.dir), SunDir), 10);

        vec3 outRadiance(0, 0, 0);
        if (hit.material->type == ROUGH) {

            outRadiance = hit.material->ka * La;

            vec3 mirrorNormal(0, -1, 0);
            for (int i = 0; i < controlPoints.size(); i++)
            {
                Ray r(hit.position + hit.normal * epsilon, normalize(controlPoints[i] - hit.position));
                float cosTheta = dot(hit.normal, normalize(controlPoints[i] - hit.position));
                if (cosTheta > 0 && !shadowIntersect(r))
                {
                    Ray s(controlPoints[i], normalize(controlPoints[i] - hit.position));
                    vec3 tmp = mirrorTrace2(s);
                    float dOmega = deltaOmega(0.2, controlPoints.size(), hit.position, controlPoints[i], mirrorNormal);
                    outRadiance = outRadiance + tmp * hit.material->kd * cosTheta * dOmega;
                    vec3 halfway = normalize(-ray.dir + r.dir);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) outRadiance = outRadiance + tmp * hit.material->ks * powf(cosDelta, hit.material->shininess) * dOmega;
                }
            }
        }
        if (hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            float cosa = -dot(ray.dir, hit.normal);
            vec3 one(1, 1, 1);
            vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
            outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }
        return outRadiance;
    }

    //TODO:Animáció
    void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram;
Scene scene;

const char* const vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

const char* const fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao;
    Texture texture;
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
            : texture(windowWidth, windowHeight, image)
    {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        unsigned int vbo;
        glGenBuffers(1, &vbo);

        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
    }

    void Draw() {
        glBindVertexArray(vao);
        gpuProgram.setUniform(texture, "textureUnit");
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    std::vector<vec4> image(windowWidth * windowHeight);
    long timeStart = glutGet(GLUT_ELAPSED_TIME);
    scene.render(image);
    long timeEnd = glutGet(GLUT_ELAPSED_TIME);
    printf("Rendering time: %d milliseconds\n", (timeEnd - timeStart));

    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

void onDisplay() {
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {
    scene.Animate(0.1f);
    glutPostRedisplay();
}
