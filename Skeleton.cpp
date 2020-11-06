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

std::vector<vec3> ctrlPointDodekaeder = {
	vec3(0, 0.618, 1.618),
	vec3(0, -0.618, 1.618),
	vec3(0, -0.618 - 1.618),
	vec3(0, 0.618, -1.618),
	vec3(1.618, 0, 0.618),
	vec3(-1.618, 0, 0.618),
	vec3(-1.618, 0, -0.618),
	vec3(1.618, 0, -0.618),
	vec3(0.618, 1.618, 0),
	vec3(-0.618, 1.618, 0),
	vec3(-0.618, -1.618, 0),
	vec3(0.618, -1.618, 0),
	vec3(1, 1, 1),
	vec3(-1, 1, 1),
	vec3(-1, -1, 1),
	vec3(1, -1, 1),
	vec3(1, -1, -1),
	vec3(1, 1, -1),
	vec3(-1, 1, -1),
	vec3(-1, -1, -1)
};

int cornersDodekaeder[12][5] = {
	1, 2, 16, 5, 13,
	1, 13, 9, 10, 14,
	1, 14, 6, 15, 2,
	2, 15, 11, 12, 16,
	3, 4, 18, 8, 17,
	3, 17, 12, 11, 20,
	3, 20, 7, 19, 4,
	19, 10, 9, 18, 4,
	16, 12, 17, 8, 5,
	5, 8, 18, 9, 13,
	14, 10, 19, 7, 6,
	6, 7, 20, 11, 15
};


enum MaterialType { ROUGH, REFLECTIVE };

bool isSpinnig = false;

const float epsilon = 0.0001f;

struct Material {
	vec3 ka, kd, ks;
	float shininess;
	vec3 F0;
	MaterialType type;

	Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) { return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z); }

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
		vec3 one(1, 1, 1);
		F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
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

	Ray(vec3 _start, vec3 _dir) {
		start = _start;
		dir = normalize(_dir);
	}
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};

struct Quadrics : public Intersectable {
	mat4 Q; // symmetric matrix
	float zmin, zmax;
	vec3 translation;

	Quadrics(mat4& _Q, float _zmin, float _zmax, vec3 _translation, Material* _material) {
		Q = _Q;
		zmin = _zmin;
		zmax = _zmax;
		translation = _translation;
		material = _material;
	}

	vec3 gradf(vec3 r) {
		// r.w = 1
		vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 2;
		return vec3(g.x, g.y, g.z);
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 start = ray.start - translation;
		vec4 S(start.x, start.y, start.z, 1);
		vec4 D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
		float a = dot(D * Q, D);
		float b = dot(S * Q, D) * 2;
		float c = dot(S * Q, S);
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);

		float t1 = (-b + sqrt_discr) / 2.0f / a;
		vec3 p1 = ray.start + ray.dir * t1;
		if (p1.z < zmin || p1.z > zmax) t1 = -1;

		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		if (p2.z < zmin || p2.z > zmax) t2 = -1;

		if (t1 <= 0 && t2 <= 0) return hit;
		if (t1 <= 0) hit.t = t2;
		else if (t2 <= 0) hit.t = t1;
		else if (t2 < t1) hit.t = t2;
		else hit.t = t1;

		hit.position = start + ray.dir * hit.t;
		hit.normal = normalize(gradf(hit.position));
		hit.position = hit.position + translation;
		hit.material = material;
		return hit;
	}
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a; // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		//hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};

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
		if (counts == false) {
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

		if (result<epsilon && result >(-1 * epsilon)) {
			return 0;
		}
		if (result < 0) {
			return -1;
		}
		else if (result > 0) {
			return 1;
		}
	}
};

struct KParaboloid : public Intersectable {
	vec3 vertex;
	float a, b;
	Plane* p;

	KParaboloid(const vec3& _vertex, float _a, float _b, Plane* _p, Material* _material) {
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

		if ((p->wichSide(p2) > 0) && (p->wichSide(p1) < 0)) {
			Hit tmp;
			tmp = p->intersect(ray);
			if (tmp.t < t1) {
				hit.t = tmp.t;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = tmp.normal;
				hit.material = tmp.material;
				return hit;
			}
		}

		if (p->wichSide(p1) > 0) {
			t1 = -1;
		}
		if (p->wichSide(p2) > 0) {
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

struct Paraboloid : public Intersectable {
	vec3 focus;
	vec3 n0;
	vec3 p0;
	float zmin, zmax;

	Paraboloid(const vec3& _focus, const vec3& _n0, const vec3& _p0, float _zmin, float _zmax, Material* _material) {
		focus = _focus;
		n0 = _n0;
		p0 = _p0;
		material = _material;
	}

	/*
	 auto a = ((ray.dir * ray.dir) - (n0* n0* ray.dir* ray.dir));
		auto b = ((2* ray.start* ray.dir)-(2* focus* ray.dir)-(2*n0 *n0 *ray.start* ray.dir)+2*n0* n0* p0* ray.dir);
		auto c = (ray.start* ray.start-(2* focus* ray.start)+(focus* focus)-(n0* n0* ray.start* ray.start)+(2*n0* n0 * p0* ray.start)+(n0* n0 * p0* p0));
		auto discr = b * b - 4.0f * a * c;
	 */

	Hit intersect(const Ray& ray) {
		Hit hit;
		auto a = (dot(ray.dir, ray.dir) - dot(n0, n0)* dot(ray.dir, ray.dir));
		auto b = ((2* dot(ray.start, ray.dir))-(2* dot(focus, ray.dir))-(2*dot(n0,n0) * dot(ray.start, ray.dir))+2*dot(n0, n0)* dot(p0, ray.dir));
		auto c = (dot(ray.start, ray.start)-(2*dot(focus, ray.start))+dot(focus, focus)-(dot(n0,n0)* dot(ray.start, ray.start))+(2*dot(n0,n0) * dot(p0, ray.start))+dot(n0, n0) * dot(p0,p0));
		auto discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		
		float t1 = (-b + sqrt_discr) / 2.0f / a; // t1 >= t2 for sure
		vec3 p1 = ray.start + ray.dir * t1;
		//if (p1.z < zmin || p1.z > zmax) t1 = -1;
		
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		vec3 p2 = ray.start + ray.dir * t2;
		//if (p2.z < zmin || p2.z > zmax) t2 = -1;
		
		if (t1 <= 0) return hit;
		
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
		/*
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a; // t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
		*/
	}
};

struct Dodekaeder : Intersectable {
	vec3 center;
	float scale;
	Dodekaeder(const vec3& _center, float _scale, Material* _material) {
		center = _center;
		scale = _scale;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
	}


	/*for (int i = 0; i < 12; ++i)
	{
		for (int j = 0; j < 5; ++j)
		{
			printf("sor: %i, oszlop:%i, ertek: %i \n", i, j, cornersDodekaeder[i][j]);
		}
	}*/
};

class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		float windowSize = focus * tanf(fov / 2);

		right = normalize(cross(vup, w)) * windowSize;
		up = normalize(cross(w, right)) * windowSize;
	}

	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1)
			- eye;
		return Ray(eye, dir);
	}

	void Lift(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(
			d.x,
			d.y * cos(dt) + d.z * sin(dt),
			-d.y * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}

	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(
			d.x * cos(dt) + d.z * sin(dt),
			d.y,
			-d.x * sin(dt) + d.z * cos(dt)) + lookat;
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

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La;
public:
	void build() {
		vec3 eye = vec3(0, 3, 0);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.4f, 0.4f, 0.4f);
		vec3 lightDirection(1, 1, 1),
		     Le(2, 2, 2);
		lights.push_back(new Light(lightDirection, Le));

		vec3 kd1(0.3f, 0.2f, 0.1f);
		vec3 kd2(0.1f, 0.2f, 0.3f);
		vec3 ks(2, 2, 2);
		Material* material1 = new RoughMaterial(kd1, ks, 50);
		vec3 n(1, 1, 1);
		vec3 kappa(5, 4, 3);
		Material* material2 = new ReflectiveMaterial(n, kappa);
		Material* gold = new ReflectiveMaterial(vec3(0.17, 0.35, 1.5), vec3(3.1, 2.7, 1.9));

		//for (int i = 0; i < 150; i++)
		//{
		//objects.push_back(new Sphere(vec3(0.5f, 0.5f, 0.5f), 0.5f, material1));
		//objects.push_back(new Sphere(vec3(-0.5f, -0.5f, -0.5f), 0.5f, material2));

		//}

		//objects.push_back(new Paraboloid(vec3(0, 0, 0), vec3(0, 0, 1), vec3(0, 0 ,2),1.0f, 2.0f, material1));
		//Plane* paraboloidPlane = new Plane(vec3(0, 0, 2), vec3(0, 0, 1), gold, true);
		//objects.push_back(new KParaboloid(vec3(-0.1, 0.2, -0.2), 0.3, 0.3, paraboloidPlane, gold));
		//objects.push_back(new KParaboloid(vec3(-0.1, 0.2, -0.2), 0.3, 0.3, paraboloidPlane, gold));
		/*
		//felfele keskenyedő:
		mat4 upperParaboloid = mat4(
			16, 0, 0, 0,
			0, 16, 0, 0,
			0, 0, 0, 4,
			0, 0, 4, 0
		);
		objects.push_back(new Quadrics(upperParaboloid, 0.5f, 0.9f, vec3(0.0f, 0.0f, 1.0f),
		                               // translation z, az eltolás egymásba
		                               new RoughMaterial(kd1, ks, 50)));

		//lefele keskenyedő:
		mat4 lowerParaboloid = mat4(
			16, 0, 0, 0,
			0, 16, 0, 0,
			0, 0, 0, -4,
			0, 0, -4, 0
		);
		objects.push_back(new Quadrics(lowerParaboloid, 0.0f, 0.5f, vec3(0.0f, 0.0f, 0.0f),
		                               new RoughMaterial(kd2, ks, 50)));
		 */
	}

	void render(std::vector<vec4>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);

		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}

		printf("Rendering time: %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {
		// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		if (depth > 5) return La;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return La;

		vec3 outRadiance(0, 0, 0);

		if (hit.material->type == ROUGH) {
			outRadiance = hit.material->ka * La;
			for (Light* light : lights) {
				Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
				float cosTheta = dot(hit.normal, light->direction);
				if (cosTheta > 0 && !shadowIntersect(shadowRay)) {
					// shadow computation
					outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + light->direction);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0)
						outRadiance = outRadiance + light->Le * hit.material->ks * powf(
							cosDelta, hit.material->shininess);
				}
			}
		}
		if (hit.material->type == REFLECTIVE) {
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			float cosa = -dot(ray.dir, hit.normal);
			vec3 one(1, 1, 1);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			outRadiance = outRadiance +
				trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
		}
		return outRadiance;
	}

	void Lift(float dt) { camera.Lift(dt); }

	void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource =
	R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource =
	R"(
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
	unsigned int vao; // vertex array object id and texture id
	unsigned int textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao); // create 1 vertex array object
		glBindVertexArray(vao); // make it active

		unsigned int vbo; // vertex buffer objects
		glGenBuffers(1, &vbo); // Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1}; // two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
		// copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL); // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec4>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]); // TO GPU
	}

	void Draw() {
		glBindVertexArray(vao); // make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4); // draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor"); // create program for the GPU
}

// Window has become invalid: Redraw
void onDisplay() {
	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image); // Execute ray casting
	fullScreenTexturedQuad->LoadTexture(image); // copy image to GPU as a texture
	fullScreenTexturedQuad->Draw(); // Display rendered image on screen
	glutSwapBuffers(); // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 's') {
		isSpinnig = !isSpinnig;
		glutPostRedisplay();
	}
	if (key == 'f') {
		scene.Lift(-0.1f);
		glutPostRedisplay();
	}
	if (key == 'F') {
		scene.Lift(0.1f);
		glutPostRedisplay();
	}
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	if (isSpinnig) {
		scene.Animate(0.1f);
		glutPostRedisplay();
	}
}
