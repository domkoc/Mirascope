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

enum MaterialType { ROUGH, REFLECTIVE};

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	vec3 F0;
	MaterialType type;
	Material(MaterialType t) {
		type = t;
	}
};

struct RoughMaterial : Material {
	RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH){
		ka = _kd * M_PI;
		kd = _kd;
		ks = _ks;
		shininess = _shininess;
	}
};

vec3 operator/(vec3 num, vec3 denom) {
	return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

vec3 one(1, 1, 1);

struct ReflectiveMaterial : Material {
	ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
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
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
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

	vec3 gradf(vec3 r) { // r.w = 1
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

/*
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
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};
*/

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
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}

	/*
	void Animate(float dt) {
		vec3 d = eye - lookat;
		eye = vec3(
			d.x * cos(dt) + d.z * sin(dt),
			d.y,
			-d.x * sin(dt) + d.z * cos(dt)) + lookat;
		set(eye, lookat, up, fov);
	}
	*/
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

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	Camera camera;
	vec3 Lsky, La;
	std::vector<vec3> lightPoints;
	float rTube2; // rTube^2
	vec3 tubePos, sunDir, sunRad;
public:
	void build() {
		vec3 eye = vec3(0, 1.8, 0);
		vec3 vup = vec3(0, 0, 1);
		vec3 lookat = vec3(0, 0, 0);
		float fov = 90 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		Lsky = vec3(3, 4, 5);
		La = vec3(0.2f, 0.1f, 0.1f);
		sunDir = normalize(vec3(1, 1, 10));
		sunRad = vec3(100, 100, 100);
		vec3 ks(2, 2, 2);

		float heightStart = 0.99, heightEnd = 2;
		mat4 room = ScaleMatrix(vec3(-0.25f, -0.25f, -1));
		objects.push_back(new Quadrics(room, -2, heightStart, vec3(0, 0, 0), new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), ks, 50)));

		rTube2 = 4 * (1 - heightStart * heightStart);
		tubePos = vec3(0, 0, heightStart);
		mat4 tube = ScaleMatrix(vec3(-1 / rTube2, -1 / rTube2, 1));
		objects.push_back(new Quadrics(tube, heightStart, heightEnd, tubePos, new ReflectiveMaterial(vec3(0.14f, 0.16f, 0.13f), vec3(4.1f, 2.3f, 3.1f))));

		mat4 cylinder = ScaleMatrix(vec3(-9, -9, 0));
		objects.push_back(new Quadrics(cylinder, -2, 0.5, vec3(0.8f, -0.5f, 0), new RoughMaterial(vec3(0.1f, 0.2f, 0.3f), ks, 50)));

		mat4 paraboloid = mat4(
			16, 0, 0, 0,
			0, 16, 0, 0,
			0, 0, 0, 4,
			0, 0, 4, 0
		);
		objects.push_back(new Quadrics(paraboloid, -2, 1, vec3(-0.6f, -0.6f, 0.5f), new ReflectiveMaterial(vec3(0.17f, 0.35f, 1.5f), vec3(3.1f, 2.7f, 1.9f))));

		mat4 hyperboloid = ScaleMatrix(vec3(-128, -129, 9));
		objects.push_back(new Quadrics(hyperboloid, -2, 0.3f, vec3(0.9f, 0.5f, 0), new RoughMaterial(vec3(0.1f, 0.3f, 0.1f), ks, 20)));

		float rTube = sqrtf(rTube2);
		while (lightPoints.size() < 50) {
			vec2 p(rnd() * 2 - 1, rnd() * 2 - 1);
			if (dot(p, p) < 1) lightPoints.push_back(vec3(p.x * rTube, p.y * rTube, heightStart));
		}
	}

	void render(std::vector<vec3>& image) {
		long timeStart = glutGet(GLUT_ELAPSED_TIME);
		
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y), 0);
				image[Y * windowWidth + X] = color;
			}
		}

		printf("Rendering time: %d milliseconds\n", glutGet(GLUT_ELAPSED_TIME) - timeStart);
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth, bool onlyReflective = false) {
		if (depth > 5) return Lsky;
		Hit hit = firstIntersect(ray);
		if (hit.t < 0) return Lsky + sunRad * pow(dot(ray.dir, sunDir), 10);

		vec3 outRadiance(0, 0, 0);
		if (hit.material->type == ROUGH) {
			if (onlyReflective) {
				outRadiance = hit.material->ka * La;
				vec3 shadedPoint = hit.position + hit.normal * epsilon;
				for (auto lightPoint : lightPoints) {
					vec3 lightDir = normalize(lightPoint - shadedPoint);
					float cosTheta = dot(hit.normal, lightDir);
					float cosTheta1 = fabsf(lightDir.z);
					if (cosTheta > 0) {	// shadow computation
						float solidAngle = rTube2 * M_PI * cosTheta1 / dot(lightPoint - shadedPoint, lightPoint - shadedPoint) / lightPoints.size();
						vec3 radience = trace(Ray(shadedPoint, lightDir), depth + 1, true) * solidAngle;
						outRadiance = outRadiance + radience * hit.material->kd * cosTheta;
						vec3 halfway = normalize(-ray.dir + lightDir);
						float cosDelta = dot(hit.normal, halfway);
						if (cosDelta > 0) outRadiance = outRadiance + radience * hit.material->ks * powf(cosDelta, hit.material->shininess);
					}
				}
				return outRadiance;
			}
		}
		else {
			float cosa = -dot(ray.dir, hit.normal);
			vec3 F = hit.material->F0 + (one - hit.material->F0) * pow(1 - cosa, 5);
			vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
			return trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1, false) * F;
		}
	}
	/*
	void Animate(float dt) {
		camera.Animate(dt);
	}
	*/
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
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
const char* fragmentSource = R"(
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
	unsigned int vao;	// vertex array object id and texture id
	unsigned int textureId = 0;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight) {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

		glGenTextures(1, &textureId);
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // sampling
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	}

	void LoadTexture(std::vector<vec3>& image) {
		glBindTexture(GL_TEXTURE_2D, textureId);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, windowWidth, windowHeight, 0, GL_RGB, GL_FLOAT, &image[0]); // TO GPU
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
		const unsigned int textureUnit = 0;
		if (location >= 0) {
			glUniform1i(location, textureUnit);
			glActiveTexture(GL_TEXTURE0 + textureUnit);
			glBindTexture(GL_TEXTURE_2D, textureId);
		}
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
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
	std::vector<vec3> image(windowWidth * windowHeight);
	scene.render(image);								// Execute ray casting
	fullScreenTexturedQuad->LoadTexture(image);		// copy image to GPU as a texture
	fullScreenTexturedQuad->Draw();						// Display rendered image on screen
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	/*
	scene.Animate(0.1f);
	glutPostRedisplay();
	*/
}
