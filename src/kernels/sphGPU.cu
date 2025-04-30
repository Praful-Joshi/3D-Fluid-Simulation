#include <glm/gtx/norm.hpp>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <neighborTable.h>
#include <kernels/sphGPU.h>

static const float BOX_WIDTH = 10.f;
static const float BOX_COLLISION_ELASTICITY = 1.f;
static const float BOX_COLLISION_OFFSET = 0.00001;
static const uint16_t MAX_NEIGHBORS = 32;


__device__ uint32_t hashDevice(const glm::ivec3 &cell) {
    return ((uint)(cell.x * 73856093) ^ (uint)(cell.y * 19349663) ^ (uint)(cell.z * 83492791)) % TABLE_SIZE;
}

__device__ glm::ivec3 cellDevice(Particle *p, float h) {
    return glm::ivec3(p->position / h);
}

__global__ void calHash(Particle *particles, size_t particleCount, float h) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;
    particles[i].hash = hashDevice(cellDevice(&particles[i], h));
}

struct HashComp {
    __host__ __device__ bool operator()(const Particle &a, const Particle &b) {
        return a.hash < b.hash;
    }
};

__global__ void hashMap(Particle *sortedParticles, size_t particleCount, uint32_t *hashMap) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particleCount) return;

    uint32_t prev = (i == 0) ? NO_PARTICLE : sortedParticles[i - 1].hash;
    uint32_t curr = sortedParticles[i].hash;
    if (curr != prev) hashMap[curr] = i;
}

__global__ void neighbourSearch(Particle *particles, size_t count, const uint32_t *hashMap, SPHSettings settings, Particle **neighborList) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle *pi = &particles[i];
    glm::ivec3 cell = cellDevice(pi, settings.h);
    size_t found = 0;

    for (int dx = -1; dx <= 1; dx++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dz = -1; dz <= 1; dz++) {
                uint16_t hash = hashDevice(cell + glm::ivec3(dx, dy, dz));
                uint32_t j = hashMap[hash];
                if (j == NO_PARTICLE) continue;

                while (j < count) {
                    if (j == i) { j++; continue; }
                    Particle *pj = &particles[j];
                    if (pj->hash != hash) break;
                    float dist2 = glm::length2(pj->position - pi->position);
                    if (dist2 < settings.h2) neighborList[i * MAX_NEIGHBORS + found++] = pj;
                    j++;
                }
            }
        }
    }

    for (size_t k = found; k < MAX_NEIGHBORS; k++) {
        neighborList[i * MAX_NEIGHBORS + k] = nullptr;
    }
}

__global__ void densitiesAndPressure(Particle *particles, size_t count, Particle **neighborList, SPHSettings settings) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle *pi = &particles[i];
    pi->density = settings.selfDens;
    size_t offset = i * MAX_NEIGHBORS;
    const Particle *pj = neighborList[offset];

    while (pj) {
        float dist2 = glm::length2(pj->position - pi->position);
        pi->density += settings.massPoly6Product * glm::pow(settings.h2 - dist2, 3);
        pj = neighborList[++offset];
    }

    pi->pressure = settings.gasConstant * (pi->density - settings.restDensity);
}

__global__ void forces(Particle *particles, size_t count, Particle **neighborList, SPHSettings settings) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle *pi = &particles[i];
    pi->force = glm::vec3(0);
    size_t offset = i * MAX_NEIGHBORS;
    const Particle *pj = neighborList[offset];

    while (pj) {
        float dist = glm::length(pj->position - pi->position);
        glm::vec3 dir = glm::normalize(pj->position - pi->position);

        glm::vec3 pf = -dir * settings.mass * (pi->pressure + pj->pressure) / (2 * pj->density) * settings.spikyGrad;
        pf *= glm::pow(settings.h - dist, 2);
        pi->force += pf;

        glm::vec3 vDif = pj->velocity - pi->velocity;
        glm::vec3 vf = settings.viscosity * settings.mass * (vDif / pj->density) * settings.spikyLap * (settings.h - dist);
        pi->force += vf;

        pj = neighborList[++offset];
    }
}

__global__ void positionUpdate(Particle *particles, size_t count, glm::mat4 *transforms, SPHSettings settings, float dt) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle *p = &particles[i];
    glm::vec3 accel = p->force / p->density + glm::vec3(0, settings.g, 0);
    p->velocity += accel * dt;
    p->position += p->velocity * dt;

    if (p->position.y < settings.h) {
        p->position.y = -p->position.y + 2 * settings.h + BOX_COLLISION_OFFSET;
        p->velocity.y = -p->velocity.y * BOX_COLLISION_ELASTICITY;
    }
    if (p->position.x < settings.h - BOX_WIDTH) {
        p->position.x = -p->position.x + 2 * (settings.h - BOX_WIDTH) + BOX_COLLISION_OFFSET;
        p->velocity.x = -p->velocity.x * BOX_COLLISION_ELASTICITY;
    }
    if (p->position.x > -settings.h + BOX_WIDTH) {
        p->position.x = -p->position.x + 2 * -(settings.h - BOX_WIDTH) - BOX_COLLISION_OFFSET;
        p->velocity.x = -p->velocity.x * BOX_COLLISION_ELASTICITY;
    }
    if (p->position.z < settings.h - BOX_WIDTH) {
        p->position.z = -p->position.z + 2 * (settings.h - BOX_WIDTH) + BOX_COLLISION_OFFSET;
        p->velocity.z = -p->velocity.z * BOX_COLLISION_ELASTICITY;
    }
    if (p->position.z > -settings.h + BOX_WIDTH) {
        p->position.z = -p->position.z + 2 * -(settings.h - BOX_WIDTH) - BOX_COLLISION_OFFSET;
        p->velocity.z = -p->velocity.z * BOX_COLLISION_ELASTICITY;
    }

    transforms[i] = glm::translate(p->position) * settings.sphereScale;
}

void updateParticlesGPU(Particle *particles, glm::mat4 *transforms, size_t count, const SPHSettings &settings, float dt) {
    size_t threads = 512;
    size_t blocks = count / threads + 1;

    thrust::device_vector<Particle> dVec(particles, particles + count);
    Particle *dParticles = thrust::raw_pointer_cast(dVec.data());

    calHash<<<blocks, threads>>>(dParticles, count, settings.h);
    cudaDeviceSynchronize();

    thrust::sort(dVec.begin(), dVec.end(), HashComp());

    thrust::device_vector<uint32_t> dHashMap(TABLE_SIZE, NO_PARTICLE);
    uint32_t *dMap = thrust::raw_pointer_cast(dHashMap.data());
    hashMap<<<blocks, threads>>>(dParticles, count, dMap);
    cudaDeviceSynchronize();

    Particle **dNeighbors;
    cudaMalloc((void**)&dNeighbors, sizeof(Particle *) * count * MAX_NEIGHBORS);
    neighbourSearch<<<blocks, threads>>>(dParticles, count, dMap, settings, dNeighbors);
    cudaDeviceSynchronize();

    densitiesAndPressure<<<blocks, threads>>>(dParticles, count, dNeighbors, settings);
    cudaDeviceSynchronize();

    forces<<<blocks, threads>>>(dParticles, count, dNeighbors, settings);
    cudaDeviceSynchronize();

    cudaFree(dNeighbors);

    glm::mat4 *dTransforms;
    cudaMalloc((void**)&dTransforms, sizeof(glm::mat4) * count);
    positionUpdate<<<blocks, threads>>>(dParticles, count, dTransforms, settings, dt);
    cudaDeviceSynchronize();

    cudaMemcpy(particles, dParticles, sizeof(Particle) * count, cudaMemcpyDeviceToHost);
    cudaMemcpy(transforms, dTransforms, sizeof(glm::mat4) * count, cudaMemcpyDeviceToHost);
    cudaFree(dTransforms);
}
