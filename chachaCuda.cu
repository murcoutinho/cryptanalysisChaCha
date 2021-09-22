/*
Created by Murilo Coutinho on 14/02/21.

Disclaimer: the sole purpose of this code is to document and make it possible to reproduce the results of the paper  
"Improved Linear Approximations to ARX Ciphers and Attacks Against ChaCha", presented at Eurocrypt 2021. 
Thus, the code will not be updated, corrected or modified in the future. We do not guarantee compatibility 
with any particular system or any specific hardware. The code was developed with the objective of obtaining 
a particular result, not to facilitate the reading or execution by third parties. We do not offer any kind 
of support or maintenance. Use it at your own risk. 

Aditional notes: 

1 - We compiled this code in a linux machine using the following command (it may change depending on the gpu):
/usr/local/cuda/bin/nvcc -o  chachaCuda chachaCuda.cu -gencode=arch=compute_75,code=sm_75 -O2

2 - Our code executes on 8 RTX GPUs. To run the code, at least 1 GPU is
necessary. One possible problem to run the code will be the number
of blocks, threads and ammount of computation per thread. Depending on the
cuda compute capability of the device the execution may fail. To reduce the
possibility of error during execution the available code will use only
1 GPU with a low number of blocks and threads. However, this choice makes
the execution time higher. If you get wrong results try to reduce the parameters
NUMBER_OF_THREADS, NUMBER_OF_BLOCKS, NUMBER_OF_TEST_PER_THREAD.

3 - We created a option for a faster test, but keep in mind that this option increases the variance of the results.

4 - We make available some tests replicating results from other authors to check our implementation:
   a) The function testChachaOnGPU executes a test vector of ChaCha
      to test the correctness of the GPU implementation.
   b) The function test_linear_relations_maitra_cuda tests some linear equations
	  presented in "Significantly Improved Multi-bit Differentials for
	  Reduced Round Salsa and ChaCha" from Arka Rai Choudhuri and Subhamoy Maitra.
   c) The function test_aumasson_pnb_attack_chacha_7 tests the attack for 7 rounds of 
      "New Features of Latin Dances" from Aumasson.
	  
5 - Statistical tests are included. The results are printed in LaTeX format only
if the correlation is statistically significant.
*/

//Comment this to use more iterations
//#define REDUCE_NUMBER_OF_ITERATIONS

//Set NUMBER_OF_DEVICES as the number of gpus available in your machine
#define NUMBER_OF_DEVICES 8
#define NUMBER_OF_THREADS (1<<7)
#define NUMBER_OF_BLOCKS (1<<7)
#define NUMBER_OF_TEST_PER_THREAD (1<<15)

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <inttypes.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

void random_uint32(uint32_t *x)
{
	*x = 0;
	*x |= (rand() & 0xFF);
	*x |= (rand() & 0xFF) << 8;
	*x |= (rand() & 0xFF) << 16;
	*x |= (rand() & 0xFF) << 24;
}

void random_uint64(uint64_t *x)
{
	*x = 0;
	*x |= (rand() & 0xFFFF);
	*x |= (((uint64_t)rand() & 0xFFFF)) << 16;
	*x |= ((uint64_t)(rand() & 0xFFFF)) << 32;
	*x |= ((uint64_t)(rand() & 0xFFFF)) << 48;
}

void random_uint32_array(uint32_t *x, uint32_t size)
{
	for (uint32_t i = 0; i < size; i++)
		random_uint32(&x[i]);
}

void transform_state_to_bits(uint32_t state[16], uint8_t bits[512])
{
	int count = 0;
	for (int i = 0; i < 16; i++)
	{
		for (int b = 0; b < 32; b++)
		{
			bits[count] = (state[i] >> b) & 1;
			count++;
		}
	}
}

typedef struct chacha_ctx chacha_ctx;

#define U32C(v) (v##U)

#define U32V(v) ((uint32_t)(v) &U32C(0xFFFFFFFF))

#define ROTATE(v, c) ((v<<c)^(v>>(32-c)))
#define XOR(v, w) ((v) ^ (w))
#define PLUS(v, w) (((v) + (w)))
#define MINUS(v, w) (((v) - (w)))
#define PLUSONE(v) (PLUS((v), 1))

#define QUARTERROUND(a, b, c, d) \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 16);   \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 12);   \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 8);    \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 7);

#define HALF_1_QUARTERROUND(a, b, c, d) \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 16);   \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 12);   \

#define HALF_2_QUARTERROUND(a, b, c, d) \
a = PLUS(a, b);              \
d = ROTATE(XOR(d, a), 8);    \
c = PLUS(c, d);              \
b = ROTATE(XOR(b, c), 7);

#define INVERT_QUARTERROUND(a,b,c,d)\
b = XOR(ROTATE(b,25), c); \
c = MINUS(c, d);              \
d = XOR(ROTATE(d,24), a); \
a = MINUS(a, b);              \
b = XOR(ROTATE(b,20), c); \
c = MINUS(c, d);              \
d = XOR(ROTATE(d,16), a); \
a = MINUS(a, b);

#define LOAD32_LE(v) (*((uint32_t *) (v)))
#define STORE32_LE(c,x) (memcpy(c,&x,4))

__host__ __device__ void chacha_init(uint32_t state[16], uint32_t k[8], uint32_t nonce[2], uint32_t ctr[2])
{
	state[0] = U32C(0x61707865);
	state[1] = U32C(0x3320646e);
	state[2] = U32C(0x79622d32);
	state[3] = U32C(0x6b206574);
	state[4] = k[0];
	state[5] = k[1];
	state[6] = k[2];
	state[7] = k[3];
	state[8] = k[4];
	state[9] = k[5];
	state[10] = k[6];
	state[11] = k[7];
	state[12] = ctr[0];
	state[13] = ctr[1];
	state[14] = nonce[0];
	state[15] = nonce[1];
}

__host__ __device__ void chacha_odd_round(uint32_t x[16])
{
	QUARTERROUND(x[0], x[4], x[8], x[12])
		QUARTERROUND(x[1], x[5], x[9], x[13])
		QUARTERROUND(x[2], x[6], x[10], x[14])
		QUARTERROUND(x[3], x[7], x[11], x[15])
}

__host__ __device__ void chacha_odd_half_round(uint32_t x[16], int half)
{
	if (half == 1)
	{
		HALF_1_QUARTERROUND(x[0], x[4], x[8], x[12])
			HALF_1_QUARTERROUND(x[1], x[5], x[9], x[13])
			HALF_1_QUARTERROUND(x[2], x[6], x[10], x[14])
			HALF_1_QUARTERROUND(x[3], x[7], x[11], x[15])
	}
	else
	{
		HALF_2_QUARTERROUND(x[0], x[4], x[8], x[12])
			HALF_2_QUARTERROUND(x[1], x[5], x[9], x[13])
			HALF_2_QUARTERROUND(x[2], x[6], x[10], x[14])
			HALF_2_QUARTERROUND(x[3], x[7], x[11], x[15])
	}
}

__host__ __device__ void chacha_even_round(uint32_t x[16])
{
	QUARTERROUND(x[0], x[5], x[10], x[15])
		QUARTERROUND(x[1], x[6], x[11], x[12])
		QUARTERROUND(x[2], x[7], x[8], x[13])
		QUARTERROUND(x[3], x[4], x[9], x[14])
}

__host__ __device__ void chacha_even_half_round(uint32_t x[16], int half)
{
	if (half == 1)
	{
		HALF_1_QUARTERROUND(x[0], x[5], x[10], x[15])
			HALF_1_QUARTERROUND(x[1], x[6], x[11], x[12])
			HALF_1_QUARTERROUND(x[2], x[7], x[8], x[13])
			HALF_1_QUARTERROUND(x[3], x[4], x[9], x[14])
	}
	else
	{
		HALF_2_QUARTERROUND(x[0], x[5], x[10], x[15])
			HALF_2_QUARTERROUND(x[1], x[6], x[11], x[12])
			HALF_2_QUARTERROUND(x[2], x[7], x[8], x[13])
			HALF_2_QUARTERROUND(x[3], x[4], x[9], x[14])
	}
}

__host__ __device__ void chacha_invert_odd_round(uint32_t x[16])
{
	INVERT_QUARTERROUND(x[3], x[7], x[11], x[15])
		INVERT_QUARTERROUND(x[2], x[6], x[10], x[14])
		INVERT_QUARTERROUND(x[1], x[5], x[9], x[13])
		INVERT_QUARTERROUND(x[0], x[4], x[8], x[12])
}

__host__ __device__ void chacha_invert_even_round(uint32_t x[16])
{
	INVERT_QUARTERROUND(x[3], x[4], x[9], x[14])
		INVERT_QUARTERROUND(x[2], x[7], x[8], x[13])
		INVERT_QUARTERROUND(x[1], x[6], x[11], x[12])
		INVERT_QUARTERROUND(x[0], x[5], x[10], x[15])
}

__host__ __device__ void chacha_rounds(uint32_t state[16], uint32_t rounds, uint32_t lastRound)
{
	uint32_t i;

	for (i = 1; i <= rounds; i++) {
		if ((i + lastRound) % 2)
			chacha_odd_round(state);
		else
			chacha_even_round(state);
	}
}

__host__ __device__ void chacha_half_rounds(uint32_t state[16], uint32_t rounds, uint32_t lastRound, int half)
{
	uint32_t i;

	for (i = 1; i <= rounds; i++) {
		if ((i + lastRound) % 2)
			chacha_odd_half_round(state, half);
		else
			chacha_even_half_round(state, half);
	}
}

__host__ __device__ void chacha_invert_rounds(uint32_t state[16], uint32_t rounds, uint32_t lastRound)
{
	uint32_t i;

	lastRound = lastRound % 2;

	if (lastRound)
	{
		for (i = 1; i <= rounds; i++) {
			if (i % 2)
				chacha_invert_odd_round(state);
			else
				chacha_invert_even_round(state);
		}
	}
	else
	{
		for (i = 1; i <= rounds; i++) {
			if (i % 2)
				chacha_invert_even_round(state);
			else
				chacha_invert_odd_round(state);
		}
	}
}

__host__ __device__ void chacha_encrypt(uint32_t output[16], uint32_t input[16], uint32_t rounds)
{
	uint32_t x[16];
	uint32_t i;

	for (i = 0; i < 16; ++i) x[i] = input[i];
	chacha_rounds(x, rounds, 0);
	for (i = 0; i < 16; ++i) x[i] = PLUS(x[i], input[i]);

	memcpy(output, x, 64);
}

__host__ __device__ void chacha_invert(uint32_t output[16], uint32_t input[16], uint32_t intermediate[16], uint32_t rounds, uint32_t lastRound)
{
	for (int i = 0; i < 16; ++i) intermediate[i] = MINUS(output[i], input[i]);
	chacha_invert_rounds(intermediate, rounds, lastRound);
}

#define ALG_TYPE_SALSA 0
#define ALG_TYPE_CHACHA 1

typedef struct {
	uint32_t algType;
	uint32_t key_positions[8];
	uint32_t iv_positions[4];
	void(*init)(uint32_t *, uint32_t *, uint32_t *, uint32_t *);
	void(*encrypt)(uint32_t *, uint32_t *, uint32_t);
	void(*rounds)(uint32_t *, uint32_t, uint32_t);
	void(*halfrounds)(uint32_t *, uint32_t, uint32_t, int);
	void(*invert)(uint32_t *, uint32_t *, uint32_t *, uint32_t, uint32_t);
	char name[20];
} ALGORITHM;

__host__ __device__ void define_alg(ALGORITHM *alg, uint32_t type)
{
	uint32_t chacha_iv_positions[4] = { 12,13,14,15 };
	uint32_t chacha_key_positions[8] = { 4,5,6,7,8,9,10,11 };

	switch (type)
	{
	case ALG_TYPE_CHACHA:
		memcpy(alg->key_positions, chacha_key_positions, 8 * sizeof(uint32_t));
		memcpy(alg->iv_positions, chacha_iv_positions, 4 * sizeof(uint32_t));
		alg->algType = ALG_TYPE_CHACHA;
		alg->init = &chacha_init;
		alg->encrypt = &chacha_encrypt;
		alg->invert = &chacha_invert;
		alg->rounds = &chacha_rounds;
		alg->halfrounds = &chacha_half_rounds;
		alg->name[0] = 'C'; alg->name[1] = 'h'; alg->name[2] = 'a'; alg->name[3] = 'c'; alg->name[4] = 'h'; alg->name[5] = 'a'; alg->name[6] = 0;
		break;

	default:
		break;
	}
}

__host__ __device__ void xor_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] ^ y[i];
}

__host__ __device__ void sub_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] - y[i];
}

__host__ __device__ uint8_t get_bit_in_position(uint32_t state[16], uint32_t pos)
{
	int w = pos / 32;
	int bit = pos % 32;

	return((state[w] >> bit) & 1);
}

__host__ __device__ uint8_t get_bit_from_word_and_bit(uint32_t state[16], uint32_t w, uint32_t bit)
{
	return((state[w] >> bit) & 1);
}

__host__ __device__ void set_bit(uint32_t state[16], uint32_t w, uint32_t bit)
{
	state[w] ^= (1 << bit);
}

__host__ __device__ void set_list_of_bits(uint32_t state[16], uint32_t *w, uint32_t *bit, uint32_t numberOfBits)
{
	for (uint32_t i = 0; i < numberOfBits; i++)
		set_bit(state, w[i], bit[i]);
}

__host__ __device__ void and_array(uint32_t *z, uint32_t *x, uint32_t *y, int size)
{
	for (int i = 0; i < size; i++)
		z[i] = x[i] & y[i];
}

__host__ __device__ uint8_t xor_bits_of_state(uint32_t state[16])
{
	uint32_t x = state[0];
	for (int i = 1; i < 16; i++)
		x ^= state[i];

	x = x ^ (x >> 16);
	x = x ^ (x >> 8);
	x = x ^ (x >> 4);
	x = x ^ (x >> 2);
	return ((x ^ (x >> 1)) & 1);
}

__host__ __device__ uint8_t check_parity_of_equation(uint32_t state[16], uint32_t ODmask[16])
{
	uint32_t aux[16];

	and_array(aux, state, ODmask, 16);
	return(xor_bits_of_state(aux));
}

__device__ uint8_t check_parity_of_linear_relation_cuda(uint32_t inputMask[16], uint32_t inputState[16], uint32_t outputMask[16], uint32_t outputState[16])
{
	uint32_t aux[16], aux2[16];

	and_array(aux, inputState, inputMask, 16);
	and_array(aux2, outputState, outputMask, 16);
	xor_array(aux, aux, aux2, 16);

	return(xor_bits_of_state(aux));
}


__device__ uint8_t chacha_test1_out[64] =
{
	0x76,0xb8,0xe0,0xad,0xa0,0xf1,0x3d,0x90,0x40,0x5d,0x6a,0xe5,0x53,0x86,0xbd,0x28,0xbd,0xd2,0x19,0xb8,0xa0,0x8d,0xed,0x1a,0xa8,0x36,0xef,0xcc,0x8b,0x77,0x0d,0xc7,0xda,0x41,0x59,0x7c,0x51,0x57,0x48,0x8d,0x77,0x24,0xe0,0x3f,0xb8,0xd8,0x4a,0x37,0x6a,0x43,0xb8,0xf4,0x15,0x18,0xa1,0x1c,0xc3,0x87,0xb6,0x69,0xb2,0xee,0x65,0x86
};

//---------------------------------------------------------------------------------------
//----------------------  Kernels
//---------------------------------------------------------------------------------------
__device__ int cudaCmp(uint8_t *v1, uint8_t *v2, int len)
{
	for (int i = 0; i < len; i++)
		if (v1[i] != v2[i])
			return 1;

	return 0;
}

/*
This function executes a test vector for chacha to check gpu implementation
*/
__global__ void testChachaKernel(int *rv)
{
	uint32_t k[8] = { 0 }, ctr[2] = { 0 }, nonce[2] = { 0 }, state[16] = { 0 }, state_final[16], inverted[16];
	ALGORITHM alg;
	int tx = threadIdx.x;

	//printf("Teste chacha %d\n", tx);
	define_alg(&alg, ALG_TYPE_CHACHA);

	alg.init(state, k, nonce, ctr);
	alg.encrypt(state_final, state, 20);

	if (cudaCmp((uint8_t *)state_final, chacha_test1_out, 64))
		rv[tx] = 1;
	else
		rv[tx] = 0;

	alg.invert(state_final, state, inverted, 20, 20);
	if (cudaCmp((uint8_t *)inverted, (uint8_t *)state, 64))
		rv[tx] = 2;
}

/*
Computes the differential bias given the number of rounds, ID and OD mask.
*/
__global__ void differential_bias_kernel(unsigned long long seed, int rounds, uint32_t *ID,
	uint32_t *ODmask, int ntestForEachThread, unsigned long long int *d_result, int algType, int addExtraHalfRound, int startFromSecondRound)
{
	ALGORITHM alg;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t K[8] = { 0 }, state[16] = { 0 }, alt_state[16] = { 0 };
	uint32_t nonce[2] = { 0 }, ctr[2] = { 0 };
	curandState_t rng;
	unsigned long long int sumParity = 0;

	//printf("Ola sou a td %d\n", tid);
	define_alg(&alg, algType);
	curand_init(seed, tid, 0, &rng);

	for (int t = 0; t < ntestForEachThread; t++)
	{
		if (startFromSecondRound)
		{
			for (int i = 0; i < 16; i++)
				state[i] = curand(&rng);
		}
		else
		{
			for (int i = 0; i < 8; i++)
				K[i] = curand(&rng);

			nonce[0] = curand(&rng); nonce[1] = curand(&rng);
			ctr[0] = curand(&rng); ctr[1] = curand(&rng);

			alg.init(state, K, nonce, ctr);
		}
		xor_array(alt_state, state, ID, 16);

		alg.rounds(state, rounds, startFromSecondRound);
		alg.rounds(alt_state, rounds, startFromSecondRound);
		if (addExtraHalfRound)
		{
			alg.halfrounds(state, 1, rounds+1, 1);
			alg.halfrounds(alt_state, 1, rounds+1, 1);
		}

		xor_array(state, state, alt_state, 16);
		sumParity += check_parity_of_equation(state, ODmask);
	}

	atomicAdd(d_result, sumParity);
}

/*
Computes the linear bias given the number of rounds, ID and OD mask.
*/
__global__ void linear_bias_kernel(unsigned long long seed, int outputRound, int inputRound, uint32_t *IDmask, uint32_t *ODmask, int ntestForEachThread, int *d_result, int algType)
{
	ALGORITHM alg;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t inputState[16] = { 0 }, outputState[16] = { 0 };
	curandState_t rng;
	uint32_t sumParity = 0;

	//printf("Ola sou a td %d\n", tid);
	define_alg(&alg, algType);
	curand_init(seed, tid, 0, &rng);

	for (int t = 0; t < ntestForEachThread; t++)
	{
		for (int i = 0; i < 16; i++)
			inputState[i] = curand(&rng);

		for (int i = 0; i < 16; i++)
			outputState[i] = inputState[i];

		alg.rounds(outputState, outputRound - inputRound, inputRound);
		//chacha_half_rounds(outputState, outputRound - inputRound, inputRound, 2);

		sumParity += check_parity_of_linear_relation_cuda(IDmask, inputState, ODmask, outputState);
	}

	atomicAdd(d_result, (int)sumParity);
}

/*This function computes \varepsilon_a from a PNB attack as presented in aumasson 2008*/
__global__ void compute_bias_of_g_for_random_key_kernel(
	unsigned long long seed, uint32_t enc_rounds, uint32_t dec_rounds,
	uint32_t *IDmask, uint32_t *ODmask,
	uint32_t *pnb, uint32_t number_of_pnb, int ntestForEachThread,
	int *d_result, int algType
)
{
	ALGORITHM alg;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t K_with_zeros[8] = { 0 }, state[16] = { 0 }, alt_state[16] = { 0 };
	uint32_t final_state[16] = { 0 }, alt_final_state[16] = { 0 }, aux[16];
	uint32_t intermediate_state[16] = { 0 }, alt_intermediate_state[16] = { 0 };
	uint32_t nonce[2] = { 0 }, ctr[2] = { 0 };
	curandState_t rng;
	uint32_t f_parity, g_parity;
	uint32_t sumParity = 0;
	uint32_t mask;

	uint32_t Krand[8];

	//printf("Ola sou a td %d\n", tid);
	define_alg(&alg, algType);
	curand_init(seed, tid, 0, &rng);

	for (int i = 0; i < 8; i++)
		Krand[i] = curand(&rng);

	for (int i = 0; i < 8; i++)
		K_with_zeros[i] = Krand[i];

	for (uint32_t j = 0; j < number_of_pnb; j++)
	{
		mask = ~(1 << (pnb[j] % 32));
		K_with_zeros[pnb[j] / 32] = K_with_zeros[pnb[j] / 32] & mask;
	}

	for (int t = 0; t < ntestForEachThread; t++)
	{
		nonce[0] = curand(&rng); nonce[1] = curand(&rng);
		ctr[0] = curand(&rng); ctr[1] = curand(&rng);

		//compute for f
		alg.init(state, Krand, nonce, ctr);
		xor_array(alt_state, state, IDmask, 16);

		alg.encrypt(final_state, state, enc_rounds);
		alg.encrypt(alt_final_state, alt_state, enc_rounds);

		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		f_parity = check_parity_of_equation(aux, ODmask);

		//compute for g
		alg.init(state, K_with_zeros, nonce, ctr);
		xor_array(alt_state, state, IDmask, 16);

		//use the same final state
		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		g_parity = check_parity_of_equation(aux, ODmask);

		if (f_parity == g_parity)
			sumParity++;
	}

	atomicAdd(d_result, (int)sumParity);
}

int testChachaOnGPU()
{
	int *d_rvs;
	const int numThreads = 1024;
	int results[numThreads];

	cudaMalloc((void **)&d_rvs, sizeof(int) * numThreads);
	testChachaKernel <<< 1, numThreads >>> (d_rvs);
	cudaDeviceSynchronize(); //make it flush

	cudaMemcpy(results, d_rvs, sizeof(int)*numThreads, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numThreads; i++)
		if (results[i])
			return 1;

	cudaFree(d_rvs);

	return 0;
}


/*
Compute the differential bias for chacha given the number of rounds, the input differential and the output differential.

addExtraHalfRound - use 1 to add half round
startFromSecondRound - use 1 to start the computation from the second round. This is used when applying a technique from bierle et. al. crypto 2020
*/
double compute_differential_bias(
	int rounds,
	int algType,
	uint32_t ID[16],
	uint32_t ODmask[16],
	uint64_t N, //a power of 2
	int addExtraHalfRound,
	int startFromSecondRound
)
{
	int nTestsForEachThread = NUMBER_OF_TEST_PER_THREAD, nThreads = NUMBER_OF_THREADS, nBlocks = NUMBER_OF_BLOCKS;
	int executionsPerKernel = nTestsForEachThread * nThreads*nBlocks;
	uint64_t iterations;
	unsigned long long int *dSumParity;
	uint32_t *dID, *dODmask;
	unsigned long long int localSumParity = 0;
	double prob = 0;

	uint64_t seed = rand();
	random_uint64(&seed);

	iterations = N / (executionsPerKernel);

	cudaMalloc(&dSumParity, sizeof(unsigned long long int));
	cudaMalloc(&dID, 16 * sizeof(uint32_t));
	cudaMalloc(&dODmask, 16 * sizeof(uint32_t));

	cudaMemcpy(dID, ID, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dODmask, ODmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	for (int i = 0; i < iterations; i++)
	{
		random_uint64(&seed);
		localSumParity = 0;
		cudaMemcpy(dSumParity, &localSumParity, sizeof(unsigned long long int), cudaMemcpyHostToDevice);

		differential_bias_kernel <<< nBlocks, nThreads >>> ((unsigned long long)seed, rounds, dID, dODmask, nTestsForEachThread, dSumParity, algType, addExtraHalfRound, startFromSecondRound);
		cudaDeviceSynchronize(); //make it flush

		cudaMemcpy(&localSumParity, dSumParity, sizeof(unsigned long long int), cudaMemcpyDeviceToHost);

		prob += ((double)(executionsPerKernel - localSumParity)) / executionsPerKernel;
	}
	prob /= iterations;

	cudaFree(dSumParity);
	cudaFree(dID);
	cudaFree(dODmask);

	return(2 * prob - 1);
}

/*
Compute the differential bias for chacha given the number of rounds, the input differential and the output differential.

addExtraHalfRound - use 1 to add half round
startFromSecondRound - use 1 to start the computation from the second round. This is used when applying a technique from bierle et. al. crypto 2020
*/
double compute_differential_bias_multiple_devices(
	int rounds,
	int algType,
	uint32_t ID[16],
	uint32_t ODmask[16],
	uint64_t N, //a power of 2
	int addExtraHalfRound,
	int startFromSecondRound
)
{
	//uint64_t nTestsForEachThread = (1 << 18), nThreads = (1 << 9), nBlocks = (1 << 8);
	uint64_t nTestsForEachThread = NUMBER_OF_TEST_PER_THREAD, nThreads = NUMBER_OF_THREADS, nBlocks = NUMBER_OF_BLOCKS;
	uint64_t executionsPerKernel = nTestsForEachThread * nThreads*nBlocks;
	uint64_t iterations;
	unsigned long long int  *dSumParityVec[8];
	uint32_t *dIDVec[8], *dODmaskVec[8];
	unsigned long long int localSumParityVec[8] = { 0 };
	double prob = 0;

	uint64_t seed = rand();
	random_uint64(&seed);

	iterations = N / (executionsPerKernel * NUMBER_OF_DEVICES);
	//printf("size of long long int = %d\n",sizeof(unsigned long long int));

	for (int d = 0; d < NUMBER_OF_DEVICES; d++)
	{
		cudaSetDevice(d);
		cudaMalloc(&(dSumParityVec[d]), sizeof(unsigned long long int));
		cudaMalloc(&(dIDVec[d]), 16 * sizeof(uint32_t));
		cudaMalloc(&(dODmaskVec[d]), 16 * sizeof(uint32_t));

		cudaMemcpy(dIDVec[d], ID, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
		cudaMemcpy(dODmaskVec[d], ODmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	}

	//printf("iterations %d = \n", iterations);
	for (int i = 0; i < iterations; i++)
	{
		for (int d = 0; d < NUMBER_OF_DEVICES; d++)
		{
			cudaSetDevice(d);
			random_uint64(&seed);
			localSumParityVec[d] = 0;
			cudaMemcpy(dSumParityVec[d], &(localSumParityVec[d]), sizeof(unsigned long long int), cudaMemcpyHostToDevice);

			differential_bias_kernel <<< nBlocks, nThreads >>> ((unsigned long long)seed, rounds,
				dIDVec[d], dODmaskVec[d], nTestsForEachThread, dSumParityVec[d], algType,
				addExtraHalfRound, startFromSecondRound);
		}
		cudaDeviceSynchronize(); //make it flush

		for (int d = 0; d < NUMBER_OF_DEVICES; d++)
		{
			cudaSetDevice(d);
			cudaMemcpy(&(localSumParityVec[d]), dSumParityVec[d], 
				sizeof(unsigned long long int), cudaMemcpyDeviceToHost);
			//printf("%" PRIu64 "\n", executionsPerKernel);
			prob += ((double)(executionsPerKernel - localSumParityVec[d])) / executionsPerKernel;
		}
	}
	prob /= (iterations*NUMBER_OF_DEVICES);

	for (int d = 0; d < NUMBER_OF_DEVICES; d++)
	{
		cudaFree(dSumParityVec[d]);
		cudaFree(dIDVec[d]);
		cudaFree(dODmaskVec[d]);
	}

	return(2 * prob - 1);
}

//compute linear correlation using cuda
double compute_linear_bias_cuda(
	int inputRound,
	int outputRound,
	int algType,
	uint32_t IDmask[16],
	uint32_t ODmask[16],
	uint64_t N //a power of 2
)
{
	//int nTestsForEachThread = (1 << 6), nThreads = (1 << 6), nBlocks = (1 << 6);
	int nTestsForEachThread = NUMBER_OF_TEST_PER_THREAD, nThreads = NUMBER_OF_THREADS, nBlocks = NUMBER_OF_BLOCKS;
	int executionsPerKernel = nTestsForEachThread * nThreads*nBlocks;
	uint64_t iterations;
	int *dSumParity;
	uint32_t *dID, *dODmask;
	int localSumParity = 0;
	double prob = 0;

	uint64_t seed = rand();
	random_uint64(&seed);

	iterations = N / (executionsPerKernel);

	cudaMalloc(&dSumParity, sizeof(int));
	cudaMalloc(&dID, 16 * sizeof(uint32_t));
	cudaMalloc(&dODmask, 16 * sizeof(uint32_t));

	cudaMemcpy(dID, IDmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dODmask, ODmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	for (int i = 0; i < iterations; i++)
	{
		random_uint64(&seed);
		localSumParity = 0;
		cudaMemcpy(dSumParity, &localSumParity, sizeof(int), cudaMemcpyHostToDevice);

		linear_bias_kernel <<< nBlocks, nThreads >>> ((unsigned long long)seed, outputRound, inputRound, dID, dODmask, nTestsForEachThread, dSumParity, algType);

		//cudaDeviceSynchronize(); //make it flush

		cudaMemcpy(&localSumParity, dSumParity, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		prob += ((double)(executionsPerKernel - localSumParity)) / executionsPerKernel;
	}
	prob /= iterations;

	cudaFree(dSumParity);
	cudaFree(dID);
	cudaFree(dODmask);

	return(2 * prob - 1);
}

/*
compute \varepsion_a in a PNB attack as in aumasson 2008
*/
double compute_mean_bias_of_g_cuda(
	uint64_t N,
	uint32_t ID[16],
	uint32_t ODmask[16],
	uint32_t enc_rounds,
	uint32_t dec_rounds,
	uint32_t *pnb,
	uint32_t number_of_pnb,
	ALGORITHM alg
)
{
	//int nTestsForEachThread = (1 << 7), nThreads = (1 << 8), nBlocks = (1 << 8);
	int nTestsForEachThread = NUMBER_OF_TEST_PER_THREAD, nThreads = NUMBER_OF_THREADS, nBlocks = NUMBER_OF_BLOCKS;
	int executionsPerKernel = nTestsForEachThread * nThreads*nBlocks;
	uint64_t iterations;
	int *dSumParity;
	uint32_t *dID, *dODmask, *dPNB;
	int localSumParity = 0;
	double prob = 0;

	uint64_t seed = rand();
	random_uint64(&seed);

	iterations = N / (executionsPerKernel);

	cudaMalloc(&dSumParity, sizeof(int));
	cudaMalloc(&dID, 16 * sizeof(uint32_t));
	cudaMalloc(&dODmask, 16 * sizeof(uint32_t));
	cudaMalloc(&dPNB, number_of_pnb * sizeof(uint32_t));

	cudaMemcpy(dID, ID, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dODmask, ODmask, 16 * sizeof(uint32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(dPNB, pnb, number_of_pnb * sizeof(uint32_t), cudaMemcpyHostToDevice);

	for (int i = 0; i < iterations; i++)
	{
		random_uint64(&seed);
		localSumParity = 0;
		cudaMemcpy(dSumParity, &localSumParity, sizeof(int), cudaMemcpyHostToDevice);

		compute_bias_of_g_for_random_key_kernel <<< nBlocks, nThreads >>> ((unsigned long long)seed,
			enc_rounds, dec_rounds, dID, dODmask, dPNB, number_of_pnb, nTestsForEachThread,
			dSumParity, alg.algType);
		//cudaDeviceSynchronize(); //make it flush

		cudaMemcpy(&localSumParity, dSumParity, sizeof(uint32_t), cudaMemcpyDeviceToHost);

		prob += ((double)(localSumParity)) / executionsPerKernel;
	}
	prob /= iterations;

	cudaFree(dSumParity);
	cudaFree(dID);
	cudaFree(dODmask);

	return(2 * prob - 1);
}

double get_max(double x, double y)
{
	if (x>y)
		return x;
	else
		return y;
}


//compute complexity of pnb attack
void compute_complexity_of_the_attack(double *data_complexity, double *time_complexity, double bias_of_g, int number_of_pnb)
{
	int alpha;
	int m = 256 - number_of_pnb;
	double N, tc, minN = 256, minTC = 256;

	for (alpha = 0; alpha < 256; alpha++)
	{
		N = (sqrt(alpha*log(4)) + 3 * sqrt(1 - bias_of_g * bias_of_g)) / bias_of_g;
		N = N * N;
		tc = get_max(256 - alpha, m + log2(N));

		if (tc < minTC)
		{
			minTC = tc;
			minN = N;
		}
	}

	*data_complexity = log2(minN);
	*time_complexity = minTC;
}


/*
Alg of sec 3.3 of New features of latin dances.
neutrality_measure - is the neutrality measure of this neutral bit.
N - number of tests to be executed
input_differential - the input differential (ID) under analysis
enc_rounds - in the paper is R
dec_rounds - in the paper is r-R
od_word - identify the word of the output differential (0 to 16)
od_bit_of_word - identify the bit of the word for the output differential (0 to 31)
neutral_word - identify the word in which is located the neutral bit being tested (0 to 16)
neutral_bit - identify the bit of the word for the neutral bit being tested (0 to 31)
init, encrypt and invert are pointer to salsa or chacha or any other initialization, encryption, or inversion functions
*/
void compute_neutrality_of_bit(
	double *neutrality_measure,
	uint64_t N, uint32_t ID[16], uint32_t ODmask[16],
	uint32_t enc_rounds, uint32_t dec_rounds,
	uint32_t neutral_word, uint32_t neutral_bit,
	ALGORITHM alg)
{
	uint32_t K[8] = { 0 }, state[16] = { 0 }, alt_state[16] = { 0 };
	uint32_t final_state[16] = { 0 }, alt_final_state[16] = { 0 };
	uint32_t intermediate_state[16] = { 0 }, alt_intermediate_state[16] = { 0 }, aux[16];
	uint32_t nonce[2] = { 0 }, ctr[2] = { 0 };
	uint32_t diff, new_diff, neutral_diff;
	uint64_t count = 0;

	neutral_diff = 1 << neutral_bit;

	for (uint64_t i = 0; i < N; i++)
	{
		random_uint32_array(K, 8);
		random_uint32_array(nonce, 2);
		random_uint32_array(ctr, 2);

		alg.init(state, K, nonce, ctr);
		xor_array(alt_state, state, ID, 16);

		alg.encrypt(final_state, state, enc_rounds);
		alg.encrypt(alt_final_state, alt_state, enc_rounds);

		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		diff = check_parity_of_equation(aux, ODmask);

		state[neutral_word] ^= neutral_diff;
		alt_state[neutral_word] ^= neutral_diff;

		alg.invert(final_state, state, intermediate_state, dec_rounds, enc_rounds);
		alg.invert(alt_final_state, alt_state, alt_intermediate_state, dec_rounds, enc_rounds);

		xor_array(aux, intermediate_state, alt_intermediate_state, 16);
		new_diff = check_parity_of_equation(aux, ODmask);

		if (diff == new_diff)
			count++;
	}

	*neutrality_measure = 2 * ((double)count) / N - 1;
}

void compute_neutrality_for_every_key_bit(
	double neutrality_list[256],
	uint64_t number_of_tests_for_each_key_bit,
	uint32_t ID[16], uint32_t ODmask[16],
	uint32_t enc_rounds, uint32_t dec_rounds,
	ALGORITHM alg)
{
	int count = 0;
	double neutrality_measure = 0;

	for (uint32_t neutral_word = 0; neutral_word < 8; neutral_word++)
	{
		for (uint32_t neutral_bit = 0; neutral_bit < 32; neutral_bit++)
		{
			//printf("%d %d\n", neutral_word, neutral_bit);
			compute_neutrality_of_bit(&neutrality_measure, number_of_tests_for_each_key_bit, ID, ODmask,
				enc_rounds, dec_rounds, alg.key_positions[neutral_word], neutral_bit, alg);

			neutrality_list[count] = neutrality_measure;
			count++;
		}
	}
}

void get_pnb_list(uint32_t pnb[256], uint32_t *number_of_pnb,
	double neutrality_measure_threshold, double neutrality_list[256])
{
	*number_of_pnb = 0;
	for (uint32_t i = 0; i < 256; i++)
	{
		if (neutrality_list[i] > neutrality_measure_threshold)
		{
			pnb[*number_of_pnb] = i;
			(*number_of_pnb)++;
		}
	}
}

/*
Test for checking the results presented in aumasson 2008
In the paper we have:

neutral bits:
{3, 6, 15, 16, 31, 35, 67, 68, 71, 91, 92, 93, 94, 95, 96, 97, 98, 99,
100, 103, 104, 127, 136, 191, 223, 224, 225, 248, 249, 250, 251, 252, 253, 254,
255}

\varepsilon_a = 0.023

Time complexity = 248
Data complexity = 27
*/
void test_aumasson_pnb_attack_chacha_7()
{
	uint32_t ODmask[16] = { 0 };
	uint32_t ID[16] = { 0 };
	uint64_t N = (1 << 20);
	ALGORITHM alg;
	double neutrality_list[256];
	uint32_t pnb[256], number_of_pnb;
	double threshold = 0.5, varepsilon_a, varepsilon_d;

	define_alg(&alg, ALG_TYPE_CHACHA);

	//Go 4, back 2
	memset(ODmask, 0x00, sizeof(uint32_t) * 16);
	N = 1;
	N <<= 14;
	ID[13] = (1 << 13);
	uint32_t finalListOfWords[1] = { 11 };
	uint32_t finalListOfBits[1] = { 0 };
	set_list_of_bits(ODmask, finalListOfWords, finalListOfBits, 1);

	compute_neutrality_for_every_key_bit(neutrality_list, N, ID, ODmask, 7, 4, alg);
	get_pnb_list(pnb, &number_of_pnb, threshold, neutrality_list);
	for (int i = 0; i < number_of_pnb; i++)
		printf("%d, ", pnb[i]);

	printf("\nNumber of pnb: %d.\n", number_of_pnb);
	
	N = 1;
	N<<=32;
	varepsilon_a = compute_mean_bias_of_g_cuda(N, ID, ODmask,
			7, 4, pnb, number_of_pnb, alg);

	varepsilon_d = 0.026; //result from the paper aumasson 2008
	double time_complexity, data_complexity;
	double e = varepsilon_a * varepsilon_d;

	compute_complexity_of_the_attack(&data_complexity, &time_complexity, e, number_of_pnb);
	printf("varepsilon_a = %f \\varepsilon = %f, data_complexity %f, time_complexity %f.\n", varepsilon_a, e, data_complexity, time_complexity);
}

typedef struct {
	uint32_t inputRound;
	uint32_t inputMask[16];
	uint32_t outputRound;
	uint32_t outputMask[16];
	uint64_t parityCount;
	uint64_t totalCount;
	double bias;
} LINEAR_RELATION;

void print_latex_linear_relation(LINEAR_RELATION *L)
{
	double prob;

	printf("$$ \\begin{array}{cl}\n");
	if (fabs(L->bias) > 0)
	{
		printf("%f = (1+%f)/2 = &\\\\ \\Pr(", (1 + L->bias) / 2, L->bias);
	}
	else
	{
		prob = ((double)L->totalCount - L->parityCount) / L->totalCount;
		printf("%f = (1+%f)/2 = \\Pr(", prob, 2 * prob - 1);
	}
	for (int w = 0; w < 16; w++)
	{
		for (int b = 0; b < 32; b++)
		{
			if (get_bit_from_word_and_bit(L->inputMask, w, b))
				printf("x^{(%d)}_{%d,%d} \\oplus ", L->inputRound, w, b);
		}
	}
	printf(" = & ");

	int count = 0;
	for (int w = 0; w < 16; w++)
	{
		for (int b = 0; b < 32; b++)
		{
			if (get_bit_from_word_and_bit(L->outputMask, w, b))
			{
				printf("x^{(%d)}_{%d,%d} \\oplus ", L->outputRound, w, b);
				count++;
				if (count == 8)
				{
					count = 0;
					printf("\\\\ &");
				}
			}
		}
	}
	printf(") \\end{array} $$ \n\n");
}

uint8_t check_parity_of_linear_relation(LINEAR_RELATION *L, uint32_t inputState[16], uint32_t outputState[16])
{
	uint32_t aux[16], aux2[16];

	and_array(aux, inputState, L->inputMask, 16);
	and_array(aux2, outputState, L->outputMask, 16);
	xor_array(aux, aux, aux2, 16);

	return(xor_bits_of_state(aux));
}

int test_linear_relation(LINEAR_RELATION *L, uint32_t N, ALGORITHM alg)
{
	uint32_t state[16], finalState[16];
	L->totalCount = N;
	for (int i = 0; i < N; i++)
	{
		random_uint32_array(state, 16);
		memcpy(finalState, state, 16 * sizeof(uint32_t));

		alg.rounds(finalState, L->outputRound - L->inputRound, L->inputRound);

		L->parityCount += check_parity_of_linear_relation(L, state, finalState);
	}

	double prob;
	prob = ((double)L->totalCount - L->parityCount) / L->totalCount;

	if (fabs(prob - 0.5) > 4 * sqrt(0.25 / N))
	{
		print_latex_linear_relation(L);
		return 1;
	}
	return 0;
}

int test_linear_relation_cuda(LINEAR_RELATION *L, uint64_t N, ALGORITHM alg)
{
	L->bias = compute_linear_bias_cuda(L->inputRound, L->outputRound,
		alg.algType, L->inputMask, L->outputMask, N);

	double comp = (2 * (0.5 + 4 * sqrt(0.25 / N)) - 1);

	printf("%f\n", fabs(L->bias));
	if (fabs(L->bias) > comp)
	{
		print_latex_linear_relation(L);
		return 1;
	}
	return 0;
}

/*
Parameters:
w = 0 for set (0,4,8,12)
w = 1 for set (1,5,9,13)
w = 2 for set (2,6,10,14)
w = 3 for set (3,7,11,15)

letter 0,1,2,3 = A,B,C,D respectivelly
*/
#define LetterA 0
#define LetterB 1
#define LetterC 2
#define LetterD 3
#define typeLinear 0
#define typeNonLinear 1
#define typeSpecialC 2
#define typeSpecialA 3
void add_elements_to_linear_equation(LINEAR_RELATION *L, int w, int bit, int letter, unsigned char type, int round)
{
	int size;
	uint32_t abcd[4] = { 0,4,8,12 };

	uint32_t shift[4] = { 0 };
	if ((round % 2) == 0)
	{
		shift[1] = 1; shift[2] = 2; shift[3] = 3;
	}

	for (int i = 0; i < 4; i++)
		abcd[i] = abcd[i] + (shift[i] + w) % 4;

	uint32_t listOfWords[9] = { 0 }, listOfBits[9] = { 0 };
	uint32_t listOfWordsA[9] = { abcd[0], abcd[1],abcd[1], abcd[2], abcd[3], abcd[1], abcd[2], abcd[3], abcd[1] };
	uint32_t listOfBitsA[9] = { 0, 7,19,12,0,  18,11,31, 6 };

	uint32_t listOfWordsB[9] = { abcd[1],abcd[2],abcd[3],abcd[2],abcd[3] };
	uint32_t listOfBitsB[9] = { 19,12,0,0,31 };

	uint32_t listOfWordsC[9] = { abcd[3],abcd[2],abcd[3],abcd[0],abcd[0],abcd[3],abcd[3] };
	uint32_t listOfBitsC[9] = { 0,0,8,0,31,7,31 };

	uint32_t listOfWordsD[9] = { abcd[3],abcd[0],abcd[0],abcd[2],abcd[1],abcd[2],abcd[1] };
	uint32_t listOfBitsD[9] = { 24,16,0,0,7,31,6 };

	for (int j = 0; j < 9; j++)
	{
		switch (letter)
		{
		case 0:
			listOfWords[j] = listOfWordsA[j];
			listOfBits[j] = listOfBitsA[j];
			break;
		case 1:
			listOfWords[j] = listOfWordsB[j];
			listOfBits[j] = listOfBitsB[j];
			break;
		case 2:
			listOfWords[j] = listOfWordsC[j];
			listOfBits[j] = listOfBitsC[j];
			break;
		case 3:
			listOfWords[j] = listOfWordsD[j];
			listOfBits[j] = listOfBitsD[j];
			break;
		}
	}

	switch (letter)
	{
	case 0:
		if (type == 0)
			size = 5;
		else if (type == 3)
		{
			size = 9;
			listOfWords[8] = abcd[2];
			listOfBits[8] = (uint32_t)-1;
		}
		else
			size = 9;
		break;
	case 1:
		if (type == 0)
			size = 4;
		else
			size = 5;
		break;
	case 2:
		if (type == 0)
			size = 4;
		else if (type == 1)
			size = 7;
		else
			size = 6;
		break;
	case 3:
		if (type == 0)
			size = 5;
		else
			size = 7;
		break;
	}

	set_bit(L->inputMask, letter * 4 + (w + shift[letter]) % 4, bit);
	for (int j = 0; j < size; j++)
	{
		listOfWords[j] = listOfWords[j];
		listOfBits[j] = (listOfBits[j] + 32 + bit) % 32;
	}
	set_list_of_bits(L->outputMask, listOfWords, listOfBits, size);
}


void computational_result_02()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int w, round = 7;

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));
	L.inputRound = 6;
	L.outputRound = 7;

	printf("\nComputational Result 02:\n");
	w = 0;
	//x_{0,0}
	add_elements_to_linear_equation(&L, w, 0, LetterA, typeLinear, round);
	//x_{8,0}
	add_elements_to_linear_equation(&L, w, 0, LetterC, typeLinear, round);
	//x_{12,0}
	add_elements_to_linear_equation(&L, w, 0, 3, typeLinear, round);

	//x_{0,16}
	add_elements_to_linear_equation(&L, w, 16, LetterA, typeNonLinear, round);
	//x_{8,31}
	add_elements_to_linear_equation(&L, w, 31, LetterC, typeNonLinear, round);
	//x_{4,13}
	add_elements_to_linear_equation(&L, w, 13, LetterB, typeNonLinear, round);

	//x_{12,11} + x_{12,12}
	add_elements_to_linear_equation(&L, w, 12, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 11, LetterD, typeLinear, round);
	//x_{12,19} + x_{12,20}
	add_elements_to_linear_equation(&L, w, 19, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 20, LetterD, typeLinear, round);
	//x_{12,30} + x_{12,31}
	add_elements_to_linear_equation(&L, w, 30, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 31, LetterD, typeLinear, round);
	//x_{4,19} + x_{8,19} (B+C)
	add_elements_to_linear_equation(&L, w, 19, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 19, LetterC, typeSpecialC, round);

	//x_{4,7} + x_{8,7} + x_{8,8}
	add_elements_to_linear_equation(&L, w, 7, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 7, LetterC, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 8, LetterC, typeLinear, round);
	set_bit(L.outputMask, 12, 7);
	
	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 34; //use for faster test
	#else
	N <<= 42; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}


void computational_result_03()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int w, round = 7;

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));
	L.inputRound = 6;
	L.outputRound = 7;

	printf("\nComputational Result 03:\n");
	w = 1;
	//x_{1,0}
	add_elements_to_linear_equation(&L, w, 0, LetterA, typeLinear, round);
	//x_{1,6} + x_{1,7}
	add_elements_to_linear_equation(&L, w, 6, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 7, LetterA, typeLinear, round);
	//x_{1,11} + x_{1,12}
	add_elements_to_linear_equation(&L, w, 11, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 12, LetterA, typeLinear, round);
	//x_{1,22} + x_{1,23}
	add_elements_to_linear_equation(&L, w, 22, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 23, LetterA, typeLinear, round);
	//x_{9,0}
	add_elements_to_linear_equation(&L, w, 0, LetterC, typeLinear, round);
	//x_{5,7} + x_{9,6} (B+C)
	add_elements_to_linear_equation(&L, w, 7, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 6, LetterC, typeSpecialC, round);
	//x_{13,0}
	add_elements_to_linear_equation(&L, w, 0, LetterD, typeLinear, round);
	//x_{13,24}
	add_elements_to_linear_equation(&L, w, 24, LetterD, typeNonLinear, round);
	//x_{13,14} + x_{13,15}
	add_elements_to_linear_equation(&L, w, 14, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 15, LetterD, typeLinear, round);
	//x_{13,26} + x_{13,27}
	add_elements_to_linear_equation(&L, w, 26, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 27, LetterD, typeLinear, round);
	////x_{9,26}
	add_elements_to_linear_equation(&L, w, 26, LetterC, typeNonLinear, round);
	//x_{9,12}
	add_elements_to_linear_equation(&L, w, 12, LetterC, typeNonLinear, round);
	set_bit(L.outputMask, 13, 11);
	set_bit(L.outputMask, 13, 10);
	
	uint64_t N = 1;
	
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 34; //use for faster test
	#else
	N <<= 42; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}


void computational_result_04()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int w, round = 7;

	printf("\nComputational Result 04:\n");
	
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));
	L.inputRound = 6;
	L.outputRound = 7;

	w = 2;
	//x_{2,0}
	add_elements_to_linear_equation(&L, w, 0, LetterA, typeLinear, round);
	//x_{10,0}
	add_elements_to_linear_equation(&L, w, 0, LetterC, typeLinear, round);
	//x_{2,16}
	add_elements_to_linear_equation(&L, w, 16, LetterA, typeNonLinear, round);
	//x_{2,24}
	add_elements_to_linear_equation(&L, w, 24, LetterA, typeNonLinear, round);
	//x_{6,13} + x_{6,14}
	add_elements_to_linear_equation(&L, w, 13, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 14, LetterB, typeLinear, round);
	//x_{14,25} + x_{14,26}
	add_elements_to_linear_equation(&L, w, 25, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 26, LetterD, typeLinear, round);

	//x_{2,18} + x_{2,19} + x_{6,19}
	add_elements_to_linear_equation(&L, w, 18, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 19, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 19, LetterB, typeNonLinear, round);
	set_bit(L.outputMask, 14, 18);
	set_bit(L.outputMask, 14, 17);

	//x_{2,6} + x_{2,7} +x_{6,7} + x_{2,8} + x_{14,8}
	add_elements_to_linear_equation(&L, w, 6, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 7, LetterA, typeLinear, round);

	add_elements_to_linear_equation(&L, w, 7, LetterB, typeNonLinear, round);
	add_elements_to_linear_equation(&L, w, 8, LetterA, typeSpecialA, round);
	add_elements_to_linear_equation(&L, w, 8, LetterD, typeLinear, round);
	set_bit(L.outputMask, 14, 6);
	set_bit(L.outputMask, 14, 7);
	
	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 34; //use for faster test
	#else
	N <<= 42; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}


void computational_result_05()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int w, round = 7;

	printf("\nComputational Result 05:\n");
	
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));
	L.inputRound = 6;
	L.outputRound = 7;

	w = 3;
	//x_{7,14} + x_{7,15}
	add_elements_to_linear_equation(&L, w, 14, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 15, LetterB, typeLinear, round);
	//x_{7,26}
	add_elements_to_linear_equation(&L, w, 26, LetterB, typeNonLinear, round);
	//x_{15,24}
	add_elements_to_linear_equation(&L, w, 24, 3, typeNonLinear, round);

	//x_{11,6} + x_{11,7}
	add_elements_to_linear_equation(&L, w, 7, 2, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 6, 2, typeLinear, round);
	//x_{7,6} + x_{7,7}
	add_elements_to_linear_equation(&L, w, 7, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, w, 6, LetterB, typeLinear, round);

	uint64_t N = 1;	
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 32; //use for faster test
	#else
	N <<= 38; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}


void test_equation_24()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int round = 5;

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));
#if 1	
	L.inputRound = round - 1;
	L.outputRound = round;
	//x_{10,0}
	add_elements_to_linear_equation(&L, 10, 0, LetterC, typeLinear, round);
	//x_{5,7}
	add_elements_to_linear_equation(&L, 5, 7, LetterB, typeNonLinear, round);
#endif
	uint64_t N = 1;
	N <<= 28;
	test_linear_relation_cuda(&L, N, alg);
}

void computational_result_06()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	uint64_t N = 1;
	
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 32; //use for faster test
	#else
	N <<= 38; //used for paper result
    #endif
	
	printf("\nComputational Result 06:\n");

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	{
		//------------------------------
		//Test Lemma 22
		L.inputRound = 4;
		L.outputRound = 6;
		set_bit(L.inputMask, 5, 7);
		set_bit(L.inputMask, 10, 0);
		uint32_t listOfWords[48] = { 0,2,2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,5,7,7,7,7,8,8,8,9,9,9,10,10,10,13,13,13,14,14,14,14,14,14,14,14,14,14,15,15,15,15 };
		uint32_t listOfBits[48] = { 0,0,6,7,22,23,0,6,7,8,16,18,19,24,7,14,15,13,7,13,14,19,6,7,12,0,8,19,0,6,26,0,30,31,0,6,7,14,15,18,19,24,26,27,0,8,25,26 };
		set_list_of_bits(L.outputMask, listOfWords, listOfBits, 48);

		test_linear_relation_cuda(&L, N, alg);
		//------------------------------
	}
}



void computational_result_07()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int round = 7, group;

	printf("\nComputational Result 07:\n");
	
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	L.inputRound = round - 1;
	L.outputRound = round;

	group = 0;
	//x_{0,0}
	add_elements_to_linear_equation(&L, group, 0, LetterA, typeLinear, round);
	//x_{4,14}+x_{4,15}
	add_elements_to_linear_equation(&L, group, 14, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 15, LetterB, typeLinear, round);
	//x_{8,12}
	add_elements_to_linear_equation(&L, group, 12, LetterC, typeNonLinear, round);
	//x_{4,7}+x_{8,6}+x_{8,7}
	add_elements_to_linear_equation(&L, group, 7, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 7, LetterC, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 6, LetterC, typeLinear, round);
	set_bit(L.outputMask, 12, 5);

	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 32; //use for faster test
	#else
	N <<= 38; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}



void computational_result_08()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int round = 7, group;

	printf("\nComputational Result 08:\n");
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	L.inputRound = round - 1;
	L.outputRound = round;

	group = 1;
	//x_{9,0}
	add_elements_to_linear_equation(&L, group, 0, LetterC, typeLinear, round);
	//x_{13,0}
	add_elements_to_linear_equation(&L, group, 0, LetterD, typeLinear, round);
	//x_{13,30}+x_{13,31}
	add_elements_to_linear_equation(&L, group, 30, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 31, LetterD, typeLinear, round);
	//x_{5,13}
	add_elements_to_linear_equation(&L, group, 13, LetterB, typeNonLinear, round);
	//x_{9,8}
	add_elements_to_linear_equation(&L, group, 8, LetterC, typeNonLinear, round);
	//x_{9,19}
	add_elements_to_linear_equation(&L, group, 19, LetterC, typeNonLinear, round);

	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 32; //use for faster test
	#else
	N <<= 38; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}



void computational_result_09()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int round = 7, group;

	printf("\nComputational Result 09:\n");
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	L.inputRound = round - 1;
	L.outputRound = round;

	group = 2;
	//x_{2,0}
	add_elements_to_linear_equation(&L, group, 0, LetterA, typeLinear, round);
	//x_{10,0}
	add_elements_to_linear_equation(&L, group, 0, LetterC, typeLinear, round);
	//x_{14,0}
	add_elements_to_linear_equation(&L, group, 0, LetterD, typeLinear, round);
	
	//x_{2,6}+x_{2,7}+x_{2,6}+x_{10,6}+x_{14,6}+x_{14,7}
	add_elements_to_linear_equation(&L, group, 6, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 7, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 7, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 6, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 6, LetterC, typeSpecialC, round);
	set_bit(L.outputMask, 14, 6);

	//x_{2,22}+x_{2,23}
	add_elements_to_linear_equation(&L, group, 22, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 23, LetterA, typeLinear, round);

	//x_{10,26}
	add_elements_to_linear_equation(&L, group, 26, LetterC, typeNonLinear, round);

	//x_{14,14}+x_{14,15}
	add_elements_to_linear_equation(&L, group, 14, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 15, LetterD, typeLinear, round);
	//x_{14,18}+x_{14,19}
	add_elements_to_linear_equation(&L, group, 18, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 19, LetterD, typeLinear, round);
	//x_{14,26}+x_{14,27}
	add_elements_to_linear_equation(&L, group, 26, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 27, LetterD, typeLinear, round);

	//x_{14,24}
	add_elements_to_linear_equation(&L, group, 24, LetterD, typeNonLinear, round);


	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 34; //use for faster test
	#else
	N <<= 42; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}


void computational_result_10()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	int round = 7, group;

	printf("\nComputational Result 10:\n");
	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	L.inputRound = round - 1;
	L.outputRound = round;

	group = 3;
	//x_{3,0}
	add_elements_to_linear_equation(&L, group, 0, LetterA, typeLinear, round);
	//x_{15,0}
	add_elements_to_linear_equation(&L, group, 0, LetterD, typeLinear, round);

	//x_{3,6}+x_{3,7}+x_{7,7}
	add_elements_to_linear_equation(&L, group, 6, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 7, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 7, LetterB, typeLinear, round);
	set_bit(L.outputMask, 15, 5);

	//x_{3,8}+x_{15,8}
	add_elements_to_linear_equation(&L, group, 8, LetterA, typeSpecialA, round);
	add_elements_to_linear_equation(&L, group, 8, LetterD, typeLinear, round);

	//x_{7,13}+x_{7,14}
	add_elements_to_linear_equation(&L, group, 13, LetterB, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 14, LetterB, typeLinear, round);

	//x_{15,25}+x_{15,26}
	add_elements_to_linear_equation(&L, group, 25, LetterD, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 26, LetterD, typeLinear, round);

	//x_{3,16}
	add_elements_to_linear_equation(&L, group, 16, LetterA, typeNonLinear, round);
	//x_{3,24}
	add_elements_to_linear_equation(&L, group, 24, LetterA, typeNonLinear, round);

	//x_{3,18}+x_{3,19}+x_{7,19}
	add_elements_to_linear_equation(&L, group, 18, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 19, LetterA, typeLinear, round);
	add_elements_to_linear_equation(&L, group, 19, LetterB, typeLinear, round);
	set_bit(L.outputMask, 15, 17);

	uint64_t N = 1;
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 36; //use for faster test
	#else
	N <<= 42; //used for paper result
    #endif
	test_linear_relation_cuda(&L, N, alg);
}



void test_linear_relations_maitra_cuda()
{
	LINEAR_RELATION L;
	ALGORITHM alg;

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	{
		//------------------------------
		//Test Lemma 3 Equation A
		L.inputRound = 2;
		L.outputRound = 3;
		set_bit(L.inputMask, 0, 0);
		uint32_t listOfWords[5] = { 0, 4,4,8,12 };
		uint32_t listOfBits[5] = { 0,7,19,12,0 };
		set_list_of_bits(L.outputMask, listOfWords, listOfBits, 5);

		test_linear_relation_cuda(&L, (1 << 29), alg);
		//------------------------------
	}

	{
		//------------------------------
		//Test Lemma 10 Equation for X_{11}
		memset(&L, 0x00, sizeof(LINEAR_RELATION));
		L.inputRound = 3;
		L.outputRound = 5;
		set_bit(L.inputMask, 11, 0);
		uint32_t listOfWords[21] = { 12, 0,0,8,4,15,11,15,3,1,5,5, 9, 13,12,0, 0,8,4, 8,4 };
		uint32_t listOfBits[21] = { 24,16,0,0,7,0, 0, 8, 0,0,7,19,12,0, 0, 24,8,8,15,7,14 };
		set_list_of_bits(L.outputMask, listOfWords, listOfBits, 21);

		test_linear_relation_cuda(&L, (1 << 29), alg);
		//------------------------------
	}
}


void computational_result_01()
{
	LINEAR_RELATION L;
	ALGORITHM alg;
	uint64_t N = 1;
	
	#ifdef REDUCE_NUMBER_OF_ITERATIONS
	N <<= 32; //use for faster test
	#else
	N <<= 38; //used for paper result
    #endif
	
	printf("\nComputational Result 01:\n");

	define_alg(&alg, ALG_TYPE_CHACHA);
	memset(&L, 0x00, sizeof(LINEAR_RELATION));

	{
		//------------------------------
		//Test Lemma 20
		L.inputRound = 3;
		L.outputRound = 6;
		set_bit(L.inputMask, 3, 0);
		set_bit(L.inputMask, 4, 0);
		uint32_t listOfWords[59] = { 0,0, 1,1,1,1, 1, 1, 1, 2,2,2,2,2, 2, 2, 2, 4,4, 4, 5,6,6, 6, 6, 7,7, 7, 7, 7, 8,8,8,8, 8, 9,9, 9, 9, 10,11,11,12,12,12,12,12,12,12,13,13,13,13,13,13,14,14,14,15 };
		uint32_t listOfBits[59] = { 0,16,0,6,7,11,12,22,23,0,6,7,8,16,18,19,24,7,19,13,7,7,13,14,19,7,14,15,26,6,0,7,8,19,31,0,12,26,6,0, 6, 7, 0, 11,12,19,20,30,31,14,15,24,26,27,0,8, 25,26,24 };
		set_list_of_bits(L.outputMask, listOfWords, listOfBits, 59);

		test_linear_relation_cuda(&L, N, alg);
		//------------------------------
	}
}

void print_latex_eq_and_bias(int rounds, double bias, uint32_t ID[16], uint32_t ODmask[16])
{
	printf("Bias = %f, for ", bias);
	printf("$$\\Delta = (");
	for (int w = 0; w < 16; w++)
	{
		for (int b = 0; b< 32; b++)
		{
			if (get_bit_from_word_and_bit(ODmask, w, b))
				printf("\\Delta X^{(%d)}_{%d,%d} \\oplus ", rounds, w, b);
		}
	}
	printf(" | ");
	for (int w = 0; w < 16; w++)
	{
		for (int b = 0; b< 32; b++)
		{
			if (get_bit_from_word_and_bit(ID, w, b))
				printf("\\Delta X^{(%d)}_{%d,%d}, ", 0, w, b);
		}
	}
	printf(")$$\n");
}



/*Test to find the bias used by Maitra, the differential bias should be 
approximately 0.0272 as in page 16 of the paper.*/
void search_until_find_significant_bias_maitra()
{
	uint32_t ID[16] = { 0 };
	uint32_t ODmask[16] = { 0 };
	uint64_t N = 1;
	uint32_t listOfWords[5] = { 0 };
	uint32_t listOfBits[5] = { 0 };

	//Init odmask
	memset(ODmask, 0x00, sizeof(uint32_t) * 16);
	listOfWords[0] =11; listOfBits[0] =0;
	set_list_of_bits(ODmask, listOfWords, listOfBits, 1);

	printf("w=%d\n", listOfWords[0]);
	memset(ID, 0x00, sizeof(uint32_t) * 16);
	ID[13] = 0x00002000;

	int level = 38, rounds = 3, halfround = 0;
	N = 1;
	N <<= level;
	double comp;
	int flagFirst = 1;
	double bias = 0;
	while (1)
	{
		bias += compute_differential_bias_multiple_devices(rounds, 
			ALG_TYPE_CHACHA, ID, ODmask, N, halfround, 0);
	
		if (flagFirst)
			flagFirst = 0;
		else
			bias /= 2;

		printf("Level %d\n", level);
		N = 1;
		N <<= level;
		comp = (2 * (0.5 + 4 * sqrt(0.25 / N)) - 1);

		level++;
		if ((fabs(bias) < comp))
			continue;

		//Check again to make sure
		double newbias = compute_differential_bias_multiple_devices(rounds, 
			ALG_TYPE_CHACHA, ID, ODmask, N, halfround, 0);
		if ((fabs(newbias) < comp))
			continue;

		printf("Bias = (%.15f e %.15f)\n", bias, newbias);
		return;
	}
}



void search_until_find_significant_bias_for_chacha3andHalf_with_IDw14b6_starting_from_second_round(int od_word, int od_bit)
{
	uint32_t ID[16] = { 0 };
	uint32_t ODmask[16] = { 0 };
	uint64_t N = 1;
	uint32_t listOfWords[5] = { 0 };
	uint32_t listOfBits[5] = { 0 };

	//Init odmask
	memset(ODmask, 0x00, sizeof(uint32_t) * 16);
	printf("Testing differential bias for OD = X_{%d,%d}\n", od_word,od_bit);
	listOfWords[0] =od_word; listOfBits[0] =od_bit;
	set_list_of_bits(ODmask, listOfWords, listOfBits, 1);

	printf("w=%d\n", listOfWords[0]);
	memset(ID, 0x00, sizeof(uint32_t) * 16);
	ID[2] = 0x00000004;
	ID[6] = 0x20020220;
	ID[10] = 0x40400400;
	ID[14] = 0x40000400;

	int level = 38, rounds = 3, halfround = 1;
	N = 1;
	N <<= level;
	double comp;
	int flagFirst = 1;
	double bias = 0;
	while (1)
	{
		bias += compute_differential_bias_multiple_devices(rounds - 1, 
			ALG_TYPE_CHACHA, ID, ODmask, N, halfround, 1);
		if (flagFirst)
			flagFirst = 0;
		else
			bias /= 2;

		printf("Level %d\n", level);
		N = 1;
		N <<= level;
		comp = (2 * (0.5 + 4 * sqrt(0.25 / N)) - 1);

		level++;
		if ((fabs(bias) < comp))
			continue;

		//Check again to make sure
		double newbias = compute_differential_bias_multiple_devices(rounds - 1, 
			ALG_TYPE_CHACHA, ID, ODmask, N, halfround, 1);
		if ((fabs(newbias) < comp))
			continue;

		printf("Bias = (%.15f e %.15f)\n", bias, newbias);
		return;
	}
}

void test_new_pnb_attacks_chacha_7_back_3_bias_in_b13_3andhalf()
{
	uint32_t ODmask[16] = { 0 };
	uint32_t ID[16] = { 0 };
	uint64_t N;
	ALGORITHM alg;
	double neutrality_list[256];
	uint32_t pnb[256], number_of_pnb;
	double threshold = 0.8;

	define_alg(&alg, ALG_TYPE_CHACHA);

	//Go 7, back 3
	memset(ODmask, 0x00, sizeof(uint32_t) * 16);
	N = 1;
	N <<= 14;
	ID[14] = (1 << 6);
	uint32_t listOfWords[2] = { 13,2 };
	uint32_t listOfBits[2] = { 8,0 };
	set_list_of_bits(ODmask, listOfWords, listOfBits, 2);

	N = 1;
	N <<= 22;
	compute_neutrality_for_every_key_bit(neutrality_list, N, ID, ODmask, 7, 3, alg);
	for (int i = 0; i < 256; i++)
		printf("%f, ", neutrality_list[i]);
		
	double varepsilon_a = 0;
	double varepsilon_d = 0.000003032;
	N = 1;
	N <<= 38;

	double comp = (2 * (0.5 + 4 * sqrt(0.25 / N)) - 1);

	threshold = 0.35;
	get_pnb_list(pnb, &number_of_pnb, threshold, neutrality_list);
	printf("Number of pnb for gamma = %f: %d.\n", threshold, number_of_pnb);
	for (int i = 0; i < number_of_pnb; i++)
		printf("%d, ", pnb[i]);
	printf("\n\n");
	varepsilon_a = compute_mean_bias_of_g_cuda(N, ID, ODmask,
		7, 3, pnb, number_of_pnb, alg);
	double time_complexity, data_complexity;
	double e = varepsilon_a * varepsilon_d;

	if ((fabs(varepsilon_a) < comp))
		printf("gamma = %f, not significant...\n", threshold);
	else
	{
		compute_complexity_of_the_attack(&data_complexity, &time_complexity, e, number_of_pnb);
		printf("varepsilon_a = %f \\varepsilon = %f, data_complexity %f, time_complexity %f.\n", varepsilon_a, e, data_complexity, time_complexity);
	}
}


int main()
{
	srand(time(NULL));
	if (testChachaOnGPU())
	{
		printf("Error, aborting\n");
		return(1);
	}
	
	//Section 4.2 (Linear approximations)
	//First testing some linear approximations presented in Maitra 2017 to check the correctness of the implementation.
	test_linear_relations_maitra_cuda();
	computational_result_01();
	computational_result_02();
	computational_result_03();
	computational_result_04();
	computational_result_05();
	test_equation_24();
	computational_result_06();
	computational_result_07();
	computational_result_08();
	computational_result_09();
	computational_result_10();
	
	//Section 5.1 (Differential biases)
	//First test the bias available in page 16 of Maitra 2017 to check the correctness of the implementation. 
	//The bias should be close to 0.0272.
        search_until_find_significant_bias_maitra();  
	search_until_find_significant_bias_for_chacha3andHalf_with_IDw14b6_starting_from_second_round(0,0);
	search_until_find_significant_bias_for_chacha3andHalf_with_IDw14b6_starting_from_second_round(13,0);
		
	//Section 5.3 (Probabilistic Neutral Bits (PNB))
	/*
	First test results from aumasson 2008 to check the correctness of the implementation.	
	In the paper we have:

	neutral bits:
	{3, 6, 15, 16, 31, 35, 67, 68, 71, 91, 92, 93, 94, 95, 96, 97, 98, 99,
	100, 103, 104, 127, 136, 191, 223, 224, 225, 248, 249, 250, 251, 252, 253, 254,
	255}

	\varepsilon_a = 0.023

	Time complexity = 248
	Data complexity = 27
	*/
	test_aumasson_pnb_attack_chacha_7();
	test_new_pnb_attacks_chacha_7_back_3_bias_in_b13_3andhalf();

	return 0;
}
