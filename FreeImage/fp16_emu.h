/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

// Conversion from/to 16-bit floating point (half-precision).


#if !defined(_FP16_EMU_H_)
#define _FP16_EMU_H_

#include "fp16_dev.h"
#include <cuda_fp16.h>

#define HLF_EPSILON 4.887581E-04
#define HLF_MIN     6.103516E-05
#define HLF_MAX     6.550400E+04

half1 cpu_float2half_rn(float f)
{
	half1 ret;

	unsigned x = *((int*)(void*)(&f));
	unsigned u = (x & 0x7fffffff), remainder, shift, lsb, lsb_s1, lsb_m1;
	unsigned sign, exponent, mantissa;

	// Get rid of +NaN/-NaN case first.
	if (u > 0x7f800000) {
		ret.x = 0x7fffU;
		return ret;
	}

	sign = ((x >> 16) & 0x8000);

	// Get rid of +Inf/-Inf, +0/-0.
	if (u > 0x477fefff) {
		ret.x = sign | 0x7c00U;
		return ret;
	}
	if (u < 0x33000001) {
		ret.x = (sign | 0x0000);
		return ret;
	}

	exponent = ((u >> 23) & 0xff);
	mantissa = (u & 0x7fffff);

	if (exponent > 0x70) {
		shift = 13;
		exponent -= 0x70;
	}
	else {
		shift = 0x7e - exponent;
		exponent = 0;
		mantissa |= 0x800000;
	}
	lsb = (1 << shift);
	lsb_s1 = (lsb >> 1);
	lsb_m1 = (lsb - 1);

	// Round to nearest even.
	remainder = (mantissa & lsb_m1);
	mantissa >>= shift;
	if (remainder > lsb_s1 || (remainder == lsb_s1 && (mantissa & 0x1))) {
		++mantissa;
		if (!(mantissa & 0x3ff)) {
			++exponent;
			mantissa = 0;
		}
	}

	ret.x = (sign | (exponent << 10) | mantissa);

	return ret;
}

float cpu_half2float(half1 h)
{
	unsigned sign = ((h.x >> 15) & 1);
	unsigned exponent = ((h.x >> 10) & 0x1f);
	unsigned mantissa = ((h.x & 0x3ff) << 13);

	if (exponent == 0x1f) {  /* NaN or Inf */
		mantissa = (mantissa ? (sign = 0, 0x7fffff) : 0);
		exponent = 0xff;
	}
	else if (!exponent) {  /* Denorm or Zero */
		if (mantissa) {
			unsigned int msb;
			exponent = 0x71;
			do {
				msb = (mantissa & 0x400000);
				mantissa <<= 1;  /* normalize */
				--exponent;
			} while (!msb);
			mantissa &= 0x7fffff;  /* 1.mantissa is implicit */
		}
	}
	else {
		exponent += 0x70;
	}

	int temp = ((sign << 31) | (exponent << 23) | mantissa);

	return *((float*)((void*)&temp));
}

static __inline__ __device__ __host__ half1 habs(half1 h)
{
    h.x &= 0x7fffU;
    return h;
}

static __inline__ __device__ __host__ half1 hneg(half1 h)
{
    h.x ^= 0x8000U;
    return h;
}

static __inline__ __device__ __host__ int ishnan(half1 h)
{
    // When input is NaN, exponent is all ones and mantissa is non-zero.
    return (h.x & 0x7c00U) == 0x7c00U && (h.x & 0x03ffU) != 0;
}

static __inline__ __device__ __host__ int ishinf(half1 h)
{
    // When input is +/- inf, exponent is all ones and mantissa is zero.
    return (h.x & 0x7c00U) == 0x7c00U && (h.x & 0x03ffU) == 0;
}

static __inline__ __device__ __host__ int ishequ(half1 x, half1 y)
{
    return ishnan(x) == 0 && ishnan(y) == 0 && x.x == y.x;
}

// Returns 0.0000 in FP16 binary form
static __inline__ __device__ __host__ half1 hzero()
{
    half1 ret;
    ret.x = 0x0000U;
    return ret;
}

// Returns 1.0000 in FP16 binary form
static __inline__ __device__ __host__ half1 hone()
{
    half1 ret;
    ret.x = 0x3c00U;
    return ret;
}

// Returns quiet NaN, the most significant fraction bit #9 is set
static __inline__ __device__ __host__ half1 hnan()
{
    half1 ret;
    ret.x = 0x7e00U;
    return ret;
}

// Largest positive FP16 value, corresponds to 6.5504e+04
static __inline__ __device__ __host__ half1 hmax()
{
    half1 ret;
    // Exponent all ones except LSB (0x1e), mantissa is all ones (0x3ff)
    ret.x = 0x7bffU;
    return ret;
}

// Smallest positive (normalized) FP16 value, corresponds to 6.1035e-05
static __inline__ __device__ __host__ half1 hmin()
{
    half1 ret;
    // Exponent is 0x01 (5 bits), mantissa is all zeros (10 bits)
    ret.x = 0x0400U;
    return ret;
}

#endif  // _FP16_EMU_H_

