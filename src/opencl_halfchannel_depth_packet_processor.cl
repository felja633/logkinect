
/*******************************************************************************
 * Process pixel stage 1
 ******************************************************************************/

#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#else
#error No FP16 support
#endif

#define INV_SQRT_2 0.70710678118f
#define ALPHA 0.0f
#define DEPTH_SCALE 1.0f

float decodePixelMeasurement(global const ushort *data, global const short *lut11to16, const uint sub, const uint x, const uint y)
{
    uint row_idx = (424 * sub + (y < 212 ? y + 212 : 423 - y)) * 352;
    uint idx = (((x >> 2) + ((x << 7) & BFI_BITMASK)) * 11) & (uint)0xffffffff;

    uint col_idx = idx >> 4;
    uint upper_bytes = idx & 15;
    uint lower_bytes = 16 - upper_bytes;

    uint data_idx0 = row_idx + col_idx;
    uint data_idx1 = row_idx + col_idx + 1;

    return (float)lut11to16[(x < 1 || 510 < x || col_idx > 352) ? 0 : ((data[data_idx0] >> upper_bytes) | (data[data_idx1] << lower_bytes)) & 2047];
}

float2 processMeasurementTriple(const float ab_multiplier_per_frq, const float p0, const float3 v, int *invalid)
{
    float3 p0vec = (float3)(p0 + PHASE_IN_RAD0, p0 + PHASE_IN_RAD1, p0 + PHASE_IN_RAD2);
    float3 p0cos = cos(p0vec);
    float3 p0sin = sin(-p0vec);

    *invalid = *invalid && any(isequal(v, (float3)(32767.0f)));

    return (float2)(dot(v, p0cos), dot(v, p0sin)) * ab_multiplier_per_frq;
}


void kernel processPixelStage1(global const short *lut11to16, global const float *z_table, global const float3 *p0_table, global const ushort *data,
                               global float3 *a_out, global float3 *b_out, global float3 *n_out, global float* ir_out, global float3 *c_out)
{
    const uint i = get_global_id(0);

    const uint x = i % 512;
    const uint y = i / 512;

    const uint y_in = (423 - y);

    const float zmultiplier = z_table[i];
    int valid = (int)(0.0f < zmultiplier);
    int saturatedX = valid;
    int saturatedY = valid;
    int saturatedZ = valid;
    int3 invalid_pixel = (int3)((int)(!valid));
    const float3 p0 = p0_table[i];

    const float3 v0 = (float3)(decodePixelMeasurement(data, lut11to16, 0, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 1, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 2, x, y_in));
    const float2 ab0 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ0, p0.x, v0, &saturatedX);

    const float3 v1 = (float3)(decodePixelMeasurement(data, lut11to16, 3, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 4, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 5, x, y_in));
    const float2 ab1 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ1, p0.y, v1, &saturatedY);

    const float3 v2 = (float3)(decodePixelMeasurement(data, lut11to16, 6, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 7, x, y_in),
                               decodePixelMeasurement(data, lut11to16, 8, x, y_in));
    const float2 ab2 = processMeasurementTriple(AB_MULTIPLIER_PER_FRQ2, p0.z, v2, &saturatedZ);

    float3 a = select((float3)(ab0.x, ab1.x, ab2.x), (float3)(0.0f), invalid_pixel);
    float3 b = select((float3)(ab0.y, ab1.y, ab2.y), (float3)(0.0f), invalid_pixel);

    float3 n = sqrt(a * a + b * b);

    int3 saturated = (int3)(saturatedX, saturatedY, saturatedZ);
    a = select(a, (float3)(0.0f), saturated);
    b = select(b, (float3)(0.0f), saturated);
		float3 c = (float3)(v0.x+v0.y+v0.z,v1.x+v1.y+v1.z,v2.x+v2.y+v2.z);
		//uint i_out = 512*y+(511-x);
    a_out[i] = a;
    b_out[i] = b;
    n_out[i] = n;
		c_out[i] = fabs(c)*0.333333333f;
		ir_out[i] = min(dot(select(n, (float3)(65535.0f), saturated), (float3)(0.333333333f  * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER)), 65535.0f);
}

/*******************************************************************************
 * Filter pixel stage 1
 ******************************************************************************/
void kernel filterPixelStage1(global const float3 *a, global const float3 *b, global const float3 *n,
                              global float3 *a_out, global float3 *b_out, global uchar *max_edge_test, global const float3 *o, global float *o_out)
{
    const uint i = get_global_id(0);

    const uint x = i % 512;
    const uint y = i / 512;

    const float3 self_a = a[i];
    const float3 self_b = b[i];
		const float3 self_o = o[i];

    const float gaussian[9] = {GAUSSIAN_KERNEL_0, GAUSSIAN_KERNEL_1, GAUSSIAN_KERNEL_2, GAUSSIAN_KERNEL_3, GAUSSIAN_KERNEL_4, GAUSSIAN_KERNEL_5, GAUSSIAN_KERNEL_6, GAUSSIAN_KERNEL_7, GAUSSIAN_KERNEL_8};

    if(x < 1 || y < 1 || x > 510 || y > 422)
    {
        a_out[i] = self_a;
        b_out[i] = self_b;
        max_edge_test[i] = 1;
			  o_out[i] = self_o.x;
				o_out[i+512*424] = self_o.y;
				o_out[i+512*424*2] = self_o.z;
    }
    else
    {
        float3 threshold = (float3)(JOINT_BILATERAL_THRESHOLD);
        float3 joint_bilateral_exp = (float3)(JOINT_BILATERAL_EXP);

        const float3 self_norm = n[i];
        const float3 self_normalized_a = self_a / self_norm;
        const float3 self_normalized_b = self_b / self_norm;
				const float3 self_normalized_o = self_o / self_norm;

        float3 weight_acc = (float3)(0.0f);
        float3 weighted_a_acc = (float3)(0.0f);
        float3 weighted_b_acc = (float3)(0.0f);
				float3 weighted_o_acc = (float3)(0.0f);
        float3 dist_acc = (float3)(0.0f);

        const int3 c0 = isless(self_norm * self_norm, threshold);

        threshold = select(threshold, (float3)(0.0f), c0);
        joint_bilateral_exp = select(joint_bilateral_exp, (float3)(0.0f), c0);

        for(int yi = -1, j = 0; yi < 2; ++yi)
        {
            uint i_other = (y + yi) * 512 + x - 1;

            for(int xi = -1; xi < 2; ++xi, ++j, ++i_other)
            {
                const float3 other_a = a[i_other];
                const float3 other_b = b[i_other];
                const float3 other_norm = n[i_other];
								const float3 other_o = o[i_other];
                const float3 other_normalized_a = other_a / other_norm;
                const float3 other_normalized_b = other_b / other_norm;
								const float3 other_normalized_o = other_o / other_norm;

                const int3 c1 = isless(other_norm * other_norm, threshold);

                const float3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
                const float3 weight = select(gaussian[j] * exp(-1.442695f * joint_bilateral_exp * dist), (float3)(0.0f), c1);

                weighted_a_acc += weight * other_a;
                weighted_b_acc += weight * other_b;
								weighted_o_acc += weight * other_o;
                weight_acc += weight;
                dist_acc += select(dist, (float3)(0.0f), c1);
            }
        }

        const int3 c2 = isless((float3)(0.0f), weight_acc.xyz);
        a_out[i] = select((float3)(0.0f), weighted_a_acc / weight_acc, c2);
        b_out[i] = select((float3)(0.0f), weighted_b_acc / weight_acc, c2);
				float3 o_filt = select((float3)(0.0f), weighted_o_acc / weight_acc, c2);
			  o_out[i] = o_filt.x;
				o_out[i+512*424] = o_filt.y;
				o_out[i+512*424*2] = o_filt.z;
				//o_out[i] = select((float3)(0.0f), weighted_o_acc / weight_acc, c2);

        max_edge_test[i] = all(isless(dist_acc, (float3)(JOINT_BILATERAL_MAX_EDGE)));
    }
}
void kernel undistort(global const float *depth, global const float *camera_intrinsics, global float *depth_out)
{
	const uint i = get_global_id(0);

  const float x = (float)(i % 512);
  const float y = (float)(i / 512);
	//const uint x = (i % 512);
  //const uint y = (i / 512);
	//const float raw_depth = depth[i];
	//depth_out[i] = raw_depth;

	float fx = camera_intrinsics[0];
	float fy = camera_intrinsics[1];
	float cx = camera_intrinsics[2];
	float cy = camera_intrinsics[3];
	
	float k1 = camera_intrinsics[4];
	float k2 = camera_intrinsics[5];
	float k3 = camera_intrinsics[6];

	float fx_inv = 1/fx; 
	float fy_inv = 1/fy;

	float cx_inv = -cx/fx;
	float cy_inv = -cy/fy;

	float x_dist = fx_inv*x+cx_inv;
	float y_dist = fy_inv*y+cy_inv;
	//x_dist = A*f(A^-1*x,k)
	float r = sqrt(x_dist*x_dist+y_dist*y_dist);
	float dist = (1.0+k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r);
	x_dist = x_dist*dist;
	y_dist = y_dist*dist;

	x_dist = fx*x_dist+cx;
	y_dist = fy*y_dist+cy;

	//int x_in_dist = (int)floor(x_dist);	
	//int y_in_dist = (int)floor(y_dist);

	//float x_e = x_dist - (float)x_in_dist;
	//float y_e = y_dist - (float)y_in_dist;

	int x_in_dist = (int)round(x_dist);	
	int y_in_dist = (int)round(y_dist);

	if(x_in_dist<0 || x_in_dist>512 || y_in_dist<0 || y_in_dist>424)	
	{
		depth_out[i] = 0.0;
		return;
	}

  /*bilinear interpolation
	int idx_00 = x_in_dist+512*y_in_dist;
	int idx_10 = x_in_dist+1+512*y_in_dist;
	int idx_01 = x_in_dist+512*(y_in_dist+1);
	int idx_11 = x_in_dist+1+512*(y_in_dist+1);

	float f00 = depth[idx_00]*(1-x_e)*(1-y_e);
	float f01 = 0.0;
	float f10 = 0.0;
	float f11 = 0.0;


	if(x_in_dist<512-1)
	{
		f10	= depth[idx_10]*x_e*(1-y_e);
	}
	
	if(y_in_dist<424-1)
	{
		f01	= depth[idx_01]*(1-x_e)*y_e;
	}

	if(y_in_dist<424-1 && x_in_dist<512-1)
	{
		f11 = depth[idx_11]*x_e*y_e;
	}

	depth_out[i] = (f00+f10+f01+f11);*/

	depth_out[i] = depth[512*y_in_dist+x_in_dist];
}


/*******************************************************************************
 * Process pixel stage 2
 ******************************************************************************/
float constant k_list[30] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
float constant n_list[30] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 9.0f, 9.0f};
float constant m_list[30] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 7.0f, 7.0f, 8.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 13.0f, 13.0f, 14.0f};

void calcErr(const float k, const float n, const float m, const float t0, const float t1, const float t2, float* err1, float* err2, float* err3)
{
    *err1 = 3.0f*n-15.0f*k-(t1-t0);
    *err2 = 3.0f*n-2.0f*m-(t2-t0);
    *err3 = 15.0f*k-2.0f*m-(t2-t1);
}

void phaseUnWrapper(float t0, float t1,float t2, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
  float err;
  float err1,err2,err3;
  //unsigned int ind_count = 1;
	//float k,nc,nf;
	//float w1 = 0.7007f;
	//float w2 = 366.2946f;
	//float w3 = 0.7016f;
	
	float w1 = 1.0f;
	float w2 = 18.0f;
	float w3 = 1.0218f;
	
	//float w1 = 1.0f;
	//float w2 = 1.0f;
	//float w3 = 1.0f;
	//float w1 = 1.0f;
	//float w2 = 3.6f;
	//float w3 = 1.0588f;

	float err_min=100000.0f;
	float err_min_second = 200000.0f;
	unsigned int ind_min, ind_second;

	float k,n,m;
	
	for(int i=0; i<30; i++)
	{
		m = m_list[i];
		n = n_list[i];
		k = k_list[i];
		calcErr(k,n,m,t0,t1,t2,&err1,&err2,&err3);
		err = w1*err1*err1+w2*err2*err2+w3*err3*err3;
		if(err<err_min)
		{
			err_min_second = err_min;
			ind_second = ind_min;
			err_min = err;
			ind_min = i;

		}
		else if(err<err_min_second)
		{
    	err_min_second = err;
			ind_second = i;
		}
		
	}

	//decode ind_min
	float mvals = m_list[ind_min];
	float nvals = n_list[ind_min];
	float kvals = k_list[ind_min];

	float phi2_out = (t2/2.0f+mvals);
	float phi1_out = (t1/15.0f+kvals);
	float phi0_out = (t0/3.0f+nvals);


	//float phi2_out = 1.23853211009f*(t2/2.0f+mvals);
	//float phi1_out = 0.16513761467f*(t1/15.0f+kvals);
	//float phi0_out = 0.82568807339f*(t0/3.0f+nvals);

/*
	float phi2_out = (t2+2*mvals)*0.61550152f;
	float phi1_out = (t1+15*kvals)*0.01094225f;
	float phi0_out = (t0+3*nvals)*0.27355623f;*/

	*err_w1 = err_min;

	*phase_first = (phi2_out+phi1_out+phi0_out)/3.0f;


	mvals = m_list[ind_second];
	nvals = n_list[ind_second];
	kvals = k_list[ind_second];


	phi2_out = (t2/2.0f+mvals);
	phi1_out = (t1/15.0f+kvals);
	phi0_out = (t0/3.0f+nvals);

	//phi2_out = 1.23853211009f*(t2/2.0f+mvals);
	//phi1_out = 0.16513761467f*(t1/15.0f+kvals);
	//phi0_out = 0.82568807339f*(t0/3.0f+nvals);

	/*phi2_out = (t2+2*mvals)*0.61550152f;
	phi1_out = (t1+15*kvals)*0.01094225f;
	phi0_out = (t0+3*nvals)*0.27355623f;*/

	*err_w2 = err_min_second;
	*phase_second = (phi2_out+phi1_out+phi0_out)/3.0f;	

}


/********************************************
 *********** Channel decoding  **************
 ********************************************/

//float constant p[8] = { 3.67443f, 1.59069183f, 3.42366477f, 1.51662235f, 3.06120769f, 2.57150253f, -65.71612711f, 9.44306263f};
//Här är resultatet med facit=1 om hyp1 eller hyp2 stämmer:
//float constant p[8] = {4.60262016f,   1.17154817f, 4.96314122f, 1.08643378f, 4.50801187f, 2.47211491f, -70.1514416f ,  12.35428024f};

//Här är resultatet med facit=1 endast om hyp1 stämmer:
//float constant p[8] = {5.52475005f, 1.76809678f, -95.50978931f, 31.13891508f, 5.88386689f,   3.7361783f , -68.79978416f,  10.35229698f

//float constant p[8]= {  5.44758639f,   1.78835189f, -95.84443187f,  31.09057199f, 5.95266885f, 3.73647938f, -69.33440188f,  10.4947256f };

//Dessa är från hårdare tröskling mot GT.
//Jag bytte till en binär träningsbild, och fick minimalt annorlunda parametrar:
//float constant p[8]={  5.90394834f, 1.49391944f, -67.17313263,  26.79363506f, 7.1874722f, 3.11018363f, -56.83466455f, 7.91574918f};


//new training 241115 ms25d1s
//float constant p[8] = {5.55257635f,   1.47361095f,   5.3596743f,   0.65984877f,  6.26740667f,   2.71722929f, -59.19955872f,   6.27931053f};

// only amplitude, no consistence measurement
//float constant p[6]={  5.83099917f,   1.46120597f,  5.45949417f,    0.7127183f , 7.13129323f, 4.005528f };

//ms25d1c
//float constant p[8] = {5.17762656f, 0.75104709f, 5.36533495f, 0.62164875f, 5.41381391f, 1.04309533f, -62.90804264f, 3.4542968f};

//Träning med a1/(|o1|+1e-9),a2/(|o2|+1e-9),a3/(|o3|+1e-9),pconf på nya datasetet:
//float constant p[6]={0.643822505f, 1.97701215f, 1.20565010f, 1.38159262f, -1.57876546f, 2.50247235f};


//Träning med a1,a2,a3,pconf på nya datasetet ms25d1f2:
//float constant p[8]={1.93252395f, 2.87975271f, 4.07562946f, 2.15845718f, -0.56857385f, 3.51395478f, -69.5114262f, 5.29692742f}; //#session 4

//Smoothade a och o-bilder:
//float constant p[8]={2.05146954f, 1.0586995f, 2.13657789f, 0.97358424f, 1.83101693f, 1.11670487f, -25.69552801f, 6.18873822f}; //# After smoothing 

//varianten utan pcons: ms25d1r3
//float constant p[6]={2.06761669f, 1.11863764f, 2.15462793f, 1.01355314f, 1.83402425f, 1.18000069f};

//Tränat på session 5 med enbart a1,a2,a3:'ms25d1f3'
float constant p[8]={ 4.49432993f,  1.36533393f,  4.42313275f,  1.15035961f,  4.11977512f, 1.83015617f, 0, 0};

//Tränat på session 5 med a1,a2,a3,pcons: 'ms25d1f4'
//float constant p[8]={ 4.48765613f, 1.3545396f, 4.42689315f, 1.14757834f, 4.1070274f, 1.80847054f, -26.03667419f, 4.36424484f};

#define VEC16_TO_ARRAY(vec,arr) \
				arr[0] = vec.s0;					\
				arr[1] = vec.s1;					\
				arr[2] = vec.s2;				\
				arr[3] = vec.s3;					\
				arr[4] = vec.s4;					\
				arr[5] = vec.s5;					\
			  arr[6] = vec.s6;  				\
				arr[7] = vec.s7;					\
				arr[8] = vec.s8;					\
				arr[9] = vec.s9;					\
				arr[10] = vec.sa;					\
				arr[11] = vec.sb;					\
				arr[12] = vec.sc;					\
				arr[13] = vec.sd;					\
				arr[14] = vec.se;					\
				arr[15] = vec.sf;					


#define DOT_16(vec1,vec2) vec1.s0*vec2.s0+vec1.s1*vec2.s1+vec1.s2*vec2.s2+vec1.s3*vec2.s3+vec1.s4*vec2.s4+vec1.s5*vec2.s5+vec1.s6*vec2.s6+vec1.s7*vec2.s7+vec1.s8*vec2.s8+vec1.s9*vec2.s9+vec1.sa*vec2.sa+vec1.sb*vec2.sb+vec1.sc*vec2.sc+vec1.sd*vec2.sd+vec1.se*vec2.se+vec1.sf*vec2.sf
		
#define DOT_16_2(vec1,vec2) dot(vec1.s0123,vec2.s0123)+dot(vec1.s4567,vec2.s4567)+dot(vec1.s89ab,vec2.s89ab)+dot(vec1.scdef,vec2.scdef)

float phaseConfidenceA(float3 a)
{
	float c =  max(fabs(a.x-a.y),max(fabs(a.x-a.z),fabs(a.y-a.z)));
	float p1 = 1.0f/(1.0f+exp(-(a.x-p[0])/p[1])); // p(t|a1)
	float p2 = 1.0f/(1.0f+exp(-(a.y-p[2])/p[3])); // p(t|a2)
	float p3 = 1.0f/(1.0f+exp(-(a.z-p[4])/p[5])); // p(t|a3)
	//float p4 = 1.0f/(1.0f+exp(-(-c-p[6])/p[7])); // p(t|c) 

	return p1*p2*p3; //ptot = p1*p2*p3*p4 # p(t|phase) 
}

float phaseConfidenceAoverO(float3 a, float3 o)
{
	float3 q = a/(o+(float3)(0.000000001));
	float c =  max(fabs(a.x-a.y),max(fabs(a.x-a.z),fabs(a.y-a.z)));
	float p1 = 1.0f/(1.0f+exp(-(q.x-p[0])/p[1])); // p(t|a1)
	float p2 = 1.0f/(1.0f+exp(-(q.y-p[2])/p[3])); // p(t|a2)
	float p3 = 1.0f/(1.0f+exp(-(q.z-p[4])/p[5])); // p(t|a3)
	float p4 = 1.0f/(1.0f+exp(-(-c-p[6])/p[7])); // p(t|c) 

	return p1*p2*p3*p4; //ptot = p1*p2*p3*p4 # p(t|phase) 
}

/*float phaseConfidenceAO(float3 a, float3 o)
{
	float3 q = a/(o+(float3)(0.000000001));
	float c =  max(fabs(a.x-a.y),max(fabs(a.x-a.z),fabs(a.y-a.z)));
	float p1 = 1.0f/(1.0f+exp(-(a.x-p[0])/p[1])); // p(t|a1)
	float p2 = 1.0f/(1.0f+exp(-(a.y-p[2])/p[3])); // p(t|a2)
	float p3 = 1.0f/(1.0f+exp(-(a.z-p[4])/p[5])); // p(t|a3)
	float p4 = 1.0f/(1.0f+exp(-(-c-p[6])/p[7])); // p(t|c) 
	float p5 = 1.0f/(1.0f+exp(-(q.x-p[8])/p[9])); // p(t|a1)
	float p6 = 1.0f/(1.0f+exp(-(q.y-p[10])/p[11])); // p(t|a2)
	float p7 = 1.0f/(1.0f+exp(-(q.z-p[12])/p[13])); // p(t|a3)
	return p1*p2*p3; //ptot = p1*p2*p3*p4 # p(t|phase) 
}*/

void encodeChannels32moduloSingelHyp(float16* channels_1, float16* channels_2, float phase_hyp)
{
 	uint num_channels = NUM_CHANNELS;
	float num_channels_f = (float)(num_channels);
	float cwidth=M_PI_F/3.0f;
	float pi_over_2 = M_PI_F/2.0f;
	float ssc = 9.0f/num_channels_f;
	float16 ndist_1;
	float16 chans_over_2 = (float16)(num_channels_f/2.0f);
	float16 modulo_tmp;

	//scale phase values to match channels
	float16 phase = (float16)(phase_hyp/ssc+1.0f);

	//initialize channel vector for channels 1-16
	float16 cpos_f = (float16)(1.0f,2.0f,3.0f,4.0f,5.0f,6.0f,7.0f,8.0f,9.0f,10.0f,11.0f,12.0f,13.0f,14.0f,15.0f,16.0f);

	//calculate modulo operation
	modulo_tmp = (cpos_f-phase)-num_channels_f*floor((cpos_f-phase)/num_channels_f);
	ndist_1 = chans_over_2 - fabs(modulo_tmp-chans_over_2);
	ndist_1 = ndist_1*cwidth; 
	
	int16 mask1 = isless(ndist_1, pi_over_2);
	*channels_1 = select((float16)(0.0f),cos(ndist_1)*cos(ndist_1),mask1);

  if(num_channels<17)
		return;

	//initialize channel vector for channels 17-32
	cpos_f=cpos_f+(float16)(16.0f);

	modulo_tmp = (cpos_f-phase)-num_channels_f*floor((cpos_f-phase)/num_channels_f);
	ndist_1 = chans_over_2 - fabs(modulo_tmp-chans_over_2);
	ndist_1 = ndist_1*cwidth; 

	mask1 = isless(ndist_1, pi_over_2);

	*channels_2 = select((float16)(0.0f),cos(ndist_1)*cos(ndist_1),mask1);
}


void kernel processPixelStage2_phase_channels(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float *ir_sums, global float16* channels_1, global float16* channels_2, global half16* channels_1_phase_1, global half16* channels_2_phase_1, global half16* channels_1_phase_2, global half16* channels_2_phase_2, constant float* camera_intrinsics, global const float* c_in)
{
  const uint i = get_global_id(0);
  float3 a = a_in[i];
  float3 b = b_in[i];
	float3 c = (float3)(c_in[i],c_in[i+512*424],c_in[i+512*424*2]);

  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));
  phase = select(phase, (float3)(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;
	ir = select(ir, (float3)(0.0f), isnan(ir));
	c = select(c, (float3)(0.0f), isnan(c));

  float ir_sum = ir.x + ir.y + ir.z;
  /*float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));*/

	float phase_first = 0.0;
	float phase_second = 0.0;
	//float conf = 1.0;
	float w_err1, w_err2;

	float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

	float t0 = t.x;
	float t1 = t.y;
	float t2 = t.z;

	phaseUnWrapper(t0, t1, t2, &phase_first, &phase_second, &w_err1, &w_err2);

	phase_1[i] = phase_first;
	phase_2[i] = phase_second;

	//float3 p = ir/(c+(float3)(0.000000001));
	float phase_conf = phaseConfidenceA(ir);
	//float phase_conf = phaseConfidenceAO(ir,c);
	//float phase_conf = phaseConfidenceAoverO(ir,c);
	w_err1 = phase_conf*exp(-w_err1/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err2 = phase_conf*exp(-w_err2/(2*CHANNEL_CONFIDENCE_SCALE));

	w_err1 = phase_first > MAX_DEPTH*9.0f/18750.0f || phase_first < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err1;
	w_err2 = phase_second > MAX_DEPTH*9.0f/18750.0f || phase_second < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err2;

	float16 ch_1_phase1, ch_2_phase1, ch_1_phase2, ch_2_phase2;
	encodeChannels32moduloSingelHyp(&ch_1_phase1, &ch_2_phase1, phase_first);
	encodeChannels32moduloSingelHyp(&ch_1_phase2, &ch_2_phase2, phase_second);

	channels_1[i] = w_err1*ch_1_phase1+w_err2*ch_1_phase2;
	channels_2[i] = w_err1*ch_2_phase1+w_err2*ch_2_phase2;
	channels_1_phase_1[i] = (half16)ch_1_phase1;
	channels_2_phase_1[i] = (half16)ch_2_phase1;
	channels_1_phase_2[i] = (half16)ch_1_phase2;
	channels_2_phase_2[i] = (half16)ch_2_phase2;

	ir_sums[i] = ir_sum;
}


/*****************************************************************
 * THREE HYPOTHESIS
 *****************************************************************/

void phaseUnWrapper3(float t0, float t1,float t2, float* phase_first, float* phase_second, float* phase_third, float* err_w1, float* err_w2, float* err_w3)
{
  float err;
  float err1,err2,err3;
  //unsigned int ind_count = 1;
	//float k,nc,nf;
	//float w1 = 0.7007;
	//float w2 = 366.2946;
	//float w3 = 0.7016;
	
	float w1 = 1.0f;
	float w2 = 18.0f;
	float w3 = 1.0218f;

	float err_min=100000.0f;
	float err_min_second = 200000.0f;
	float err_min_third = 300000.0f;
	unsigned int ind_min, ind_second, ind_third;

	float k,n,m;
	
	for(int i=0; i<30; i++)
	{
		m = m_list[i];
		n = n_list[i];
		k = k_list[i];
		calcErr(k,n,m,t0,t1,t2,&err1,&err2,&err3);
		err = w1*err1*err1+w2*err2*err2+w3*err3*err3;
		if(err<err_min)
		{
			err_min_third = err_min_second;
			ind_third = ind_second;
			err_min_second = err_min;
			ind_second = ind_min;
			err_min = err;
			ind_min = i;

		}
		else if(err<err_min_second)
		{
			err_min_third = err_min_second;
			ind_third = ind_second;
    	err_min_second = err;
			ind_second = i;
		}
		else if(err<err_min_third)
		{
    	err_min_third = err;
			ind_third = i;
		}
		
	}

	//decode ind_min
	float mvals = m_list[ind_min];
	float nvals = n_list[ind_min];
	float kvals = k_list[ind_min];

	float phi2_out = (t2/2.0f+mvals);
	float phi1_out = (t1/15.0f+kvals);
	float phi0_out = (t0/3.0f+nvals);

	*err_w1 = err_min;

	*phase_first = (phi2_out+phi1_out+phi0_out)/3.0f;

	mvals = m_list[ind_second];
	nvals = n_list[ind_second];
	kvals = k_list[ind_second];

	phi2_out = (t2/2.0f+mvals);
	phi1_out = (t1/15.0f+kvals);
	phi0_out = (t0/3.0f+nvals);	

	*err_w2 = err_min_second;
	*phase_second = (phi2_out+phi1_out+phi0_out)/3.0f;

	mvals = m_list[ind_third];
	nvals = n_list[ind_third];
	kvals = k_list[ind_third];

	phi2_out = (t2/2.0f+mvals);
	phi1_out = (t1/15.0f+kvals);
	phi0_out = (t0/3.0f+nvals);

	*err_w3 = err_min_third;
	*phase_third = (phi2_out+phi1_out+phi0_out)/3.0f;
}


void kernel processPixelStage2_phase_channels3(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float *phase_3, global float *ir_sums, global float16* channels_1, global float16* channels_2)
{
  const uint i = get_global_id(0);
  float3 a = a_in[i];
  float3 b = b_in[i];
	
  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));
  phase = select(phase, (float3)(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;
	ir = select(ir, (float3)(0.0f), isnan(ir));
	
  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

	float phase_first = 0.0f;
	float phase_second = 0.0f;
	float phase_third = 0.0f;

	float w_err1, w_err2, w_err3;

	float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

	float t0 = t.x;
	float t1 = t.y;
	float t2 = t.z;

	phaseUnWrapper3(t0, t1, t2, &phase_first, &phase_second, &phase_third, &w_err1, &w_err2, &w_err3);

	phase_1[i] = phase_first;
	phase_2[i] = phase_second;
	phase_3[i] = phase_third;

	float phase_conf = phaseConfidenceA(ir);
	w_err1 = phase_conf*exp(-w_err1/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err2 = phase_conf*exp(-w_err2/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err3 = phase_conf*exp(-w_err3/(2*CHANNEL_CONFIDENCE_SCALE));

	w_err1 = phase_first > MAX_DEPTH*9.0f/18750.0f || phase_first < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err1;
	w_err2 = phase_second > MAX_DEPTH*9.0f/18750.0f || phase_second < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err2;
	w_err3 = phase_third > MAX_DEPTH*9.0f/18750.0f || phase_third < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err3;

	float16 ch_1_phase1, ch_2_phase1, ch_1_phase2, ch_2_phase2, ch_1_phase3, ch_2_phase3;
	encodeChannels32moduloSingelHyp(&ch_1_phase1, &ch_2_phase1, phase_first);
	encodeChannels32moduloSingelHyp(&ch_1_phase2, &ch_2_phase2, phase_second);
	encodeChannels32moduloSingelHyp(&ch_1_phase3, &ch_2_phase3, phase_third);

	channels_1[i] = w_err1*ch_1_phase1+w_err2*ch_1_phase2+w_err3*ch_1_phase3;
	channels_2[i] = w_err1*ch_2_phase1+w_err2*ch_2_phase2+w_err3*ch_2_phase3;
	ir_sums[i] = phase_conf;
}
/*
void filter_channels16_fast_help(global const float16* channels, global float16* channels_filtered, const uint th_x, const uint th_y, const uint block_x, const uint block_y, global const float* gauss_filt_array)
{
	int k, l;
  float16 sumx;

	int blocksize_x = BLOCK_SIZE_COL;
  int blocksize_y = BLOCK_SIZE_ROW;  

  local float16 blockImage[(BLOCK_SIZE_COL+CHANNEL_FILT_SIZE*2)*(BLOCK_SIZE_ROW+CHANNEL_FILT_SIZE*2)];
  //load
  int loadY = block_y*BLOCK_SIZE_ROW+th_y-CHANNEL_FILT_SIZE;
  int loadX = block_x*BLOCK_SIZE_COL+th_x-CHANNEL_FILT_SIZE;
	
  if(loadY >= 0 && loadY < 424 && loadX >= 0 && loadX < 512) {
    blockImage[(th_y*(blocksize_x+2*CHANNEL_FILT_SIZE)+th_x)] = channels[(loadY*512+loadX)];
  }

  //sync
  barrier(CLK_LOCAL_MEM_FENCE);

  //compute
	float divby = (float)((2*CHANNEL_FILT_SIZE+1)*(2*CHANNEL_FILT_SIZE+1));
	//if inside filter sized edge of block
  if(th_y >= CHANNEL_FILT_SIZE && th_x >= CHANNEL_FILT_SIZE && th_y < blocksize_y+CHANNEL_FILT_SIZE && th_x < blocksize_x+CHANNEL_FILT_SIZE) {
    if(loadY < 424 && loadX < 512) // If inside image
    {
			//if inside filter sized edge of image
			if(loadX >= CHANNEL_FILT_SIZE+1 && loadX < 512-CHANNEL_FILT_SIZE-1 && loadY >= CHANNEL_FILT_SIZE && loadY < 424-CHANNEL_FILT_SIZE)
			{
				// Filter kernel
				sumx=(float16)(0.0f);
				for(k=-CHANNEL_FILT_SIZE;k<=CHANNEL_FILT_SIZE;k++)
					for(l=-CHANNEL_FILT_SIZE;l<=CHANNEL_FILT_SIZE;l++)	
					{
						//sumx += gauss_filt_array[k+CHANNEL_FILT_SIZE]*gauss_filt_array[l+CHANNEL_FILT_SIZE]*blockImage[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
						sumx = blockImage[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
					}
				//channels_filtered[(loadY*512+loadX)] = sumx;
				channels_filtered[(loadY*512+loadX)] = sumx/divby;
			}
			else
			{
				channels_filtered[(loadY*512+loadX)] = blockImage[(th_y*(blocksize_x+2*CHANNEL_FILT_SIZE)+th_x)];
			}
    }
  }
}
*/
/*
void filter_channels32_fast_help(global const float16* channels1, global const float16* channels2, global float16* channels1_filtered, global float16* channels2_filtered, const uint th_x, const uint th_y, const uint block_x, const uint block_y, global const float* gauss_filt_array)
{
	int k, l;
  float16 sumx1, sumx2;

	int blocksize_x = BLOCK_SIZE_COL;
  int blocksize_y = BLOCK_SIZE_ROW;  

  local float16 blockImage1[(BLOCK_SIZE_COL+CHANNEL_FILT_SIZE*2)*(BLOCK_SIZE_ROW+CHANNEL_FILT_SIZE*2)];
	local float16 blockImage2[(BLOCK_SIZE_COL+CHANNEL_FILT_SIZE*2)*(BLOCK_SIZE_ROW+CHANNEL_FILT_SIZE*2)];
  //load
  int loadY = block_y*BLOCK_SIZE_ROW+th_y-CHANNEL_FILT_SIZE;
  int loadX = block_x*BLOCK_SIZE_COL+th_x-CHANNEL_FILT_SIZE;
	
  if(loadY >= 0 && loadY < 424 && loadX >= 0 && loadX < 512) {
    blockImage1[(th_y*(blocksize_x+2*CHANNEL_FILT_SIZE)+th_x)] = channels1[(loadY*512+loadX)];
		if(NUM_CHANNELS>16)
			blockImage2[(th_y*(blocksize_x+2*CHANNEL_FILT_SIZE)+th_x)] = channels2[(loadY*512+loadX)];
  }

  //sync
  barrier(CLK_LOCAL_MEM_FENCE);

  //compute
	//float divby = (float)((2*CHANNEL_FILT_SIZE+1)*(2*CHANNEL_FILT_SIZE+1));
	float gauss;
	//if inside filter sized edge of block
  if(th_y >= CHANNEL_FILT_SIZE && th_x >= CHANNEL_FILT_SIZE && th_y < blocksize_y+CHANNEL_FILT_SIZE && th_x < blocksize_x+CHANNEL_FILT_SIZE) {
    if(loadY < 424 && loadX < 512) // If inside image
    {
			//if inside filter sized edge of image
			if(loadX >= 1 && loadX < 511 && loadY >= 0)
			{
				// Filter kernel
				sumx1=(float16)(0.0f);
				sumx2=(float16)(0.0f);
				int from_x = loadX>CHANNEL_FILT_SIZE+1 ? -CHANNEL_FILT_SIZE : -loadX+1;
				int from_y = loadY>CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadY+1;
				int to_x = loadX<512-CHANNEL_FILT_SIZE-1 ? CHANNEL_FILT_SIZE: 511-loadX-1;
				int to_y = loadY<424-CHANNEL_FILT_SIZE-1 ? CHANNEL_FILT_SIZE: 423-loadY;
				for(k=-from_y;k<=to_y;k++)
					for(l=-from_x;l<=to_x;l++)	
					{
						gauss = gauss_filt_array[k+CHANNEL_FILT_SIZE]*gauss_filt_array[l+CHANNEL_FILT_SIZE];
						sumx1 += gauss*blockImage1[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
						sumx2 += gauss*blockImage2[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
						//sumx1 += blockImage1[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
						//sumx2 += blockImage2[(th_y+k)*(blocksize_y+CHANNEL_FILT_SIZE*2)+(th_x+l)];
					}
				//channels_filtered[(loadY*512+loadX)] = sumx;
				channels1_filtered[(loadY*512+loadX)] = sumx1;
				if(NUM_CHANNELS>16)
					channels2_filtered[(loadY*512+loadX)] = sumx2;

			}
    }
  }
}*/

void filter_channels16_help(global const half16* channels, global float16* channels_filtered, const uint i, global const float* gauss_filt_array)
{
	const int loadX = i % 512;
	const int loadY = i / 512;
	int k, l;
  float16 sumx;
	
	int from_x = (loadX > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadX+1);
	int from_y = (loadY > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadY+1);
	int to_x = (loadX < 511-CHANNEL_FILT_SIZE-1 ? CHANNEL_FILT_SIZE: 511-loadX-1);
	int to_y = (loadY < 423-CHANNEL_FILT_SIZE ? CHANNEL_FILT_SIZE: 423-loadY);
  //compute
	float divby = (float)((to_x-from_x+1)*(to_y-from_y+1));
  
	if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
  // Filter kernel
    sumx=(float16)(0.0f);
		float gauss;
		float sum_gauss = 0.0f;

		for(k=from_y; k<=to_y; k++)
		  for(l=from_x; l<=to_x; l++)
	    {
				gauss = gauss_filt_array[k+CHANNEL_FILT_SIZE]*gauss_filt_array[l+CHANNEL_FILT_SIZE];
				sum_gauss += gauss;
		    sumx += gauss*(float16)channels[((loadY+k)*512+(loadX+l))];
				//sumx += channels[((loadY+k)*512+(loadX+l))];
	    }
		channels_filtered[i] = sumx/sum_gauss;
    //channels_filtered[(loadY*512+loadX)] = sumx/divby;
  }
  
}


/*void kernel filter_channels_fast(global const float16* channels_1, global const float16* channels_2, global float16* channels_filtered_1, global float16* channels_filtered_2, global const float* gauss_filt_array)
{
  const uint th_y = get_local_id(1);
  const uint th_x = get_local_id(0);
  const uint block_y = get_group_id(1);
  const uint block_x = get_group_id(0);

	//filter_channels32_fast_help(channels_1, channels_2, channels_filtered_1, channels_filtered_2, th_x, th_y, block_x, block_y, gauss_filt_array);

}
*/


void kernel filter_channels(global const half16* channels_1, global const half16* channels_2, global float16* channels_filtered_1, global float16* channels_filtered_2, global const float* gauss_filt_array)
{
	const uint i = get_global_id(0);

	filter_channels16_help(channels_1, channels_filtered_1, i, gauss_filt_array);

	if(NUM_CHANNELS>16)
	{
		filter_channels16_help(channels_2, channels_filtered_2, i, gauss_filt_array);
	}
		
}

/***************************************
            1.0000         0    1.0000 *
pinv(A) =  -0.5000    0.8660    1.0000 *
           -0.5000   -0.8660    1.0000 *
****************************************/
/*
float decodeChannelsModulo(global const float16* channels_1, global const float16* channels_2, const uint i)
{
	float num_channels_f = (float)(NUM_CHANNELS);
	float cwidth=M_PI_F/3.0f;
	float channels_arr_1[16],channels_arr_2[16];

	VEC16_TO_ARRAY(channels_1[i],channels_arr_1);
	//channels
	if(NUM_CHANNELS>16)
		VEC16_TO_ARRAY(channels_2[i],channels_arr_2);

	float val_1,val_2,val_3, offset, signal_val_tmp;
	uint num_channels = NUM_CHANNELS;
	uint cpos, pos_1,pos_3,mask;
	float signal_val[NUM_CHANNELS], r_val[NUM_CHANNELS], m_val[NUM_CHANNELS]; 
	float3 vec;
	for(cpos = 0; cpos < NUM_CHANNELS; cpos++)
	{
		pos_1 = (cpos-1) % num_channels;
		pos_3 = (cpos+1) % num_channels;

		//Extract values from channels
		val_1 = pos_1 < 16 ? channels_arr_1[pos_1] : channels_arr_2[pos_1-16];
		val_2 = cpos < 16 ? channels_arr_1[cpos] : channels_arr_2[cpos-16];
		val_3 = pos_3 < 16 ? channels_arr_1[pos_3] : channels_arr_2[pos_3-16];

		// pinv(A)*val_vec		
		vec.x = 4.0f/3.0f*val_1-2.0f/3.0f*val_2-2.0f/3.0f*val_3;
		vec.y = 0.0f*val_1+2.0f/sqrt(3.0f)*val_2-2.0f/sqrt(3.0f)*val_3;
		vec.z = 2.0f/3.0f*(val_1+val_2+val_3);

		//
		offset = 0.5f*atan2(vec.y,vec.x)/cwidth;
		mask = offset < 1.5 && offset > -1.49;
		signal_val_tmp = cpos+offset;

		//modulo modulo_tmp = (cpos_f-phase2)-num_channels_f*floor((cpos_f-phase2)/num_channels_f);
		signal_val_tmp = (signal_val_tmp-2.0f) - num_channels_f*floor((signal_val_tmp-2.0f)/num_channels_f);

		signal_val_tmp *= (vec.z*mask > 0.0f);
		//scaling
		signal_val[cpos] = signal_val_tmp*9.0f/num_channels_f;

		r_val[cpos] = vec.z*mask;
		m_val[cpos] = sqrt(vec.y*vec.y+vec.x*vec.x)*mask;
	}

	float max_r = 0.0f;
	uint max_pos = 0;
	for(cpos = 0; cpos < NUM_CHANNELS; cpos++)
	{
		if(max_r<r_val[cpos])
		{
			max_r=r_val[cpos];
			max_pos = cpos;
		}
	}

	return signal_val[max_pos];
	
}*/

void kernel processPixelStage2_depth_channels(global float *phase_1, global float *phase_2, global const float16* channels_filtered_1, global const float16* channels_filtered_2, global const float16* channels_1_phase_1, global const float16* channels_2_phase_1, global const float16* channels_1_phase_2, global const float16* channels_2_phase_2, global const float *x_table, global const float *z_table, global float* depth)
{
	const uint i = get_global_id(0);

	//float s = decodeChannelsModulo(channels_filtered_1, channels_filtered_2, i);
	float16 ch1,ch2;
	float16 channels_filtered_1_local = channels_filtered_1[i];
	float16 channels_filtered_2_local = channels_filtered_2[i];

	float phase_first = phase_1[i];
	float phase_second = phase_2[i];
	//encodeChannels32moduloSingelHyp(&ch1, &ch2, phase_first);
	ch1 = channels_1_phase_1[i];
	ch2 = channels_2_phase_1[i];
	float val1 = DOT_16(ch1,channels_filtered_1_local)+(NUM_CHANNELS > 16 ? DOT_16(ch2,channels_filtered_2_local): 0.0f);
	//encodeChannels32moduloSingelHyp(&ch1, &ch2, phase_second);
	ch1 = channels_1_phase_2[i];
	ch2 = channels_2_phase_2[i];
	float val2 = DOT_16(ch1,channels_filtered_1_local)+(NUM_CHANNELS > 16 ? DOT_16(ch2,channels_filtered_2_local): 0.0f);
	//float phase_final = fabs(phase_first-s) < fabs(phase_second-s) ? phase_first: phase_second;
	int val_ind = val2 <= val1 ? 1: 0;
	//int val_ind = isless(val2,val1);
	float phase_final = val_ind ? phase_first: phase_second;
	float max_val = val_ind ? val1: val2;
	
	float zmultiplier = z_table[i];
	float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z

	max_val = depth_fit < MIN_DEPTH || depth_fit > MAX_DEPTH ? 0.0f: max_val;
  //depth[i] = max_val >= 0.14f ? DEPTH_SCALE*depth_linear: 0.0f;
	depth[i] = depth_fit;
	depth[i+512*424] = 9.0f/8.0f*max_val;

}

void kernel processPixelStage2_depth_channels3(global const float *phase_1, global const float *phase_2, global const float *phase_3, global const float16* channels_filtered_1, global const float16* channels_filtered_2, global const float *x_table, global const float *z_table, global float* depth)
{
	const uint i = get_global_id(0);


	//float s = decodeChannelsModulo(channels_filtered_1, channels_filtered_2, i);
	float16 ch1,ch2;
	float16 channels_filtered_1_local = channels_filtered_1[i];
	float16 channels_filtered_2_local = channels_filtered_2[i];

	float phase_first = phase_1[i];
	float phase_second = phase_2[i];
	float phase_third = phase_3[i];

	encodeChannels32moduloSingelHyp(&ch1, &ch2, phase_first);
	float val1 = DOT_16(ch1,channels_filtered_1_local)+(NUM_CHANNELS > 16 ? DOT_16(ch2,channels_filtered_2_local): 0.0f);
	encodeChannels32moduloSingelHyp(&ch1, &ch2, phase_second);
	float val2 = DOT_16(ch1,channels_filtered_1_local)+(NUM_CHANNELS > 16 ? DOT_16(ch2,channels_filtered_2_local): 0.0f);
	encodeChannels32moduloSingelHyp(&ch1, &ch2, phase_third);
	float val3 = DOT_16(ch1,channels_filtered_1_local)+(NUM_CHANNELS > 16 ? DOT_16(ch2,channels_filtered_2_local): 0.0f);

	float max_val, phase_final;
	if(val1 >= val2 && val1 >= val3)
	{
		max_val = val1;
		phase_final = phase_first;
	} 
	else if(val2 > val1 && val2 >= val3)
	{
		max_val = val2;
		phase_final = phase_second;
	}	
	else
	{
		max_val = val3;
		phase_final = phase_third;
	}
	
	float zmultiplier = z_table[i];
	float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z
	max_val = d < MIN_DEPTH || d > MAX_DEPTH ? 0.0f: max_val;
  //depth[i] = max_val >= 0.01 ? DEPTH_SCALE*depth_linear: 0.0f;
	depth[i] = d;
	depth[i+512*424] = max_val;
}
