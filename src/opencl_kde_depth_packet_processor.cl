
/*******************************************************************************
 * Process pixel stage 1
 ******************************************************************************/

#define DEPTH_SCALE 1.0f
//#define DEPTH_SCALE 0.15f

#define PHASE (float3)(PHASE_IN_RAD0, PHASE_IN_RAD1, PHASE_IN_RAD2)
#define AB_MULTIPLIER_PER_FRQ (float3)(AB_MULTIPLIER_PER_FRQ0, AB_MULTIPLIER_PER_FRQ1, AB_MULTIPLIER_PER_FRQ2)

float decodePixelMeasurement(global const ushort *data, global const short *lut11to16, const uint sub, const uint x, const uint y)
{
  uint row_idx = (424 * sub + y) * 352;
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
                               global float3 *a_out, global float3 *b_out, global float3 *n_out, global float *ir_out)
{
    const uint i = get_global_id(0);

    const uint x = i % 512;
    const uint y = i / 512;

		const uint y_tmp = (423 - y);
		const uint y_in = (y_tmp < 212 ? y_tmp + 212 : 423 - y_tmp);

		const int3 invalid = (int)(0.0f >= z_table[i]);
		const float3 p0 = p0_table[i];
		float3 p0x_sin, p0y_sin, p0z_sin;
		float3 p0x_cos, p0y_cos, p0z_cos;

		p0x_sin = -sincos(PHASE + p0.x, &p0x_cos);
		p0y_sin = -sincos(PHASE + p0.y, &p0y_cos);
		p0z_sin = -sincos(PHASE + p0.z, &p0z_cos);

		int3 invalid_pixel = (int3)(invalid);

	const float3 v0 = (float3)(decodePixelMeasurement(data, lut11to16, 0, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 1, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 2, x, y_in));
  const float3 v1 = (float3)(decodePixelMeasurement(data, lut11to16, 3, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 4, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 5, x, y_in));
  const float3 v2 = (float3)(decodePixelMeasurement(data, lut11to16, 6, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 7, x, y_in),
                             decodePixelMeasurement(data, lut11to16, 8, x, y_in));

  float3 a = (float3)(dot(v0, p0x_cos),
                      dot(v1, p0y_cos),
                      dot(v2, p0z_cos)) * AB_MULTIPLIER_PER_FRQ;
  float3 b = (float3)(dot(v0, p0x_sin),
                      dot(v1, p0y_sin),
                      dot(v2, p0z_sin)) * AB_MULTIPLIER_PER_FRQ;

  a = select(a, (float3)(0.0f), invalid_pixel);
  b = select(b, (float3)(0.0f), invalid_pixel);
  float3 n = sqrt(a * a + b * b);

  int3 saturated = (int3)(any(isequal(v0, (float3)(32767.0f))),
                          any(isequal(v1, (float3)(32767.0f))),
                          any(isequal(v2, (float3)(32767.0f))));

  a_out[i] = select(a, (float3)(0.0f), saturated);
  b_out[i] = select(b, (float3)(0.0f), saturated);
  n_out[i] = n;
  ir_out[i] = min(dot(select(n, (float3)(65535.0f), saturated), (float3)(0.333333333f  * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER)), 65535.0f);
}

/*******************************************************************************
 * Filter pixel stage 1
 ******************************************************************************/
void kernel filterPixelStage1(global const float3 *a, global const float3 *b, global const float3 *n,
                              global float3 *a_out, global float3 *b_out, global uchar *max_edge_test)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const float3 self_a = a[i];
  const float3 self_b = b[i];

  const float gaussian[9] = {GAUSSIAN_KERNEL_0, GAUSSIAN_KERNEL_1, GAUSSIAN_KERNEL_2, GAUSSIAN_KERNEL_3, GAUSSIAN_KERNEL_4, GAUSSIAN_KERNEL_5, GAUSSIAN_KERNEL_6, GAUSSIAN_KERNEL_7, GAUSSIAN_KERNEL_8};

  if(x < 1 || y < 1 || x > 510 || y > 422)
  {
    a_out[i] = self_a;
    b_out[i] = self_b;
    max_edge_test[i] = 1;
  }
  else
  {
    float3 threshold = (float3)(JOINT_BILATERAL_THRESHOLD);
    float3 joint_bilateral_exp = (float3)(JOINT_BILATERAL_EXP);

    const float3 self_norm = n[i];
    const float3 self_normalized_a = self_a / self_norm;
    const float3 self_normalized_b = self_b / self_norm;

    float3 weight_acc = (float3)(0.0f);
    float3 weighted_a_acc = (float3)(0.0f);
    float3 weighted_b_acc = (float3)(0.0f);
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
        const float3 other_normalized_a = other_a / other_norm;
        const float3 other_normalized_b = other_b / other_norm;

        const int3 c1 = isless(other_norm * other_norm, threshold);

        const float3 dist = 0.5f * (1.0f - (self_normalized_a * other_normalized_a + self_normalized_b * other_normalized_b));
        const float3 weight = select(gaussian[j] * exp(-1.442695f * joint_bilateral_exp * dist), (float3)(0.0f), c1);

        weighted_a_acc += weight * other_a;
        weighted_b_acc += weight * other_b;
        weight_acc += weight;
        dist_acc += select(dist, (float3)(0.0f), c1);
      }
    }

    const int3 c2 = isless((float3)(0.0f), weight_acc.xyz);
    a_out[i] = select((float3)(0.0f), weighted_a_acc / weight_acc, c2);
    b_out[i] = select((float3)(0.0f), weighted_b_acc / weight_acc, c2);

    max_edge_test[i] = all(isless(dist_acc, (float3)(JOINT_BILATERAL_MAX_EDGE)));
  }
}

/*******************************************************************************
 * Process pixel stage 2
 ******************************************************************************/
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
//float constant p[8]={ 4.49432993f,  1.36533393f,  4.42313275f,  1.15035961f,  4.11977512f, 1.83015617f, 0, 0};

//Tränat på session 5 med a1,a2,a3,pcons: 'ms25d1f4'
float constant p[8]={ 4.48765613f, 1.3545396f, 4.42689315f, 1.14757834f, 4.1070274f, 1.80847054f, -26.03667419f, 4.36424484f};

float phaseConfidenceA(float3 a)
{
	float c =  max(fabs(a.x-a.y),max(fabs(a.x-a.z),fabs(a.y-a.z)));
	float p1 = 1.0f/(1.0f+exp(-(a.x-p[0])/p[1])); // p(t|a1)
	float p2 = 1.0f/(1.0f+exp(-(a.y-p[2])/p[3])); // p(t|a2)
	float p3 = 1.0f/(1.0f+exp(-(a.z-p[4])/p[5])); // p(t|a3)
	float p4 = 1.0f/(1.0f+exp(-(-c-p[6])/p[7])); // p(t|c) 

	return p1*p2*p3; //ptot = p1*p2*p3*p4 # p(t|phase) 
}


float constant k_list[30] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
float constant n_list[30] = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 5.0f, 6.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 8.0f, 8.0f, 7.0f, 8.0f, 9.0f, 9.0f};
float constant m_list[30] = {0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 3.0f, 3.0f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f, 7.0f, 7.0f, 7.0f, 7.0f, 8.0f, 8.0f, 9.0f, 9.0f, 10.0f, 10.0f, 11.0f, 11.0f, 12.0f, 12.0f, 13.0f, 13.0f, 14.0f};

void calcErr(const float k, const float n, const float m, const float t0, const float t1, const float t2, float* err1, float* err2, float* err3)
{
    *err1 = 3.0f*n-15.0f*k-(t1-t0);
    *err2 = 3.0f*n-2.0f*m-(t2-t0);
    *err3 = 15.0f*k-2.0f*m-(t2-t1);
}

float moduloDistance(float x, float y, float m)
{
	return m/2-fabs((x-y)-m*floor((x-y)/m)-m/2);
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
	
	//float std0 = 3*exp(1.2725f-ir.x*0.808f);
	//float std1 = 15*exp(-1.8149f-ir.y*0.0370f);
	//float std2 = 2*exp(0.8242f-ir.z*0.0865f);

	//float w1 = 1.0f/(std0*std0+std1*std1);
	//float w2 = 1.0f/(std0*std0+std2*std2);//366.2946;
	//float w3 = 1.0f/(std1*std1+std2*std2);
	
	float w1 = 1.0f;
	float w2 = 10.0f;
	float w3 = 1.0218f;

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

}


void calculatePhaseUnwrappingWeightsInv(float3 ir, float* weight1, float* weight2, float* weight3)
{
	// s0_est = 1.6520f;
	// s1_est = 1.1506f;
	// s2_est = 2.0713f;
	float q0 = (ir.x/2.0009f);
	float q1 = (ir.y/1.4843f);
	float q2 = (ir.z/2.4758f);
  float alpha0 = q0 > 1.0001f ? atan(sqrt(1.0f/(q0*q0-1.0f))) : ir.x > 0.0f ? 2.0009f*0.5f*M_PI_F/ir.x : 2.0f*M_PI_F;
	float alpha1 = q1 > 1.0001f ? atan(sqrt(1.0f/(q1*q1-1.0f))) : ir.y > 0.0f ? 1.4843f*0.5f*M_PI_F/ir.y : 2.0f*M_PI_F;
	float alpha2 = q2 > 1.0001f ? atan(sqrt(1.0f/(q2*q2-1.0f))) : ir.z > 0.0f ? 2.4758f*0.5f*M_PI_F/ir.z : 2.0f*M_PI_F;
	//alpha0 = alpha0 > 0.5f*M_PI_F ? 0.5f*M_PI_F : alpha0;
	//alpha1 = alpha1 > 0.5f*M_PI_F ? 0.5f*M_PI_F : alpha1;
	//alpha2 = alpha2 > 0.5f*M_PI_F ? 0.5f*M_PI_F : alpha2;

	float sigma0 = 3.0f*alpha0;
	float sigma1 = 15.0f*alpha1;
	float sigma2 = 2.0f*alpha2;

	float w1=1.0f/(sigma1*sigma1+sigma0*sigma0);
	float w2=1.0f/(sigma2*sigma2+sigma0*sigma0);
	float w3=1.0f/(sigma2*sigma2+sigma1*sigma1);
	*weight2 = w2;
	*weight3 = w3;
	*weight1 = w1;
}

void calculatePhaseUnwrappingWeightsLin(float3 ir, float* weight1, float* weight2, float* weight3)
{
	// s0_est = [0.6030 -1.2627];
	// s1_est = [0.8494 -0.7601];
	// s2_est = [0.4927 -1.5650];

	float q0 = 0.6030f*ir.x-1.2627f;
	float q1 = 0.8494f*ir.y-0.7601f;
	float q2 = 0.4927f*ir.z-1.5650f;
	q0*=q0;
	q1*=q1;
	q2*=q2;

  float alpha0 = q0 > 1.0001f ? atan(sqrt(1.0f/(q0-1.0f))) : ir.x > 0.0f ? 0.5f*M_PI_F/ir.x : 2.0f*M_PI_F;
	float alpha1 = q1 > 1.0001f ? atan(sqrt(1.0f/(q1-1.0f))) : ir.y > 0.0f ? 0.5f*M_PI_F/ir.y : 2.0f*M_PI_F;
	float alpha2 = q2 > 1.0001f ? atan(sqrt(1.0f/(q2-1.0f))) : ir.z > 0.0f ? 0.5f*M_PI_F/ir.z : 2.0f*M_PI_F;

	float sigma0 = 3.0f*alpha0;
	float sigma1 = 15.0f*alpha1;
	float sigma2 = 2.0f*alpha2;

	float w1=1.0f/(sigma1*sigma1+sigma0*sigma0);
	float w2=1.0f/(sigma2*sigma2+sigma0*sigma0);
	float w3=1.0f/(sigma2*sigma2+sigma1*sigma1);
	*weight2 = w2;
	*weight3 = w3;
	*weight1 = w1;
}

void calculatePhaseUnwrappingWeightsQuad(float3 ir, float* weight1, float* weight2, float* weight3)
{
	//Learning on mean ir 
	// s0_est = [0.7854 -1.0018 -3.1426];
	// s1_est = [1.1106 -0.0028 -3.1280];
	// s2_est = [0.4927 -1.0014 -3.2086];

	//Learning on ir
	// s0_est = [0.752836906983593 -0.001906531018764 -2.644774735852631]; root = 4.902247094595301
	// s1_est = [1.084642404842933 -0.002607599266011 -2.942871512263318]; root = 3.6214441719887
	// s2_est = [0.615468609808938 -0.001252148462401 -2.764589412408904]; root = 6.194693909705903
	float q0 = 0.7528369f*ir.x-0.001906531f*ir.x*ir.x-2.64477474f;
	float q1 = 1.0846424f*ir.y-0.002607599f*ir.y*ir.y-2.94287151f;
	float q2 = 0.61546861f*ir.z-0.00125214846f*ir.z*ir.z-2.7645894124f;
	q0*=q0;
	q1*=q1;
	q2*=q2;

  float alpha0 = ir.x > 4.9022471f ? atan(sqrt(1.0f/(q0-1.0f))) : ir.x > 0.0f ? 4.88786f*0.5f*M_PI_F/ir.x : 2.0f*M_PI_F;
	float alpha1 = ir.y > 3.62144418f ? atan(sqrt(1.0f/(q1-1.0f))) : ir.y > 0.0f ?  3.621444f*0.5f*M_PI_F/ir.y : 2.0f*M_PI_F;
	float alpha2 = ir.z > 6.19469391f ? atan(sqrt(1.0f/(q2-1.0f))) : ir.z > 0.0f ? 6.19469391f*0.5f*M_PI_F/ir.z : 2.0f*M_PI_F;

	alpha0 = alpha0 < 0.001f ? 0.001f: alpha0;
	alpha1 = alpha1 < 0.001f ? 0.001f: alpha1;
	alpha2 = alpha2 < 0.001f ? 0.001f: alpha2;

	float sigma0 = 3.0f*alpha0;
	float sigma1 = 15.0f*alpha1;
	float sigma2 = 2.0f*alpha2;

	float w1=1.0f/(sigma1*sigma1+sigma0*sigma0);
	float w2=1.0f/(sigma2*sigma2+sigma0*sigma0);
	float w3=1.0f/(sigma2*sigma2+sigma1*sigma1);
	*weight2 = w2;
	*weight3 = w3;
	*weight1 = w1;
}

void calculatePhaseUnwrappingVarDirect(float3 ir, float* var0, float* var1, float* var2)
{
	//Learning on ir
	// s0_est = [0.826849742438142 -0.003233242852675 -3.302669070396207]; 
	// s1_est = [1.214266793857512 -0.005810826339530 -3.863119924097905]; 
	// s2_est = [0.610145746418116 -0.001136792329934 -2.846144420368535]; 
	float q0 = 0.82684974*ir.x-0.00323324f*ir.x*ir.x-3.86311992f;
	float q1 = 1.2142668f*ir.y-0.005810826f*ir.y*ir.y-2.94287151f;
	float q2 = 0.610145746f*ir.z-0.0011367923f*ir.z*ir.z-2.8461444204f;

  float alpha0 = 1.0f/q0;
	float alpha1 = 1.0f/q1;
	float alpha2 = 1.0f/q2;

	alpha0 = alpha0 < 0.001f ? 0.001f: alpha0;
	alpha1 = alpha1 < 0.001f ? 0.001f: alpha1;
	alpha2 = alpha2 < 0.001f ? 0.001f: alpha2;
	
	*var0 = alpha0;
	*var1 = alpha1;
	*var2 = alpha2;

}

void calculatePhaseUnwrappingVar(float3 ir, float* var0, float* var1, float* var2)
{
	//Learning on mean ir 
	// s0_est = [0.7854 -1.0018 -3.1426];
	// s1_est = [1.1106 -0.0028 -3.1280];
	// s2_est = [0.4927 -1.0014 -3.2086];

	//Learning on ir
	// s0_est = [0.752836906983593 -0.001906531018764 -2.644774735852631]; root = 4.902247094595301
	// s1_est = [1.084642404842933 -0.002607599266011 -2.942871512263318]; root = 3.6214441719887
	// s2_est = [0.615468609808938 -0.001252148462401 -2.764589412408904]; root = 6.194693909705903
	float q0 = 0.7528369f*ir.x-0.001906531f*ir.x*ir.x-2.64477474f;
	float q1 = 1.0846424f*ir.y-0.002607599f*ir.y*ir.y-2.94287151f;
	float q2 = 0.61546861f*ir.z-0.00125214846f*ir.z*ir.z-2.7645894124f;
	q0*=q0;
	q1*=q1;
	q2*=q2;

  float alpha0 = ir.x > 4.9022471f ? atan(sqrt(1.0f/(q0-1.0f))) : ir.x > 0.0f ? 4.88786f*0.5f*M_PI_F/ir.x : 2.0f*M_PI_F;
	float alpha1 = ir.y > 3.62144418f ? atan(sqrt(1.0f/(q1-1.0f))) : ir.y > 0.0f ?  3.621444f*0.5f*M_PI_F/ir.y : 2.0f*M_PI_F;
	float alpha2 = ir.z > 6.19469391f ? atan(sqrt(1.0f/(q2-1.0f))) : ir.z > 0.0f ? 6.19469391f*0.5f*M_PI_F/ir.z : 2.0f*M_PI_F;

	alpha0 = alpha0 < 0.001f ? 0.001f: alpha0;
	alpha1 = alpha1 < 0.001f ? 0.001f: alpha1;
	alpha2 = alpha2 < 0.001f ? 0.001f: alpha2;
	
	*var0 = alpha0;
	*var1 = alpha1;
	*var2 = alpha2;

}

void phaseIrUnWrapper(float t0, float t1,float t2, float3 ir, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
	float w1,w2,w3;
	calculatePhaseUnwrappingWeightsQuad(ir,&w1,&w2,&w3);
 	float err;
  float err1,err2,err3;
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
}

void phaseUnWrapper2(float t0, float t1,float t2, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
	float err, err0,err1,err2;
	float k,n,m;
	float t_star;

	float err_min=100000.0f;
	float err_min_second = 200000.0f;
	unsigned int ind_min, ind_second;

	float phi2_out;
	float phi1_out;
	float phi0_out;

	for(int i=0; i<30; i++)
	{
		m = m_list[i];
		n = n_list[i];
		k = k_list[i];
		phi2_out = (t2/2.0f+m);
		phi1_out = (t1/15.0f+k);
		phi0_out = (t0/3.0f+n);
		
		t_star = (phi2_out+phi1_out+phi0_out)/3.0f;

		err0 = moduloDistance(t0,10*t_star/9.0f,10)/3;
		err1 = moduloDistance(t1,2*t_star/9.0f,2)/15;
		err2 = moduloDistance(t2,15*t_star/9.0f,15)/2;
		//err0 = (t0-t_star)-3.0f*floor((t0-t_star)/3.0f);		
		//err1 = (t1-t_star)-15.0f*floor((t1-t_star)/15.0f);
		//err2 = (t2-t_star)-2.0f*floor((t2-t_star)/2.0f);

		//err = pow(9.0f*phi2_out/15.0f-t_star,2.0f)/2.0f+pow(9.0f*phi1_out/2.0f-t_star,2.0f)/15.0f+pow(9.0f*phi0_out/10.0f-t_star,2.0f)/3.0f;		
		err = err0*err0+err1*err1+err2*err2;

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

	phi2_out = (t2/2.0f+mvals);
	phi1_out = (t1/15.0f+kvals);
	phi0_out = (t0/3.0f+nvals);

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
}

void phaseUnWrapperVonMises(float3 phase, float* phase_first, float* phase_second, float* err_w1, float* err_w2)
{
  float err;
  float err1,err2,err3;
  //unsigned int ind_count = 1;
	
	float err_max = -1.0f;
	float err_max_second = -2.0f;
	unsigned int ind_max, ind_second;
	float kp = 2.0f;
	float k,n,m;
	
	float phi2_out;
	float phi1_out;
	float phi0_out;
	float t_star;
	float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

	for(int i=0; i<30; i++)
	{
		m = m_list[i];
		n = n_list[i];
		k = k_list[i];

		phi2_out = (t.z/2.0f+m);
		phi1_out = (t.y/15.0f+k);
		phi0_out = (t.x/3.0f+n);
		
		t_star = (phi2_out+phi1_out+phi0_out)/27.0f;
		err = exp(kp*cos(20.0*M_PI_F*t_star-phase.x))*exp(kp*cos(4.0*M_PI_F*t_star-phase.y))*exp(kp*cos(30*M_PI_F*t_star-phase.z));
		if(err>err_max)
		{
			err_max_second = err_max;
			ind_second = ind_max;
			err_max = err;
			ind_max = i;

		}
		else if(err>err_max_second)
		{
    	err_max_second = err;
			ind_second = i;
		}
		
	}

	//decode ind_min
	float mvals = m_list[ind_max];
	float nvals = n_list[ind_max];
	float kvals = k_list[ind_max];

	phi2_out = (t.z/2.0f+mvals);
	phi1_out = (t.y/15.0f+nvals);
	phi0_out = (t.x/3.0f+kvals);

	*err_w1 = err_max/400.0f;

	*phase_first = (phi2_out+phi1_out+phi0_out)/3.0f;


	mvals = m_list[ind_second];
	nvals = n_list[ind_second];
	kvals = k_list[ind_second];

	phi2_out = (t.z/2.0f+mvals);
	phi1_out = (t.y/15.0f+nvals);
	phi0_out = (t.x/3.0f+kvals);

	*err_w2 = err_max_second/400.0f;
	*phase_second = (phi2_out+phi1_out+phi0_out)/3.0f;	

}

void kernel processPixelStage2_phase(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float* conf1, global float* conf2, global float *ir_sums)
{
  const uint i = get_global_id(0);
  float3 a = a_in[i];
  float3 b = b_in[i];
	
  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));
  phase = select(phase, (float3)(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;
	ir = select(ir, (float3)(0.0001f), isnan(ir));
	
  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

	float phase_first = 0.0;
	float phase_second = 0.0;
	float conf = 1.0;
	float w_err1, w_err2;

	float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

	float t0 = t.x;
	float t1 = t.y;
	float t2 = t.z;

  phaseUnWrapper(t0, t1, t2, &phase_first, &phase_second, &w_err1, &w_err2);

	float std0,std1,std2;
	float phase_conf;
	if(ir_sum < 0.4f*65535.0f)
	{
		calculatePhaseUnwrappingVarDirect(ir, &std0, &std1, &std2);
		phase_conf = exp(-(std0*std0+std1*std1+std2*std2)/(2.0f*3.0f));
		phase_conf = select(phase_conf, 0.0f, isnan(phase_conf));
	}
	else
	{
		phase_conf = 0.0f;
	}
	w_err1 = phase_conf*exp(-w_err1/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err2 = phase_conf*exp(-w_err2/(2*CHANNEL_CONFIDENCE_SCALE));

	//suppress confidence if phase is beyond allowed range
	//w_err1 = phase_first > MAX_DEPTH*9.0f/18750.0f ? 0.0f: w_err1;
	//w_err2 = phase_second > MAX_DEPTH*9.0f/18750.0f ? 0.0f: w_err2;

	phase_first = 0.0f < phase_first ? phase_first + PHASE_OFFSET : phase_first;
	phase_second = 0.0f < phase_second ? phase_second + PHASE_OFFSET : phase_second;
	phase_1[i] = phase_first;//<30.0f ? phase_first : 0.0f; //conf > 0.9 ? phase_second : phase_first;
	phase_2[i] = phase_second;
	conf1[i] = w_err1;//select(0.0f, 1.0f, isnan(phase_conf));//phase_conf;
	conf2[i] = w_err2;
	ir_sums[i] = ir_sum;
}

void kernel filter_kde(global const float *phase_1, global const float *phase_2, global const float* conf1, global const float* conf2, global const float* gauss_filt_array, global const float* x_table, global const float* z_table, global float* depth, global float* ir_sums)
{
	const uint i = get_global_id(0);
	float kde_val_1, kde_val_2;

	const int loadX = i % 512;
	const int loadY = i / 512;
	int k, l;
  float sum_1, sum_2;
	
	int from_x = (loadX > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadX+1);
	int from_y = (loadY > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadY+1);
	int to_x = (loadX < 511-CHANNEL_FILT_SIZE-1 ? CHANNEL_FILT_SIZE: 511-loadX-1);
	int to_y = (loadY < 423-CHANNEL_FILT_SIZE ? CHANNEL_FILT_SIZE: 423-loadY);
  //compute
	float divby = (float)((to_x-from_x+1)*(to_y-from_y+1));
  kde_val_1 = 0.0f;
	kde_val_2 = 0.0f;
	float phase_first = phase_1[i];
	float phase_second = phase_2[i];
	float sum_gauss = 0.0f;
	if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
  // Filter kernel
    sum_1=0.0f;
		sum_2=0.0f;
		float gauss;
		
		float phase_1_local;
		float phase_2_local;
		float conf1_local;
		float conf2_local;
		uint ind;
		for(k=from_y; k<=to_y; k++)
		  for(l=from_x; l<=to_x; l++)
	    {
				ind = (loadY+k)*512+(loadX+l);
				conf1_local = conf1[ind];
				conf2_local = conf2[ind];
				phase_1_local = phase_1[ind];
				phase_2_local = phase_2[ind];
				gauss = gauss_filt_array[k+CHANNEL_FILT_SIZE]*gauss_filt_array[l+CHANNEL_FILT_SIZE];
				sum_gauss += gauss*(conf1_local+conf2_local);
		    sum_1 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_first,2)/(2*KDE_SIGMA_SQR))+conf2_local*exp(-pow(phase_2_local-phase_first, 2)/(2*KDE_SIGMA_SQR)));
				sum_2 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_second, 2)/(2*KDE_SIGMA_SQR))+conf2_local*exp(-pow(phase_2_local-phase_second, 2)/(2*KDE_SIGMA_SQR)));
	    }
		kde_val_1 = sum_gauss > 0.5f ? sum_1/sum_gauss : sum_1*2.0f;
		kde_val_2 = sum_gauss > 0.5f ? sum_2/sum_gauss : sum_2*2.0f;
  }
	int val_ind = kde_val_2 <= kde_val_1 ? 1: 0;
	//int val_ind = isless(val2,val1);
	float phase_final = val_ind ? phase_first: phase_second;
	float max_val = val_ind ? kde_val_1: kde_val_2;
	
	float zmultiplier = z_table[i];
	float xmultiplier = x_table[i];

  float depth_linear = zmultiplier * phase_final;
  float max_depth = phase_final * UNAMBIGIOUS_DIST * 2.0;

  bool cond1 =  true && 0.0f < depth_linear && 0.0f < max_depth;

  xmultiplier = (xmultiplier * 90.0) / (max_depth * max_depth * 8192.0);

  float depth_fit = depth_linear / (-depth_linear * xmultiplier + 1);
  depth_fit = depth_fit < 0.0f ? 0.0f : depth_fit;

  float d = cond1 ? depth_fit : depth_linear; // r1.y -> later r2.z
	//float q = conf2[i]/conf1[i];
	max_val = d < MIN_DEPTH || d > MAX_DEPTH ? 0.0f: max_val;
	float ir_conf = 0.000001f*max_val*d*d;
  //depth[i] = max_val >= 0.29f ? DEPTH_SCALE*d: 0.0f;
	//depth[i] = max_val >= 0.37f ? DEPTH_SCALE*d: 0.0f;
	//depth[i] = max_val >= 0.3366 ? DEPTH_SCALE*d: 0.0f;
	//depth[i] = max_val >2.0*0.4121 ? d: 0.0f;
	depth[i] = d;//conf2[i];
	depth[i+512*424] = max_val;//0.001f*conf1[i];//max_val;//conf1[i];//- conf2[i]; 
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


void kernel processPixelStage2_phase3(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float *phase_3, global float *conf1, global float *conf2, global float *conf3, global float *ir_sums)
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

	float var0,var1,var2;
	calculatePhaseUnwrappingVar(ir, &var0, &var1, &var2);
	float phase_conf = exp(-(var0*var0+var1*var1+var2*var2)/(2.0f*3.0f));//(1.0f/var0)*(1.0f/var1)*(1.0f/var2);//1.0f/(var0+var1+var2);//exp(-(var0+var1+var2)/(6.0f*0.2f));
	phase_conf = select(phase_conf, 0.0f, isnan(phase_conf));

	//float phase_conf = phaseConfidenceA(ir);
	w_err1 = phase_conf*exp(-w_err1/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err2 = phase_conf*exp(-w_err2/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err3 = phase_conf*exp(-w_err3/(2*CHANNEL_CONFIDENCE_SCALE));

	w_err1 = phase_first > MAX_DEPTH*9.0f/18750.0f || phase_first < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err1;
	w_err2 = phase_second > MAX_DEPTH*9.0f/18750.0f || phase_second < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err2;
	w_err3 = phase_third > MAX_DEPTH*9.0f/18750.0f || phase_third < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err3;

	conf1[i] = w_err1;
	conf2[i] = w_err2;
	conf3[i] = w_err3;

	ir_sums[i] = ir_sum;
}



void kernel filter_kde3(global const float *phase_1, global const float *phase_2, global const float *phase_3, global const float* conf1, global const float* conf2, global const float* conf3, global const float* gauss_filt_array, global const float* x_table, global const float* z_table, global float* depth)
{
	const uint i = get_global_id(0);
	float kde_val_1, kde_val_2, kde_val_3;

	const int loadX = i % 512;
	const int loadY = i / 512;
	int k, l;
  float sum_1, sum_2, sum_3;
	
	int from_x = (loadX > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadX+1);
	int from_y = (loadY > CHANNEL_FILT_SIZE ? -CHANNEL_FILT_SIZE : -loadY+1);
	int to_x = (loadX < 511-CHANNEL_FILT_SIZE-1 ? CHANNEL_FILT_SIZE: 511-loadX-1);
	int to_y = (loadY < 423-CHANNEL_FILT_SIZE ? CHANNEL_FILT_SIZE: 423-loadY);
  //compute
	float divby = (float)((to_x-from_x+1)*(to_y-from_y+1));
  kde_val_1 = 0.0f;
	kde_val_2 = 0.0f;
	kde_val_3 = 0.0f;
	float phase_first = phase_1[i];
	float phase_second = phase_2[i];
	float phase_third = phase_3[i];
	if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
  // Filter kernel
    sum_1=0.0f;
		sum_2=0.0f;
		sum_3=0.0f;
		float gauss;
		float sum_gauss = 0.0f;
		
		float phase_1_local;
		float phase_2_local;
		float phase_3_local;
		float conf1_local;
		float conf2_local;
		float conf3_local;
		uint ind;
		for(k=from_y; k<=to_y; k++)
		  for(l=from_x; l<=to_x; l++)
	    {
				ind = (loadY+k)*512+(loadX+l);
				conf1_local = conf1[ind];
				conf2_local = conf2[ind];
				conf3_local = conf3[ind];
				phase_1_local = phase_1[ind];
				phase_2_local = phase_2[ind];
				phase_3_local = phase_3[ind];
				gauss = gauss_filt_array[k+CHANNEL_FILT_SIZE]*gauss_filt_array[l+CHANNEL_FILT_SIZE];
				sum_gauss += gauss*(conf1_local+conf2_local+conf3_local);
		    sum_1 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_first,2)/(2*KDE_SIGMA_SQR))+conf2_local*exp(-pow(phase_2_local-phase_first, 2)/(2*KDE_SIGMA_SQR))+conf3_local*exp(-pow(phase_3_local-phase_first,2)/(2*KDE_SIGMA_SQR)));
				sum_2 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_second, 2)/(2*KDE_SIGMA_SQR))+conf2_local*exp(-pow(phase_2_local-phase_second, 2)/(2*KDE_SIGMA_SQR))+conf3_local*exp(-pow(phase_3_local-phase_second,2)/(2*KDE_SIGMA_SQR)));
				sum_3 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_third, 2)/(2*KDE_SIGMA_SQR))+conf2_local*exp(-pow(phase_2_local-phase_third, 2)/(2*KDE_SIGMA_SQR))+conf3_local*exp(-pow(phase_3_local-phase_third,2)/(2*KDE_SIGMA_SQR)));
	    }
		kde_val_1 = sum_gauss > 0.5f ? sum_1/sum_gauss : sum_1*2.0f;
		kde_val_2 = sum_gauss > 0.5f ? sum_2/sum_gauss : sum_2*2.0f;
		kde_val_3 = sum_gauss > 0.5f ? sum_3/sum_gauss : sum_3*2.0f;
  }
	
	float phase_final, max_val;
	if(kde_val_2 > kde_val_1 || kde_val_3 > kde_val_1)
	{
		if(kde_val_3 > kde_val_2)
		{
			phase_final = phase_third;
			max_val = kde_val_3;
		}
		else
		{
			phase_final = phase_second;
			max_val = kde_val_2;
		}
	}
	else
	{
		phase_final = phase_first;
		max_val = kde_val_1;
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
    //depth[i] = max_val >= 0.6f ? DEPTH_SCALE*d: 0.0f;
	depth[i] = d;
	depth[i+512*424] = max_val; 
}



void kernel mapColorToDepth(global const float *depth, global const float *undist_map, global const float *rgb_camera_intrinsics, global const float *rel_rot, global const float *rel_trans, global int* rgb_index)
{
	const uint i = get_global_id(0);

	float fx_rgb = rgb_camera_intrinsics[0];
	float fy_rgb = rgb_camera_intrinsics[1];
	float cx_rgb = rgb_camera_intrinsics[2];
	float cy_rgb = rgb_camera_intrinsics[3];

	float k1 = rgb_camera_intrinsics[4];
	float k2 = rgb_camera_intrinsics[5];
	float k3 = rgb_camera_intrinsics[6];

	float d = depth[i];
	if(d < 0.1f)
	{
		rgb_index[2*i] = -1;
		rgb_index[2*i+1] = -1;
		return;
	}
	float3 vert = (float3)(undist_map[2*i]*d, undist_map[2*i+1]*d, d);
	
	float3 rgb_vert = (float3)(rel_rot[0]*vert.x+rel_rot[1]*vert.y+rel_rot[2]*vert.z,rel_rot[3]*vert.x+rel_rot[4]*vert.y+rel_rot[5]*vert.z,rel_rot[6]*vert.x+rel_rot[7]*vert.y+rel_rot[8]*vert.z);
	
	rgb_vert+=(float3)(rel_trans[0],rel_trans[1],rel_trans[2]);
	
	rgb_vert/=rgb_vert.z;
	float r = sqrt(rgb_vert.x*rgb_vert.x+rgb_vert.y*rgb_vert.y);
	float dist = (1.0+k1*r*r+k2*r*r*r*r+k3*r*r*r*r*r*r);
	rgb_vert.x = rgb_vert.x*dist;
	rgb_vert.y = rgb_vert.y*dist;
	float3 rgb_vert_proj = (float3)(fx_rgb*rgb_vert.x+cx_rgb, fy_rgb*rgb_vert.y+cy_rgb, 1.0f);

	//rgb_index[i] = (float2)(rgb_vert_proj.x/1930.0f,rgb_vert_proj.y/1080.0f);
	rgb_index[2*i] = (int)(round(rgb_vert_proj.x));
	rgb_index[2*i+1] = (int)(round(rgb_vert_proj.y));
	
}

void kernel filterPixelStage2(global const float *depth, global const float *ir_sums, global const uchar *max_edge_test, global float *filtered)
{
  const uint i = get_global_id(0);

  const uint x = i % 512;
  const uint y = i / 512;

  const float raw_depth = depth[i];
  const float ir_sum = ir_sums[i];
  const uchar edge_test = max_edge_test[i];

  if(raw_depth >= MIN_DEPTH && raw_depth <= MAX_DEPTH)
  {
    if(x < 1 || y < 1 || x > 510 || y > 422)
    {
      filtered[i] = raw_depth;
    }
    else
    {
      float ir_sum_acc = ir_sum;
      float squared_ir_sum_acc = ir_sum * ir_sum;
      float min_depth = raw_depth;
      float max_depth = raw_depth;

      for(int yi = -1; yi < 2; ++yi)
      {
        uint i_other = (y + yi) * 512 + x - 1;

        for(int xi = -1; xi < 2; ++xi, ++i_other)
        {
          if(i_other == i)
          {
            continue;
          }

          const float raw_depth_other = depth[i_other];
          const float ir_sum_other = ir_sums[i_other];

          ir_sum_acc += ir_sum_other;
          squared_ir_sum_acc += ir_sum_other * ir_sum_other;

          if(0.0f < raw_depth_other)
          {
            min_depth = min(min_depth, raw_depth_other);
            max_depth = max(max_depth, raw_depth_other);
          }
        }
      }

      float tmp0 = sqrt(squared_ir_sum_acc * 9.0f - ir_sum_acc * ir_sum_acc) / 9.0f;
      float edge_avg = max(ir_sum_acc / 9.0f, EDGE_AB_AVG_MIN_VALUE);
      tmp0 /= edge_avg;

      float abs_min_diff = fabs(raw_depth - min_depth);
      float abs_max_diff = fabs(raw_depth - max_depth);

      float avg_diff = (abs_min_diff + abs_max_diff) * 0.5f;
      float max_abs_diff = max(abs_min_diff, abs_max_diff);

      bool cond0 =
          0.0f < raw_depth &&
          tmp0 >= EDGE_AB_STD_DEV_THRESHOLD &&
          EDGE_CLOSE_DELTA_THRESHOLD < abs_min_diff &&
          EDGE_FAR_DELTA_THRESHOLD < abs_max_diff &&
          EDGE_MAX_DELTA_THRESHOLD < max_abs_diff &&
          EDGE_AVG_DELTA_THRESHOLD < avg_diff;

      if(!cond0)
      {
        if(edge_test != 0)
        {
          float tmp1 = 1500.0f > raw_depth ? 30.0f : 0.02f * raw_depth;
          float edge_count = 0.0f;

          filtered[i] = edge_count > MAX_EDGE_COUNT ? 0.0f : DEPTH_SCALE*raw_depth;
        }
        else
        {
          filtered[i] = 0.0f;
        }
      }
      else
      {
        filtered[i] = 0.0f;
      }
    }
  }
  else
  {
    filtered[i] = 0.0f;
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


