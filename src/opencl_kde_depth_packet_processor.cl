
/*******************************************************************************
 * Process pixel stage 1
 ******************************************************************************/

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
                               global float3 *a_out, global float3 *b_out, global float3 *n_out, global float *ir_out, global float3 *c_out)
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

		float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;
		float3 c = (float3)(v0.x+v0.y+v0.z,v1.x+v1.y+v1.z,v2.x+v2.y+v2.z);
		//uint i_out = 512*y+(511-x);
    a_out[i] = a;
    b_out[i] = b;
    n_out[i] = n;
		//ir_out[i] = min(dot(select(n, (float3)(65535.0f), saturated), (float3)(0.333333333f  * AB_MULTIPLIER * AB_OUTPUT_MULTIPLIER)), 65535.0f);

		ir_out[i] = ir.x;
		ir_out[i+512*424] = ir.y;
		ir_out[i+512*424*2] = ir.z;
		c_out[i] = 0.33333333f*fabs(c);
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
	
	//float std0 = 3*exp(1.2725f-ir.x*0.808f);
	//float std1 = 15*exp(-1.8149f-ir.y*0.0370f);
	//float std2 = 2*exp(0.8242f-ir.z*0.0865f);

	//float w1 = 1.0f/(std0*std0+std1*std1);
	//float w2 = 1.0f/(std0*std0+std2*std2);//366.2946;
	//float w3 = 1.0f/(std1*std1+std2*std2);
	
	float w1 = 1.0f;
	float w2 = 18.0f;
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



void kernel processPixelStage2_phase(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float* conf1, global float* conf2, global float *ir_sums)
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

	float phase_first = 0.0;
	float phase_second = 0.0;
	float conf = 1.0;
	float w_err1, w_err2;
  //if(ir_min >= INDIVIDUAL_AB_THRESHOLD && ir_sum >= AB_THRESHOLD)
 	//{
		float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

		float t0 = t.x;
		float t1 = t.y;
		float t2 = t.z;
		//libfreenect2_unwrapper(t0, t1, t2, ir_min, ir_max, &phase_first);
		phaseUnWrapper(t0, t1, t2, &phase_first, &phase_second, &w_err1, &w_err2);
  //}
	float phase_conf = phaseConfidenceA(ir);
	//float phase_conf = phaseConfidenceAO(ir,c);
	//float phase_conf = phaseConfidenceAoverO(ir,c);
	w_err1 = phase_conf*exp(-w_err1/(2*CHANNEL_CONFIDENCE_SCALE));
	w_err2 = phase_conf*exp(-w_err2/(2*CHANNEL_CONFIDENCE_SCALE));

	w_err1 = phase_first > MAX_DEPTH*9.0f/18750.0f || phase_first < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err1;
	w_err2 = phase_second > MAX_DEPTH*9.0f/18750.0f || phase_second < MIN_DEPTH*9.0f/18750.0f ? 0.0f: w_err2;

	phase_first = 0.0f < phase_first ? phase_first + PHASE_OFFSET : phase_first;
	phase_second = 0.0f < phase_second ? phase_second + PHASE_OFFSET : phase_second;
	phase_1[i] = phase_first;//<30.0f ? phase_first : 0.0f; //conf > 0.9 ? phase_second : phase_first;
	phase_2[i] = phase_second;
	conf1[i] = w_err1;
	conf2[i] = w_err2;
	ir_sums[i] = ir_sum;
}

void kernel kde_filter(global const float *phase_1, global const float *phase_2, global const float* conf1, global const float* conf2, global float* kde_val_1, global float* kde_val_2, global const float* gauss_filt_array)
{
	const uint i = get_global_id(0);
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
  kde_val_1[i] = 0.0f;
	kde_val_2[i] = 0.0f;

	if(loadX >= 1 && loadX < 511 && loadY >= 0 && loadY<424)
  {
  // Filter kernel
    sum_1=0.0f;
		sum_2=0.0f;
		float gauss;
		float sum_gauss = 0.0f;
		float phase_first = phase_1[i];
		float phase_second = phase_2[i];
		float phase_1_local;
		float phase_2_local;
		float conf1_local;
		float conf2_local;
		uint ind;
		float sigma_sqr = 0.25f*0.07884864f;
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
		    sum_1 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_first,2)/(2*sigma_sqr))+conf2_local*exp(-pow(phase_2_local-phase_first, 2)/(2*sigma_sqr)));
				sum_2 += gauss*(conf1_local*exp(-pow(phase_1_local-phase_second, 2)/(2*sigma_sqr))+conf2_local*exp(-pow(phase_2_local-phase_second, 2)/(2*sigma_sqr)));
				//sumx += channels[((loadY+k)*512+(loadX+l))];
	    }
		kde_val_1[i] = sum_1/sum_gauss;
		kde_val_2[i] = sum_2/sum_gauss;
    //channels_filtered[(loadY*512+loadX)] = sumx/divby;
  }
  
}



void kernel processPixelStage2_depth(global const float* phase_1, global const float* phase_2, global const float* kde_val_1, global const float* kde_val_2, global const float *x_table, global const float *z_table, global float* depth)
{
	const uint i = get_global_id(0);


	float phase_final = kde_val_2[i] > kde_val_1[i] ? phase_2[i]: phase_1[i];
	float max_val = kde_val_2[i] > kde_val_1[i] ? kde_val_2[i]: kde_val_1[i];
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
  depth[i] = max_val >= 0.5f ? DEPTH_SCALE*depth_linear: 0.0f;
	//depth[i] = depth_fit;
	depth[i+512*424] = max_val;
}

/*
void kernel outProjectAndMapToRgb(global const float *depth, global const float *ir_camera_intrinsics, global const float *rgb_camera_intrinsics, global const float *rel_rot, global const float *rel_trans, global float3 *depth_out, global float2* rgb_index)
{
	const uint i = get_global_id(0);

  const float x = (float)(i % 512);
  const float y = (float)(i / 512);

	float fx_ir = ir_camera_intrinsics[0];
	float fy_ir = ir_camera_intrinsics[1];
	float cx_ir = ir_camera_intrinsics[2];
	float cy_ir = ir_camera_intrinsics[3];
	float fx_ir_inv = 1.0f/fx_ir; 
	float fy_ir_inv = 1.0f/fy_ir;

	float cx_ir_inv = -cx_ir/fx_ir;
	float cy_ir_inv = -cy_ir/fy_ir;

	float fx_rgb = rgb_camera_intrinsics[0];
	float fy_rgb = rgb_camera_intrinsics[1];
	float cx_rgb = rgb_camera_intrinsics[2];
	float cy_rgb = rgb_camera_intrinsics[3];

	float3 vert = (float3)(fx_ir_inv*x+cx_ir_inv, fy_ir_inv*y+cy_ir_inv, 1.0f);
	vert *= depth[i];
	//depth_out[i].z = 0.0f ; //2.0f;
	//depth_out[i].xy = (float2)(x/512.0f,y/424.0f);
	
	float3 rgb_vert = (float3)(rel_rot[0]*vert.x+rel_rot[1]*vert.y+rel_rot[2]*vert.z,rel_rot[3]*vert.x+rel_rot[4]*vert.y+rel_rot[5]*vert.z,rel_rot[6]*vert.x+rel_rot[7]*vert.y+rel_rot[8]*vert.z);
	//float3 rgb_vert = vert;
	rgb_vert+=(float3)(rel_trans[0],rel_trans[1],rel_trans[2]);
	rgb_vert/=rgb_vert.z;
	float3 rgb_vert_proj = (float3)(fx_rgb*rgb_vert.x+cx_rgb, fy_rgb*rgb_vert.y+cy_rgb, 1.0f);

	//rgb_index[i] = (float2)(rgb_vert_proj.x/1930.0f,rgb_vert_proj.y/1080.0f);
	rgb_index[i] = (float2)(rgb_vert_proj.x/1930.0f,rgb_vert_proj.y/1080.0f);
	
	//float k1 = camera_intrinsics[4];
	//float k2 = camera_intrinsics[5];
	//float k3 = camera_intrinsics[6];
	
	vert.y *=-1.0f;
	depth_out[i] = 0.01f*vert;

}*/





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

/*****************************************************************
 * THREE HYPOTHESIS
 *****************************************************************/

void phaseUnWrapper3(float t0, float t1,float t2, float* phase_first, float* phase_second, float* phase_third, float* err_w1, float* err_w2, float* err_w3)
{
  float err;
  float err1,err2,err3;
  //unsigned int ind_count = 1;
	//float k,nc,nf;
	float w1 = 0.7007;
	float w2 = 366.2946;
	float w3 = 0.7016;
	
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

void kernel processPixelStage2_phase3(global const float3 *a_in, global const float3 *b_in, global float *phase_1, global float *phase_2, global float *phase_3, global float *phase_out_vertical, global float *phase_out_horizontal, global float* w1, global float* w2, global float* w3, global float* cost_vertical, global float* cost_horizontal, global float* conf1, global float* conf2, global float *ir_sums)
{
  const uint i = get_global_id(0);
  float3 a = a_in[i];
  float3 b = b_in[i];
	
  float3 phase = atan2(b, a);
  phase = select(phase, phase + 2.0f * M_PI_F, isless(phase, (float3)(0.0f)));
  phase = select(phase, (float3)(0.0f), isnan(phase));
  float3 ir = sqrt(a * a + b * b) * AB_MULTIPLIER;

  float ir_sum = ir.x + ir.y + ir.z;
  float ir_min = min(ir.x, min(ir.y, ir.z));
  float ir_max = max(ir.x, max(ir.y, ir.z));

	float phase_first = 0.0;
	float phase_second = 0.0;
	float phase_third = 0.0;
	float conf = 1.0;
	float w_err1, w_err2, w_err3, conf1_tmp, conf2_tmp;
  //if(ir_min >= INDIVIDUAL_AB_THRESHOLD && ir_sum >= AB_THRESHOLD)
 	//{
		float3 t = phase / (2.0f * M_PI_F) * (float3)(3.0f, 15.0f, 2.0f);

		float t0 = t.x;
		float t1 = t.y;
		float t2 = t.z;
		//libfreenect2_unwrapper(t0, t1, t2, ir_min, ir_max, &phase_first);
		phaseUnWrapper3(t0, t1, t2, &phase_first, &phase_second, &phase_third, &w_err1, &w_err2, &w_err3);
  //}
	w1[i] = w_err1;
	w2[i] = w_err2;
	w3[i] = w_err3;
	cost_vertical[i] = 0.0f;
	cost_horizontal[i] = 0.0f;
	conf1[i] = 1.0f;//w_err1;//ir_min*1/w_err1;//ir_min*ir_min;
	conf2[i] = 1.0f;//w_err1;//w_err1;//ir_min*1/w_err1;//ir_min*ir_min;
	phase_first = 0.0f < phase_first ? phase_first + PHASE_OFFSET : phase_first;
	phase_second = 0.0f < phase_second ? phase_second + PHASE_OFFSET : phase_second;
	phase_third = 0.0f < phase_third ? phase_third + PHASE_OFFSET : phase_third;
	phase_1[i] = phase_first;//<30.0f ? phase_first : 0.0f; //conf > 0.9 ? phase_second : phase_first;
	phase_2[i] = phase_second;
	phase_3[i] = phase_third;
	phase_out_vertical[i] = phase_first;
	phase_out_horizontal[i] = phase_first;

	ir_sums[i] = ir_sum;
}
void kernel propagateVertical3(global const float* phase_1, global const float* phase_2, global const float* phase_3, global const float* w1, global const float* w2, global const float* w3, global const float* conf1, global const float* conf2, global const uint* count, global float* phase_out, global float* cost)
{
    const uint i = get_global_id(0);
    const int row = i % 424;
    const int dir = i>423 ? 1:-1;
    const int col = 256+dir*count[0];
   
    float phi_1 = phase_1[row*512+col];
    float phi_2 = phase_2[row*512+col];
		float phi_3 = phase_3[row*512+col];  
    float w_phi_1 = w1[row*512+col];
    float w_phi_2 = w2[row*512+col];
		float w_phi_3 = w3[row*512+col];

		if(phi_1 == 0.0f)
			return;
		//float conf_phi_1 = conf1[row*512+col];
		//float conf_phi_2 = conf2[row*512+col];

		float conf1_first = i>0 ? conf1[(row-1)*512+col-dir] : 0.0f;
		float conf1_second = conf1[row*512+col-dir];
		float conf1_third = row < 423 ? conf1[(row+1)*512+col-dir] : 0.0f;

		//float conf_first = i>0 ? w_quota[(row-1)*512+col-dir] : 0.0f;
		//float conf_second = w_quota[row*512+col-dir];
		//float conf_third = row < 423 ? w_quota[(row+1)*512+col-dir] : 0.0f;

    /*float w1_first = i>0 ? w1[(row-1)*512+col-dir] : 0.0f;
    float w1_second = w1[row*512+col-dir];
    float w1_third = row < 423 ? w1[(row+1)*512+col-dir] : 0.0f;*/
		
    float phase_out_first = row>0 ? phase_out[(row-1)*512+col-dir] : 0.0f;
    float phase_out_second = phase_out[row*512+col-dir];
    float phase_out_third = row < 423 ? phase_out[(row+1)*512+col-dir] : 0.0f;
    uint mask1 = phase_out_first > 0.0f;
		uint mask2 = phase_out_second > 0.0f;
		uint mask3 = phase_out_third > 0.0f;  

 		if(mask1 == 0 && mask2 == 0 && mask3 == 0)
		{
			cost[row*512+col] = 1000.0f;
			return;
		}

		float sum_masks = (float)(INV_SQRT_2*mask1+mask2+INV_SQRT_2*mask3);

		float tot_cost = mask2*cost[row*512+col-dir] + mask1*INV_SQRT_2*(row>0 ? cost[(row-1)*512+col-dir]: 0.0f) + mask3*INV_SQRT_2*(row<423 ? cost[(row+1)*512+col-dir] : 0.0f); 
		tot_cost /= sum_masks;

		float diff1 = INV_SQRT_2/conf1_first*mask1*(phi_1-phase_out_first)*(phi_1-phase_out_first);
		float diff2 = 1.0f/conf1_second*mask2*(phi_1-phase_out_second)*(phi_1-phase_out_second);
		float diff3 = INV_SQRT_2/conf1_third*mask3*(phi_1-phase_out_third)*(phi_1-phase_out_third);
		float val1 = 0.0f;

  	val1 =  row>0 ? diff1 : 0.0f;
  	val1 += diff2;
  	val1 += row < 423 ? diff3 : 0.0f;

		val1 *= w_phi_1;
			
		diff1 = INV_SQRT_2/conf1_first*mask1*(phi_2-phase_out_first)*(phi_2-phase_out_first);
		diff2 = 1.0f/conf1_second*mask2*(phi_2-phase_out_second)*(phi_2-phase_out_second);
		diff3 = INV_SQRT_2/conf1_third*mask3*(phi_2-phase_out_third)*(phi_2-phase_out_third);
		float val2 = 0.0f;

  	val2 =  row>0 ? diff1 : 0.0f;
  	val2 += diff2;
  	val2 += row < 423 ? diff3 : 0.0f;
	
		val2 *= w_phi_2;
 		
		diff1 = INV_SQRT_2/conf1_first*mask1*(phi_3-phase_out_first)*(phi_3-phase_out_first);
		diff2 = 1.0f/conf1_second*mask2*(phi_3-phase_out_second)*(phi_3-phase_out_second);
		diff3 = INV_SQRT_2/conf1_third*mask3*(phi_3-phase_out_third)*(phi_3-phase_out_third);
		float val3 = 0.0f;

  	val3 =  row>0 ? diff1 : 0.0f;
  	val3 += diff2;
  	val3 += row < 423 ? diff3 : 0.0f;
	
		val3 *= w_phi_3;
		//float val_max = val1 < val2 ? val2: val1;
		//float val_min = val1 < val2 ? val1: val2;
		//w_quota[i] = val_min > 0.0f ? val_max / val_min: 10.0f;
		//w_quota[row*512+col] = val1 < val2 ? 1.0f : 0.0f;
		//uint mask = w_quota[row*512+col] < 8.0f ? 0: 1;
		sum_masks = (float)(INV_SQRT_2*mask1/conf1_first+mask2/conf1_second+INV_SQRT_2*mask3/conf1_third);
    phase_out[row*512+col] = select(phi_1, select(phi_2, phi_3, val3 < val2), select(val2, val3, val3 < val2) < val1);//(val1 < val2 ? phi_1 : phi_2);
		cost[row*512+col] = min(val1, min(val2,val3))/(sum_masks); //val1 < val2 ? val1 : val2;
}

void kernel propagateHorizontal3(global const float* phase_1, global const float* phase_2, global const float* phase_3, global const float* w1, global const float* w2, global const float* w3, global const float* conf1, 
global const float* conf2, global const uint* count, global float* phase_out, global float* cost)
{
    const uint i = get_global_id(0);
    const int col = i % 512;
    const int dir = i>511 ? 1:-1;
    const int row = 212+dir*count[0];
   
    float phi_1 = phase_1[row*512+col];
    float phi_2 = phase_2[row*512+col];
		float phi_3 = phase_3[row*512+col];  
    float w_phi_1 = w1[row*512+col];
    float w_phi_2 = w2[row*512+col];
		float w_phi_3 = w3[row*512+col];

		if(phi_1 == 0.0f)
			return;
/*
		float w_first = i>0 ? w1[(row-dir)*512+col-1] : 0.0f;
		float w_second = w1[(row-dir)*512+col];
		float w_third = row < 423 ? w1[(row-dir)*512+col+1] : 0.0f;

		if(w_first > 60.0f && w_second > 60.0f && w_third > 60.0f)
			return;		
*/

		float conf1_first = col>0 ? conf2[(row-dir)*512+col-1] : 1.0f;
		float conf1_second = conf2[(row-dir)*512+col];
		float conf1_third = col < 511 ? conf2[(row-dir)*512+col+1] : 1.0f;

    float phase_out_first = col>0 ? phase_out[(row-dir)*512+col-1] : 0.0f;
    float phase_out_second = phase_out[(row-dir)*512+col];
    float phase_out_third = col < 511 ? phase_out[(row-dir)*512+col+1] : 0.0f;
    uint mask1 = phase_out_first > 0.0f;
		uint mask2 = phase_out_second > 0.0f;
		uint mask3 = phase_out_third > 0.0f;    
		
		if(mask1 == 0 && mask2 == 0 && mask3 == 0)
		{
			cost[row*512+col] = 1000.0f;
			return;
		}

		float sum_masks = (float)(INV_SQRT_2*mask1+mask2+INV_SQRT_2*mask3);
		float tot_cost = mask2*cost[(row-dir)*512+col] + mask1*INV_SQRT_2*(col>0 ? cost[(row-dir)*512+col-1] : 0.0f) + mask3*INV_SQRT_2*(col<511 ? cost[(row-dir)*512+col+1] : 0.0f); 
		tot_cost /= sum_masks;

		float diff1 = INV_SQRT_2/conf1_first*mask1*(phi_1-phase_out_first)*(phi_1-phase_out_first);
		float diff2 = 1.0f/conf1_second*mask2*(phi_1-phase_out_second)*(phi_1-phase_out_second);
		float diff3 = INV_SQRT_2/conf1_third*mask3*(phi_1-phase_out_third)*(phi_1-phase_out_third);
		float val1 = 0.0f;

  	val1 =  row>0 ? diff1 : 0.0f;
  	val1 += diff2;
  	val1 += row < 423 ? diff3 : 0.0f;

		val1 *= w_phi_1;
			
		diff1 = INV_SQRT_2/conf1_first*mask1*(phi_2-phase_out_first)*(phi_2-phase_out_first);
		diff2 = 1.0f/conf1_second*mask2*(phi_2-phase_out_second)*(phi_2-phase_out_second);
		diff3 = INV_SQRT_2/conf1_third*mask3*(phi_2-phase_out_third)*(phi_2-phase_out_third);
		float val2 = 0.0f;

  	val2 =  col>0 ? diff1 : 0.0f;
  	val2 += diff2;
  	val2 += col < 511 ? diff3 : 0.0f;
	
		val2 *= w_phi_2;
 		
		diff1 = INV_SQRT_2/conf1_first*mask1*(phi_3-phase_out_first)*(phi_3-phase_out_first);
		diff2 = 1.0f/conf1_second*mask2*(phi_3-phase_out_second)*(phi_3-phase_out_second);
		diff3 = INV_SQRT_2/conf1_third*mask3*(phi_3-phase_out_third)*(phi_3-phase_out_third);
		float val3 = 0.0f;

  	val3 =  col>0 ? diff1 : 0.0f;
  	val3 += diff2;
  	val3 += col < 511 ? diff3 : 0.0f;
	
		val3 *= w_phi_3;

		//float val_max = val1 < val2 ? val2: val1;
		//float val_min = val1 < val2 ? val1: val2;
		//w_quota[i] = val_min > 0.0f ? val_max / val_min: 10.0f;
		//w_quota[row*512+col] = val1 < val2 ? 1.0f : 0.0f;
		//uint mask = w_quota[row*512+col] < 8.0f ? 0: 1;
		sum_masks = (float)(INV_SQRT_2*mask1/conf1_first+mask2/conf1_second+INV_SQRT_2*mask3/conf1_third);
    phase_out[row*512+col] = select(phi_1, select(phi_2, phi_3, val3 < val2), select(val2, val3, val3 < val2) < val1);//(val1 < val2 ? phi_1 : phi_2);
		cost[row*512+col] = min(val1, min(val2,val3))/(sum_masks); //val1 < val2 ? val1 : val2;
}

