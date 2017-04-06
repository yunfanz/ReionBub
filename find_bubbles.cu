#define BLOCK_SIZE 8

// Convert 3D index to 1D index.
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))

// Texture memory for image.
texture<float,3> img;

// Step 1. Label local minima or flatland as PLATEAU
__global__ void real_tophat_kernel(float* ionized, const int w, float R, float S0)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  if (j >= w || i >= w || k >= w || ionized[p] == 1) return;

  float rsq;
  float deltasum = 0;
  float deltac = %(DELTAC)s;
  float smin = %(SMIN)s;
  int count = 0;
  for (int kk = 0; kk < w; kk++) {
  	for (int jj = 0; jj < w; jj++) {
  		for (int ii = 0; ii < w; ii++){
  			rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
  			if (rsq < R*R)
  			{
  				deltasum += tex3D(img,i,j,k);
  				count ++;
  			}
  		}
  	}
  }
  float delta0 = deltasum/count;
  float fcoll = 1 - erf((deltac - delta0)/sqrt(2*(smin - S0)));
  //ionized[p] = fcoll* %(ZETA)s;;
  if (fcoll >= 1/%(ZETA)s) ionized[p] = 1.0;
  else { ionized[p] = fcoll * %(ZETA)s; }
 }
 __global__ void k_tophat_kernel(float* ionized, const int w, float R, float S0)
{
	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	float ks = pow((9*3.14159/2),1.0/3) / R;
	if (j >= w || i >= w || k >= w || ionized[p] == 1) return;

	float rsq, r, y;
	float deltasum = 0;
	float deltac = %(DELTAC)s;
	float smin = %(SMIN)s;
	float count = 0;
	float W;
	for (int kk = 0; kk < w; kk++) {
		for (int jj = 0; jj < w; jj++) {
			for (int ii = 0; ii < w; ii++){
				rsq = (ii-i)*(ii-i)+(jj-j)*(jj-j)+(kk-k)*(kk-k);
				r = pow( rsq, float(0.5) );
				y = ks*r;
				W = (sin(y) - y*cos(y))/( 2*pow(3.14159,2)*pow(r,3) );
				deltasum += W * tex3D(img,i,j,k);
				count += W;
			}
		}
	}
	float delta0 = deltasum/count;
	float fcoll = 1 - erf((deltac - delta0)/sqrt(2*(smin - S0)));
	//ionized[p] = fcoll* %(ZETA)s;;
	if (fcoll >= 1/%(ZETA)s) ionized[p] = 1.0;
	else { ionized[p] = fcoll * %(ZETA)s; }
 }


__global__ void real_tophat(float* filter, int w, float R)
{
	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	if (j >= w || i >= w || k >= w) return;
	float rsq = (w/2-i)*(w/2-i)+(w/2-j)*(w/2-j)+(w/2-k)*(w/2-k);
	float vol = 4.0*3.1415926*R*R*R/3.0;
	if (rsq < R*R)
	{
		filter[p] = 1./vol;
	}
	else
	{
		filter[p] = 0;
	}
 }

 __global__ void k_tophat(float* filter, int w, float ks)
{
	int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
	int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
	int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
	int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
	int p = INDEX(k,j,i,w);
	if (j >= w || i >= w || k >= w) return;
	float ksq = i*i + j*j + k*k;
	float vol = 4.0*3.1415926*ks*ks*ks/3.0;
	if (ksq < ks*ks)
	{
		filter[p] = 1./vol;
	}
	else
	{
		filter[p] = 0;
	}
 }

__global__ void update_kernel(float* ionized, float* smoothed, const int w, float R, float S0)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
  int p = INDEX(k,j,i,w);
  if (j >= w || i >= w || k >= w || ionized[p] == 1) return;

  float delta0 = smoothed[p];
  float deltac = %(DELTAC)s;
  float smin = %(SMIN)s;
  float fcoll = 1 - erf((deltac - delta0)/sqrt(2*(smin - S0)));
  ionized[p] = fcoll;
  //if (fcoll >= 1/%(ZETA)s) ionized[p] = 1.0;
  //else if (R==%(RMIN)s) { ionized[p] = fcoll * %(ZETA)s; }
 }