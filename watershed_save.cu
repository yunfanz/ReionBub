#define INF 9999999999
#define PLATEAU 0
#define BLOCK_SIZE 10

// Convert 3D index to 1D index.
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / BLOCK_SIZE) * (BLOCK_SIZE - 2) + (off)-1)

// Texture memory for image.
texture<float,3> img;
// Neighbour pixel generator (N-W to W order).
__constant__ int N_xs[26] = {0,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,0};
__constant__ int N_ys[26] = {0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,0};
__constant__ int N_zs[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};

// Step 1. Label local minima or flatland as PLATEAU
__global__ void descent_kernel(float* labeled, const int w, const int h, const int d)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
 
  __shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int img_z = L2I(k,tz);
  //int new_w = w + w * 2;
  //int new_h = h + h * 2;
  //int new_d = d + d * 2;
  int new_w = w + w/(bdx-2)*2+1;
  int new_h = h + h/(bdy-2)*2+1;
  int new_d = d + d/(bdz-2)*2+1;
  int p = INDEX(img_z,img_y,img_x,w);

  int ghost = (tx == 0 || ty == 0 || tz == 0 ||
  tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

  if ((img_x < 0) || (img_y < 0) || (img_z < 0) ||
     (img_x == w) || (img_y == h) || (img_z == d)) {
       s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] = INF;
  } else {
     s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] = tex3D(img,img_x,img_y,img_z);
  }

  __syncthreads();

  if (j < new_h && i < new_w && k < new_d && ghost == 0) {
    float I_q_min = INF;
    float I_p = tex3D(img,img_x,img_y,img_z);
    int exists_q = 0;

    for (int kk = 0; kk < 26; kk++) {
      int n_x = N_xs[kk]+tx; int n_y = N_ys[kk]+ty; int n_z = N_zs[kk]+tz;
      float I_q = s_I[INDEX(n_z,n_y,n_x,BLOCK_SIZE)];
      if (I_q < I_q_min) I_q_min = I_q;
    }
    
    for (int kk = 0; kk < 26; kk++) {
      int x = N_xs[kk]; int y = N_ys[kk]; int z = N_zs[kk];
      int n_x = x+tx; int n_y = y+ty; int n_z = z+tz;
      int n_tx = L2I(i,n_x); int n_ty = L2I(j,n_y); int n_tz = L2I(k,n_z);
      float I_q = s_I[INDEX(n_z,n_y,n_x,BLOCK_SIZE)];
      int q = INDEX(n_tz,n_ty,n_tx,w);
      if (I_q < I_p && I_q == I_q_min) {
        labeled[p] = -q;
        exists_q = 1; break;
      }
    }
    if (exists_q == 0) labeled[p] = PLATEAU;
  }

}
//step1B stabilize the plateau and remove saddle points
__global__ void stabilize_kernel(float* L, int* C, const int w, const int h, const int d)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
 
  __shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
  __shared__ float s_I[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int img_z = L2I(k,tz);
  int true_p = INDEX(img_z, img_y,img_x,w);
  int s_p = INDEX(tz,ty,tx,BLOCK_SIZE);
  int new_w = w + w/(bdx-2)*2+1;
  int new_h = h + h/(bdy-2)*2+1;
  int new_d = d + d/(bdz-2)*2+1;
  int ghost = (tx == 0 || ty == 0 || tz == 0 ||
  tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

  if ((img_x < 0) || (img_y < 0) || (img_z < 0) ||
     (img_x == w) || (img_y == h) || (img_z == d)) {
     s_L[INDEX(tz,ty,tx,BLOCK_SIZE)] = INF;
     s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] = INF;
  } else {
    s_L[s_p] = L[INDEX(img_z,img_y,img_x,w)];
    s_I[INDEX(tz,ty,tx,BLOCK_SIZE)] = tex3D(img,img_x,img_y,img_z);
  }

  __syncthreads();

  int active = (j < new_h && i < new_w && k < new_d && s_L[s_p] > 0) ? 1 : 0;  //whether still is in plateau

  if (active == 1 && ghost == 0) {
    for (int kk = 0; kk < 26; kk++) {
      int n_x = N_xs[kk] + tx; int n_y = N_ys[kk] + ty; int n_z = N_zs[kk] + tz;
      int s_q = INDEX(n_z,n_y,n_x,BLOCK_SIZE);
      if (s_L[s_q] == INF) continue;
      if ( s_I[s_q] == s_I[s_p] && s_L[s_q] < 0 ) {
        s_L[s_p] = s_L[s_q];
        break;
      }
    }
    if (L[true_p] != s_L[s_p]) {
      L[true_p] = s_L[s_p];
      atomicAdd(&C[0],1);
    }
  }
}

// Step 2A: change the PLATEAU labels to be location p+1
__global__ void increment_kernel(float* L, const int w, const int h, const int d)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int p = INDEX(k,j,i,w);

  if (k < d && j < h && i < w && L[p] == PLATEAU) {
    L[p] = p + 1;
  }
}

// Step 2B. Propagate the labels of the plateaus (iterate till convergence)
__global__ void minima_kernel(float* L, int* C, const int w, const int h, const int d)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
 
  __shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int img_z = L2I(k,tz);
  int true_p = INDEX(img_z, img_y,img_x,w);
  int s_p = INDEX(tz,ty,tx,BLOCK_SIZE);
  //int new_w = w + w * 2;
  //int new_h = h + h * 2;
  //int new_d = d + d * 2;
  int new_w = w + w/(bdx-2)*2+1;
  int new_h = h + h/(bdy-2)*2+1;
  int new_d = d + d/(bdz-2)*2+1;
  int ghost = (tx == 0 || ty == 0 || tz == 0 ||
  tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

  if ((img_x < 0) || (img_y < 0) || (img_z < 0) ||
     (img_x == w) || (img_y == h) || (img_z == d)) {
     s_L[INDEX(tz,ty,tx,BLOCK_SIZE)] = INF;
  } else {
    s_L[s_p] = L[INDEX(img_z,img_y,img_x,w)];
  }

  __syncthreads();

  int active = (j < new_h && i < new_w && k < new_d && s_L[s_p] > 0) ? 1 : 0;

  if (active == 1 && ghost == 0) {
    for (int kk = 0; kk < 26; kk++) {
      int n_x = N_xs[kk] + tx; int n_y = N_ys[kk] + ty; int n_z = N_zs[kk] + tz;
      int s_q = INDEX(n_z,n_y,n_x,BLOCK_SIZE);
      if (s_L[s_q] == INF) continue;
      if (s_L[s_q] > s_L[s_p]) //if not plateau, propagete to lower image values
                               //if plateau propagate to higher indices
        s_L[s_p] = s_L[s_q];
    }
    if (L[true_p] != s_L[s_p]) {
      L[true_p] = s_L[s_p];
      atomicAdd(&C[0],1);
    }
  }
}


// Step 3.
__global__ void plateau_kernel(float* L, int* C, const int w, const int h, const int d)
{
  int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
  int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
  int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
 
  __shared__ float s_L[BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE];
  int size = BLOCK_SIZE - 2;
  int img_x = L2I(i,tx);
  int img_y = L2I(j,ty);
  int img_z = L2I(k,tz);
  int true_p = INDEX(img_z,img_y,img_x,w);
  int p = INDEX(tz,ty,tx,BLOCK_SIZE);
  //int new_w = w + w * 2;
  //int new_h = h + h * 2;
  //int new_d = d + d * 2;
  int new_w = w + w/(bdx-2)*2+1;
  int new_h = h + h/(bdy-2)*2+1;
  int new_d = d + d/(bdz-2)*2+1;
  int ghost = (tx == 0 || ty == 0 || tz == 0 ||
  tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

  // Load data into shared memory.
  if ((img_x < 0) || (img_y < 0) || (img_z < 0) ||
     (img_x == w) || (img_y == h) || (img_z == d)) {
       s_L[INDEX(tz,ty,tx,BLOCK_SIZE)] = INF;
  } else {
     s_L[INDEX(tz,ty,tx,BLOCK_SIZE)] =
     L[INDEX(img_z,img_y,img_x,w)];
  }

  __syncthreads();

  if (j < new_h && i < new_w && k < new_d &&
    s_L[p] == PLATEAU && ghost == 0) {
    float I_p = tex3D(img,img_x,img_y,img_z); 
    float I_q;
    int n_x, n_y, n_z; float L_q;

    for (int kk = 0; kk < 26; kk++) {
      n_x = N_xs[kk]+tx; n_y = N_ys[kk]+ty; n_z = N_zs[kk]+tz;
      L_q = s_L[INDEX(n_z,n_y,n_x,BLOCK_SIZE)];
      if (L_q == INF || L_q >= 0) continue;
      int n_tx = L2I(i,n_x); int n_ty = L2I(j,n_y); int n_tz = L2I(k,n_z);
      int q = INDEX(n_tz,n_ty,n_tx,w);
      I_q = tex3D(img,n_tx,n_ty,n_tz);
      if (I_q == I_p && L[true_p] != -q) {
        L[true_p] = -q; 
        atomicAdd(&C[0], 1); 
        break;
      }
    }
  }

}
// Step 4.
__global__ void flood_kernel(float* L, int* C, const int w, const int h, const int d)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int p = INDEX(k,j,i,w); int q;

  if (j < h && i < w && k < d && L[p] <= 0) {
    q = -L[p];
    if (L[q] > 0 && L[p] != L[q]) {
    //if (L[p] != L[q]) {
      L[p] = L[q];
      atomicAdd(&C[0],1);
    }
  }
}