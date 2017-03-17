#define INF 9999999999
// Convert 3D index to 1D index.
#define INDEX(k,j,i,ld) ((k)*ld*ld + (j) * ld + (i))

// Convert local (shared memory) coord to global (image) coordinate.
#define L2I(ind,off) (((ind) / blockDim.x) * (blockDim.x - 2) + (off)-1)
#define BS 10
__constant__ int N1_xs[6] = {0,1,1,-1,-1,0};
__constant__ int N1_ys[6] = {0,1,-1,-1,1,0};
__constant__ int N1_zs[6] = {-1,0,0,0,0,1};

__constant__ int N2_xs[18] = {0,0,1,0,-1,-1,0,1,1,1,0,-1,-1,0,1,0,-1,0};
__constant__ int N2_ys[18] = {0,-1,0,1,0,-1,-1,-1,0,1,1,1,0,-1,0,1,0,0};
__constant__ int N2_zs[18] = {-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1};

__constant__ int N3_xs[26] = {0,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,0};
__constant__ int N3_ys[26] = {0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,-1,-1,-1,0,1,1,1,0,0};
__constant__ int N3_zs[26] = {-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1};

__global__ void step(float *C, float *M, bool *Mask, bool *maxima)
{
    bool ismax = true;
    int w = %(NDIM)s; 
    int bsize = blockDim.x - 2;
    int tx = threadIdx.x;  int ty = threadIdx.y; int tz = threadIdx.z;
    int bx = blockIdx.x;   int by = blockIdx.y; int bz = blockIdx.z;
    int bdx = blockDim.x;  int bdy = blockDim.y; int bdz = blockDim.z;
    int i = bdx * bx + tx; int j = bdy * by + ty; int k = bdz * bz + tz;
    int img_x = L2I(i,tx);
    int img_y = L2I(j,ty);
    int img_z = L2I(k,tz);
    int new_w = w + w/(bdx-2)*2+1;
    int threadId = INDEX(k,j,i, new_w);
    int p = INDEX(img_z,img_y,img_x,w);
    int ghost = (tx == 0 || ty == 0 || tz == 0 || tx == bdx - 1 || ty == bdy - 1 || tz == bdz - 1);

    __shared__ float s_C[BS*BS*BS];
    if ((bx == 0 && tx == 0) || (by == 0 && ty == 0) || (bz == 0 && tz == 0) ||
     (bx == (w / bsize - 1) && tx == bdx - 1) ||
     (by == (w / bsize - 1) && ty == bdy - 1) ||
     (bz == (w / bsize - 1) && tz == bdz - 1)) {
       s_C[INDEX(tz,ty,tx,bdx)] = INF;
    } else {
     s_C[INDEX(tz,ty,tx,bdx)] = C[p];
    }

    __syncthreads();


    if ( ( i < new_w) && ( j < new_w ) && ( k < new_w ) && ghost==0 )
    {

        int n_neigh; 
        int* neigh_xs = NULL; int* neigh_ys = NULL; int* neigh_zs = NULL;

        switch (%(CONN)s) {
          case 1:
            n_neigh = 6;
            neigh_xs = N1_xs; neigh_ys = N1_ys; neigh_zs = N1_zs;
            break;
          case 2:
            n_neigh = 18;
            neigh_xs = N2_xs; neigh_ys = N2_ys; neigh_zs = N2_zs;
            break;
          case 3:
            n_neigh = 26;
            neigh_xs = N3_xs; neigh_ys = N3_ys; neigh_zs = N3_zs;
            break;
        }

        int ne;

        if (!Mask[p]) {ismax = false;}
        else
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            int x = neigh_xs[ni]; int y = neigh_ys[ni]; int z = neigh_zs[ni];
            int nex = x+tx; int ney = y+ty; int nez = z+tz;  //shared memory indices of neighbors
            if (s_C[INDEX(tz,ty,tx,bdx)] < s_C[INDEX(nez,ney,nex,bdx)]) {ismax = false;}
          }
        }
        maxima[p] = ismax;
        M[p] = C[p];

        __syncthreads();

        if (Mask[p])
        {
          for (int ni=0; ni<n_neigh; ni++) 
          {
            int x = neigh_xs[ni]; int y = neigh_ys[ni]; int z = neigh_zs[ni];
            int nex = x+tx; int ney = y+ty; int nez = z+tz;  //shared memory indices of neighbors
            ne = INDEX(nez,ney,nex,bdx);
            if ( (maxima[ne]) && (s_C[INDEX(tz,ty,tx,bdx)] >= s_C[ne] - %(HVAL)s) ) 
            {
                M[p] = (s_C[INDEX(tz,ty,tx,bdx)]<s_C[ne]) ? s_C[ne] : s_C[INDEX(tz,ty,tx,bdx)];
            }
          }
        }
    }
}
__global__ void finalize(const float *C, float *M, bool *Mask, bool *maxima)
{
    int w = %(NDIM)s; 
    int i = threadIdx.x + blockDim.x*blockIdx.x;
    int j = threadIdx.y + blockDim.y*blockIdx.y;
    int k = threadIdx.z + blockDim.z*blockIdx.z;
    if ( ( i < %(NDIM)s ) && ( j < %(NDIM)s ) && ( k < %(NDIM)s ) )
    {
        int threadId = k*w*w + j*w + i;
        
        if (maxima[threadId])
        {
            M[threadId] = C[threadId]- %(HVAL)s;
        }
    }
}