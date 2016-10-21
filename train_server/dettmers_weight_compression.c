// algorithm, implementation is based on "8-Bit Approximations for Parallelism in Deep Learning" by Tim Dettmers
// https://github.com/TimDettmers/clusterNet/blob/master/source/clusterKernels.cu

#include <math.h>

const float tbl_floats[126] = {2.750000021e-06,7.249999726e-06,1.875000089e-05,3.624999954e-05,5.874999624e-05,8.624999464e-05,1.437500032e-04,2.312500001e-04,3.187500115e-04,4.062500084e-04,5.187499919e-04,6.562499912e-04,7.937499322e-04,9.312499315e-04,1.218750025e-03,1.656249980e-03,2.093750052e-03,2.531250007e-03,2.968749963e-03,3.406249918e-03,3.843750106e-03,4.281249829e-03,4.843750037e-03,5.531250034e-03,6.218749564e-03,6.906249560e-03,7.593749557e-03,8.281249553e-03,8.968749084e-03,9.656248614e-03,1.109374966e-02,1.328125037e-02,1.546875015e-02,1.765624993e-02,1.984374970e-02,2.203124948e-02,2.421874925e-02,2.640625089e-02,2.859375067e-02,3.078125045e-02,3.296874836e-02,3.515625000e-02,3.734375164e-02,3.953124955e-02,4.171875119e-02,4.390624911e-02,4.671875015e-02,5.015625060e-02,5.359374732e-02,5.703124776e-02,6.046874821e-02,6.390624493e-02,6.734374911e-02,7.078124583e-02,7.421874255e-02,7.765624672e-02,8.109374344e-02,8.453124017e-02,8.796874434e-02,9.140624106e-02,9.484373778e-02,9.828124195e-02,1.054687500e-01,1.164062470e-01,1.273437440e-01,1.382812560e-01,1.492187530e-01,1.601562500e-01,1.710937470e-01,1.820312440e-01,1.929687560e-01,2.039062530e-01,2.148437500e-01,2.257812470e-01,2.367187440e-01,2.476562560e-01,2.585937381e-01,2.695312500e-01,2.804687619e-01,2.914062440e-01,3.023437560e-01,3.132812381e-01,3.242187500e-01,3.351562619e-01,3.460937440e-01,3.570312560e-01,3.679687381e-01,3.789062500e-01,3.898437619e-01,4.007812440e-01,4.117187560e-01,4.226562381e-01,4.335937500e-01,4.445312619e-01,4.585937560e-01,4.757812321e-01,4.929687381e-01,5.101562142e-01,5.273437500e-01,5.445312262e-01,5.617187023e-01,5.789062381e-01,5.960937142e-01,6.132812500e-01,6.304687262e-01,6.476562023e-01,6.648437381e-01,6.820312142e-01,6.992186904e-01,7.164062262e-01,7.335937023e-01,7.507811785e-01,7.679687142e-01,7.851561904e-01,8.023436666e-01,8.195312023e-01,8.367186785e-01,8.539061546e-01,8.710936904e-01,8.882811666e-01,9.054686427e-01,9.226561785e-01,9.398436546e-01,9.570311308e-01,9.742186666e-01,9.914061427e-01};

const float thres_low = 1.5e-6F;
const float thres_high = 0.995703F;

void compression_8bit(float* src, unsigned char* dst, int size) {
  // get max (abs)
  float maxval = 1e-20F;//avoid zero division

#pragma omp parallel for reduction(max: maxval)
  for (int i = 0; i < size; i++) {
    float abssrc = fabsf(src[i]);
    if (maxval < abssrc) {
      maxval = abssrc;
    }
  }

#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    float srcval = src[i];
    unsigned char signval = srcval >= 0.0F ? 0 : 128;
    float absnumber = fabsf(srcval) / maxval;
    unsigned char code = 0;
    if (absnumber < thres_low) {
      code = 126;
    } else if (absnumber > thres_high) {
      code = 127;
    } else {

	int pivot = 63;
	int upper_pivot = 125;
  int lower_pivot = 0;
  		  for(int j = 32; j > 0; j>>=1)
		  {
			  if(absnumber > tbl_floats[pivot])
			  {
				  lower_pivot = pivot;
				  pivot+=j;
			  }
			  else
			  {
				  upper_pivot = pivot;
				  pivot-=j;
			  }

		  }

		  if(lower_pivot == pivot)
			  if(fabsf(tbl_floats[pivot]-absnumber) < (tbl_floats[upper_pivot]-absnumber))
          code = pivot;
			  else
        code=upper_pivot;
		  else
			  if((tbl_floats[pivot]-absnumber) < fabsf(tbl_floats[lower_pivot]-absnumber))
				  code=pivot;
			  else
		  	  	  code=lower_pivot;
    }

    dst[i] = code + signval;
  }

  unsigned char *maxval_bytes = (unsigned char*)&maxval;
  dst[size+0]=maxval_bytes[0];
  dst[size+1]=maxval_bytes[1];
  dst[size+2]=maxval_bytes[2];
  dst[size+3]=maxval_bytes[3];
}

void decompression_8bit(unsigned char* src, float* dst, int size) {
  // table generation
  float restore_table_floats[256];
  float maxval;
  unsigned char *maxval_bytes = (unsigned char*)&maxval;
  maxval_bytes[0]=src[size+0];
  maxval_bytes[1]=src[size+1];
  maxval_bytes[2]=src[size+2];
  maxval_bytes[3]=src[size+3];

  for (int i = 0; i < 126; i++) {
    restore_table_floats[i] = tbl_floats[i] * maxval;
    restore_table_floats[i+128] = -(tbl_floats[i] * maxval);
  }
  restore_table_floats[126] = 0.0F;
  restore_table_floats[126+128] = 0.0F;
  restore_table_floats[127] = maxval;
  restore_table_floats[127+128] = -maxval;

  // mapping
#pragma omp parallel for
  for (int i = 0; i < size; i++) {
    dst[i] = restore_table_floats[src[i]];
  }
}
