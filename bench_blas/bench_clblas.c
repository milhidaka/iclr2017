#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <sys/time.h>

/* Include the clBLAS header. It includes the appropriate OpenCL headers */
#include <clBLAS.h>


int is_trans(const char* c) {
    if (strcasecmp(c, "N") == 0) {
        return 0;
    } else if (strcasecmp(c, "T") == 0) {
        return 1;
    } else {
        printf("Unknown transposition directive\n");
        exit(EXIT_FAILURE);
    }
}

static void fill_mat(float* m, int rows, int cols) {
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            m[i+j*rows] = (float)(i + j * rows + 1);
        }
    }
}

int main( int argc, char* argv[] )
{
    int exec_times_offset = 10;
    int exec_times_actual = 20;

    int M, N, K;// M*K * K*N = M*N
    M = atol(argv[1]); N = atol(argv[2]); K = atol(argv[3]);
    int transA = is_trans(argv[4]);
    int transB = is_trans(argv[5]);
    cl_float *A, *B, *C;
    A = (cl_float *)malloc (M * K * sizeof (*A));
    B = (cl_float *)malloc (K * N * sizeof (*A));
    C = (cl_float *)malloc (M * N * sizeof (*A));
		cl_float alpha = 1.0F, beta = 0.0F;
		size_t lda, ldb, ldc;
		lda = transA ? K : M;
		ldb = transB ? N : K;
		ldc = M;

	cl_int err;
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
	cl_context ctx = 0;
	cl_command_queue queue = 0;
	cl_mem bufA, bufB, bufC;
	cl_event event = NULL;
	int ret = 0;

	fill_mat(A, M, K);
	fill_mat(B, K, N);
	fill_mat(C, M, N);

	/* Setup OpenCL environment. */
	err = clGetPlatformIDs( 1, &platform, NULL );
	err = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );

	props[1] = (cl_context_properties)platform;
	ctx = clCreateContext( props, 1, &device, NULL, NULL, &err );
	queue = clCreateCommandQueue( ctx, device, 0, &err );

	/* Setup clBLAS */
	err = clblasSetup( );

	/* Prepare OpenCL memory objects and place matrices inside them. */
	bufA = clCreateBuffer( ctx, CL_MEM_READ_ONLY, M * K * sizeof(*A),
			NULL, &err );
	bufB = clCreateBuffer( ctx, CL_MEM_READ_ONLY, K * N * sizeof(*B),
			NULL, &err );
	bufC = clCreateBuffer( ctx, CL_MEM_READ_WRITE, M * N * sizeof(*C),
			NULL, &err );

    long long exec_us[3];
    for (int run_i = 0; run_i < 3; run_i++){
	err = clEnqueueWriteBuffer( queue, bufA, CL_TRUE, 0,
			M * K * sizeof( *A ), A, 0, NULL, NULL );
	err = clEnqueueWriteBuffer( queue, bufB, CL_TRUE, 0,
			K * N * sizeof( *B ), B, 0, NULL, NULL );
//	err = clEnqueueWriteBuffer( queue, bufC, CL_TRUE, 0,
//			M * N * sizeof( *C ), C, 0, NULL, NULL );

//        err = clFinish(queue);
        struct timeval tv_begin;
	gettimeofday(&tv_begin, NULL);
	/* Call clBLAS extended function. Perform gemm for the lower right sub-matrices */
	for (int i = 0; i < exec_times_offset + exec_times_actual * run_i; i++) {
	err = clblasSgemm( clblasColumnMajor,
	    transA ? clblasTrans : clblasNoTrans,
			transB ? clblasTrans : clblasNoTrans, 
			M, N, K,
			alpha, bufA, 0, lda,
			bufB, 0, ldb, beta,
			bufC, 0, ldc,
			1, &queue, 0, NULL, &event );

	/* Wait for calculations to be finished. */
//	err = clWaitForEvents( 1, &event );

	}
	/* Fetch results of calculations from GPU memory. */
	err = clEnqueueReadBuffer( queue, bufC, CL_TRUE, 0,
			M * N * sizeof(*C),
			C, 0, NULL, NULL );
        struct timeval tv_end;
	gettimeofday(&tv_end, NULL);
        long long elapsed_usec = ((long long)tv_end.tv_sec * 1000000LL + (long long)tv_end.tv_usec) - ((long long)tv_begin.tv_sec * 1000000LL + (long long)tv_begin.tv_usec);
        exec_us[run_i] = elapsed_usec;
		}

		//calculate time difference of second and third run
    double time_per_one_calc_ms = (double)(exec_us[2] - exec_us[1]) / exec_times_actual / 1000;
    printf("%d,%d,%d,%d,%d,%f\n", M, N, K, transA, transB, time_per_one_calc_ms);

	/* Release OpenCL memory objects. */
	clReleaseMemObject( bufC );
	clReleaseMemObject( bufB );
	clReleaseMemObject( bufA );

	/* Finalize work with clBLAS */
	clblasTeardown( );

	/* Release OpenCL working objects. */
	clReleaseCommandQueue( queue );
	clReleaseContext( ctx );

	return ret;
}
