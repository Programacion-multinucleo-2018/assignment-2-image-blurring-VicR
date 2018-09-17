/*
  Image Blurring using GPU
  Víctor Rendón Suárez
  A01022462
*/
#include <iostream>
#include <cstdio>
#include <cmath>
#include <chrono>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <cuda_runtime.h>

using namespace std;
#define SIZE 5

__global__ void focus_to_blur_kernel(unsigned char* input, unsigned char* output, int width, int height, int blurStep)
{
	// 2D Index of current thread
	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	int nBase = floor(SIZE/2.0);

	// Only valid threads perform memory I/O
	if ((xIndex < width) && (yIndex < height)) {
		// Location of pixel in input
		const int blur_tid = yIndex * blurStep + (3 * xIndex);

		int red_avg = 0;
		int green_avg = 0;
		int blue_avg = 0;
		int ct = 0;
		//Get the average of the surrounding pixels
		for (int i = -nBase; i <= nBase; i++) {
			for (int j = -nBase; j <= nBase; j++) {
				const int blur_tid = (yIndex + i) * blurStep + (3 * (xIndex + j));
				if(xIndex+j>0 && yIndex+i>0 && xIndex+j<width && yIndex+i<height ) {
					ct++;
					red_avg += input[blur_tid + 2];
					green_avg += input[blur_tid + 1];
					blue_avg += input[blur_tid];
				}
			}
		}
		//Changing the central pixel with the average of the others
		output[blur_tid + 2] = static_cast<unsigned char>(red_avg / ct);
		output[blur_tid + 1] = static_cast<unsigned char>(green_avg / ct);
		output[blur_tid] = static_cast<unsigned char>(blue_avg / ct);
	}
}

void blur_image(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;
	// Calculate total number of bytes of input and output image
	size_t i_bytes = input.step * input.rows;
	size_t o_bytes = output.step * output.rows;
	unsigned char *d_input, *d_output;

	// Allocate device memory
	cudaMalloc(&d_input, i_bytes);
	cudaMalloc(&d_output, o_bytes);

	// Copy data from OpenCV input image to device memory
	cudaMemcpy(d_input, input.ptr(), i_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, input.ptr(), i_bytes, cudaMemcpyHostToDevice);

	// Specify a reasonable block size
	const dim3 block(16, 16);

	// Calculate grid size to cover the whole image
	// const dim3 grid((input.cols + block.x - 1) / block.x, (input.rows + block.y - 1) / block.y);
	const dim3 grid((int)ceil((float)input.cols / block.x), (int)ceil((float)input.rows/ block.y));
	printf("focus_to_blur_kernel<<<(%d, %d) , (%d, %d)>>>\n", grid.x, grid.y, block.x, block.y);

	// Launch the blurring kernel
	auto start_time = std::chrono::high_resolution_clock::now();
	focus_to_blur_kernel <<<grid, block >>>(d_input, d_output, input.cols, input.rows, static_cast<int>(input.step));
	auto end_time = std::chrono::high_resolution_clock::now();
	chrono::duration<float, milli> duration_ms = end_time - start_time;
	printf("Image blurring using GPU, time elapsed: %f ms\n", duration_ms.count());

	// Synchronize to check for any kernel launch errors
	cudaDeviceSynchronize();

	// Copy back data from destination device meory to OpenCV output image
	cudaMemcpy(output.ptr(), d_output, o_bytes, cudaMemcpyDeviceToHost);

	// Free the device memory
	cudaFree(d_input);
	cudaFree(d_output);
}

int main(int argc, char *argv[])
{
	string input_image;
	// Use default image if none specified in command
	if (argc < 2)
		input_image = "image.jpg";
	else
		input_image = argv[1];

	cv::Mat input = cv::imread(input_image, CV_LOAD_IMAGE_COLOR);
	if (input.empty()) {
		cout << "Error: Specified image not found." << std::endl;
		return -1;
	}

	//Output image
	cv::Mat output(input.rows, input.cols, input.type());
	//Call the wrapper function
	blur_image(input, output);

	//Allow the windows to resize
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);
	//Show the input and output
	imshow("Input", input);
	imshow("Output", output);
	//Wait for key press
	cv::waitKey();

	return 0;
}
