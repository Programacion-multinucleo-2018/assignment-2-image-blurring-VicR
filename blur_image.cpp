/*
  Image Blurring
  Víctor Rendón Suárez
  A01022462
*/
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
#define SIZE 5

// Image blurring function, receives (Opencv mat, output)
void blur_image(const cv::Mat& input, cv::Mat& output)
{
	cout << "Input image; step: " << input.step << " rows: " << input.rows << " cols: " << input.cols << endl;

  int nBase = floor(SIZE/2.0);
	// Iterate on image pixels
	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
      int red_avg = 0;
      int green_avg = 0;
			int blue_avg = 0;
			int ct = 0;

      // Handle surrounding pixels
			for (int i = -nBase; i <= nBase; i++) {
				for (int j = -nBase; j <= nBase; j++) {
          int xn = x + j;
          int yn = y + i;
          // Handle borders
          if ((xn > 0 && yn > 0) && (xn < input.cols && yn < input.rows)) {
            ct++;
            red_avg += input.at<cv::Vec3b>(yn,xn)[2];
            green_avg += input.at<cv::Vec3b>(yn,xn)[1];
  					blue_avg += input.at<cv::Vec3b>(yn,xn)[0];
          }
				}
			}
      output.at<cv::Vec3b>(y,x)[2] = red_avg / ct;
      output.at<cv::Vec3b>(y,x)[1] = green_avg / ct;
      output.at<cv::Vec3b>(y,x)[0] = blue_avg / ct;
		}
	}
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

	// Output image
	cv::Mat output(input.rows, input.cols, input.type());
	// Blur image, measure time elapsed
  auto start_time = std::chrono::high_resolution_clock::now();
	blur_image(input, output);
	auto end_time = std::chrono::high_resolution_clock::now();
	chrono::duration<float, milli> duration_ms = end_time - start_time;
	printf("Image blurring, time elapsed: %f ms\n", duration_ms.count());

	// Resize and show images
	namedWindow("Input", cv::WINDOW_NORMAL);
	namedWindow("Output", cv::WINDOW_NORMAL);
	imshow("Input", input);
	imshow("Output", output);
	// Close with keypress
	cv::waitKey();

	return 0;
}
