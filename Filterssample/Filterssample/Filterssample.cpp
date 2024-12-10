#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem> 
class Filter {
public:

    //Function for rotate left
    cv::Mat rotateLeft(const cv::Mat& src) {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, dst, 0);
        return dst;
    }
    // Function for rotate right
    cv::Mat rotateRight(const cv::Mat& src) {
        cv::Mat dst;
        cv::transpose(src, dst);
        cv::flip(dst, dst, 1);
        return dst;
    }
    //Function for invertimage
    cv::Mat invertImage(const cv::Mat& inputImage) {
        cv::Mat invertedImage;
        cv::bitwise_not(inputImage, invertedImage);
        return invertedImage;
    }
    //Function for horizontalflip
    cv::Mat flipHorizontal(const cv::Mat& image) {
        cv::Mat flipped;
        cv::flip(image, flipped, 1);
        return flipped;
    }

    //Function for verticalflip
    cv::Mat flipVertical(const cv::Mat& image) {
        cv::Mat flipped;
        cv::flip(image, flipped, 0);
        return flipped;
    }
    // New function to adjust brightness
    cv::Mat adjustBrightness(const cv::Mat& image, int brightnessValue) {
        cv::Mat brightImage;
        image.convertTo(brightImage, -1, 1, brightnessValue);
        return brightImage;
    }
    // Contrast adjustment function
    cv::Mat adjustContrast(const cv::Mat& image, double alpha) {
        cv::Mat contrastImage;
        image.convertTo(contrastImage, -1, alpha, 0);  // Adjust contrast with alpha
        return contrastImage;
    }
    //Updated Window/Level function for both 8-bit and 16-bit images
    cv::Mat windowLevel(const cv::Mat& image, int window, int level) {
        cv::Mat output;
        int maxValue = (image.depth() == CV_16U) ? 65535 : 255;  // Max pixel value based on bit depth

        // Convert image to float for accurate calculations
        image.convertTo(output, CV_32F);

        // Apply the Window/Level formula
        output = ((output - level) / window) * maxValue / 2.0 + maxValue / 2.0;

        // Clip the values to the valid range for the image depth
        cv::threshold(output, output, maxValue, maxValue, cv::THRESH_TRUNC);
        cv::threshold(output, output, 0, 0, cv::THRESH_TOZERO);

        // Convert back to the original image depth
        output.convertTo(output, image.type());

        return output;
    }
    // Adaptive thresholding (Gaussian method)
    cv::Mat adaptiveThresholdGaussian(const cv::Mat& image, int blockSize, int C) {
        cv::Mat thresholded;
        cv::adaptiveThreshold(image, thresholded, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY, blockSize, C);
        return thresholded;
    }
    cv::Mat applyBlur(const cv::Mat& image, int kernelSize) {
        cv::Mat blurredImage;
        cv::blur(image, blurredImage, cv::Size(kernelSize, kernelSize));
        return blurredImage;
    }
    // Function to apply Gaussian blur
    cv::Mat applyGaussianBlur(const cv::Mat& image, int kernelSize, double sigmaX) {
        cv::Mat gaussianBlurredImage;
        cv::GaussianBlur(image, gaussianBlurredImage, cv::Size(kernelSize, kernelSize), sigmaX);
        return gaussianBlurredImage;
    }
    // Function to apply median blur
    cv::Mat applyMedianBlur(const cv::Mat& image, int kernelSize) {
        cv::Mat medianBlurredImage;
        cv::medianBlur(image, medianBlurredImage, kernelSize);
        return medianBlurredImage;
    }
   
    cv::Mat applyCannyEdgeDetection(const cv::Mat& image, double threshold1, double threshold2) {
        cv::Mat edges, processedImage;

        // Check if the image is 16-bit (CV_16U) and convert to 8-bit if needed
        if (image.depth() == CV_16U) {
            // Scale the 16-bit image to 8-bit by normalizing between 0 and 255
            double minVal, maxVal;
            cv::minMaxLoc(image, &minVal, &maxVal);  // Find the min and max pixel values
            image.convertTo(processedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * (255.0 / (maxVal - minVal)));
        }
        else {
            // If already 8-bit, just use the original image
            processedImage = image;
        }

        // Apply Canny edge detection on the 8-bit processed image
        cv::Canny(processedImage, edges, threshold1, threshold2);

        return edges;
    }
    // Sobel edge detection function for 16-bit images
    cv::Mat applySobelEdgeDetection(const cv::Mat& image, int ddepth, int ksize) {
        cv::Mat grad_x, grad_y;
        cv::Mat abs_grad_x, abs_grad_y;
        cv::Mat processedImage;

        // If the image is 16-bit, normalize to 8-bit for better visualization of gradients
        if (image.depth() == CV_16U) {
            double minVal, maxVal;
            cv::minMaxLoc(image, &minVal, &maxVal);  // Find the min and max pixel values
            image.convertTo(processedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * (255.0 / (maxVal - minVal)));
        }
        else {
            processedImage = image;
        }

        // Apply Sobel operator in X and Y directions
        cv::Sobel(processedImage, grad_x, ddepth, 1, 0, ksize);
        cv::Sobel(processedImage, grad_y, ddepth, 0, 1, ksize);

        // Convert gradients to absolute values
        cv::convertScaleAbs(grad_x, abs_grad_x);
        cv::convertScaleAbs(grad_y, abs_grad_y);

        // Combine the gradients
        cv::Mat sobelEdges;
        cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, sobelEdges);

        return sobelEdges;
    }
    // Laplacian edge detection function for 16-bit images
    cv::Mat applyLaplacianEdgeDetection(const cv::Mat& image, int ddepth, int ksize) {
        cv::Mat laplacianEdges, processedImage;

        // If the image is 16-bit, normalize to 8-bit for better visualization of edges
        if (image.depth() == CV_16U) {
            double minVal, maxVal;
            cv::minMaxLoc(image, &minVal, &maxVal);  // Find the min and max pixel values
            image.convertTo(processedImage, CV_8U, 255.0 / (maxVal - minVal), -minVal * (255.0 / (maxVal - minVal)));
        }
        else {
            processedImage = image;
        }

        // Apply the Laplacian operator
        cv::Laplacian(processedImage, laplacianEdges, ddepth, ksize);

        // Convert the result to absolute values to visualize edges better
        cv::convertScaleAbs(laplacianEdges, laplacianEdges);

        return laplacianEdges;
    }


};


int main() {

    Filter o;
   
    cv::Mat inputImage = cv::imread("C:\\Users\\Dhvani\\Downloads\\sample image.jpg", cv::IMREAD_GRAYSCALE);
  
    //std::cout << "inpyt read";
    //if (inputImage.empty()) {
    //    std::cerr << "Error loading image" << std::endl;
    //    return -1;
    //}
   /* cv::Mat right90=o.rotateRight(inputImage);
    cv::Mat left90=o.rotateLeft(inputImage);*/
    //cv::Mat fv=o.flipVertical(inputImage);
   /* cv::Mat dilatedImage = o.applyDilation(inputImage, 3);
    cv::Mat erodedImage = o.applyErosion(inputImage, 3);
    cv::Mat blackHatImage = o.applyBlackHat(inputImage, 3);




    cv::imwrite("E:\\img\\output\\dilated_output.jpg", dilatedImage);
    cv::imwrite("E:\\img\\output\\eroded_output.jpg", erodedImage);
    cv::imwrite("E:\\img\\output\\original.jpg", inputImage);
    cv::imwrite("E:\\img\\output\\blackHat.jpg", blackHatImage);

    */
    //cv::imwrite("E:\\img\\output\\right.jpg", right90);
    //cv::imwrite("E:\\img\\output\\left.jpg",fv);

    // Apply brightness adjustment
    int brightnessValue = 50;  // Increase brightness by 50 (can be negative to reduce brightness)
    cv::Mat brightImage = o.adjustBrightness(inputImage, brightnessValue);
    std::cout << "bright read";

    //Apply contrast adjustment
    double alpha = 1.5;  // Increase contrast by 50% (1.5x)
    cv::Mat contrastImage = o.adjustContrast(inputImage, alpha);
    std::cout << "contr read";

    //// Save the brightened image
    //cv::imwrite("C:\\save\\bright.png", brightImage);
    //cv::imwrite("C:\\save\\contrast.png", contrastImage);

    //Apply for normal blur
    int kernelSize = 5;  // Adjust kernel size as needed
    cv::Mat blurredImage = o.applyBlur(inputImage, kernelSize);



    // Apply Canny edge detection
    double threshold1 = 100;  // First threshold for hysteresis
    double threshold2 = 200;  // Second threshold for hysteresis
    cv::Mat edges = o.applyCannyEdgeDetection(inputImage, threshold1, threshold2);

    // Apply for sobel detection
    int ddepth = CV_16S;  // Use 16-bit signed depth for more precise gradients
    int ksize = 3;        // Kernel size for the Sobel operator (usually 3 or 5)
    cv::Mat sobelEdges = o.applySobelEdgeDetection(inputImage, ddepth, ksize);

    // Apply Laplacian edge detection
    int ddepth1 = CV_16S;  // Use 16-bit signed depth for more precise gradients
    int ksize1 = 3;        // Kernel size for the Laplacian operator (typically 1, 3, or 5)
    cv::Mat laplacianEdges = o.applyLaplacianEdgeDetection(inputImage, ddepth1, ksize1);

    return 0;
}
