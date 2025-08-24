package com.example.droneguard.yolo;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

public class Preprocessor {
    
    // Load OpenCV native library
    static {
        nu.pattern.OpenCV.loadShared();
    }
    
    public static class Input {
        public final float[] data;      // RGB pixel data normalized to [0,1]
        public final double scale;      // Scale factor applied
        public final double padX, padY; // Padding added
        public final int origWidth, origHeight; // Original image dimensions
        
        public Input(float[] data, double scale, double padX, double padY, int origWidth, int origHeight) {
            this.data = data;
            this.scale = scale;
            this.padX = padX;
            this.padY = padY;
            this.origWidth = origWidth;
            this.origHeight = origHeight;
        }
    }
    
    /**
     * Letterbox preprocessing for YOLO model
     * Resizes image to target size while maintaining aspect ratio and adding padding
     */
    public static Input letterbox(Mat src, int targetSize) {
        int origHeight = src.rows();
        int origWidth = src.cols();
        
        // Calculate scale to fit image in target size while maintaining aspect ratio
        double scale = Math.min((double) targetSize / origWidth, (double) targetSize / origHeight);
        
        // Calculate new dimensions
        int newWidth = (int) (origWidth * scale);
        int newHeight = (int) (origHeight * scale);
        
        // Resize the image
        Mat resized = new Mat();
        Imgproc.resize(src, resized, new Size(newWidth, newHeight));
        
        // Calculate padding to center the image
        double padX = (targetSize - newWidth) / 2.0;
        double padY = (targetSize - newHeight) / 2.0;
        
        // Create padded image with gray background (114, 114, 114)
        Mat padded = new Mat(targetSize, targetSize, CvType.CV_8UC3, new Scalar(114, 114, 114));
        
        // Copy resized image to center of padded image
        Rect roi = new Rect((int) padX, (int) padY, newWidth, newHeight);
        Mat roiMat = new Mat(padded, roi);
        resized.copyTo(roiMat);
        
        // Convert BGR to RGB and normalize to [0,1]
        Mat rgb = new Mat();
        Imgproc.cvtColor(padded, rgb, Imgproc.COLOR_BGR2RGB);
        
        // Convert to float array (CHW format: channels, height, width)
        float[] data = new float[3 * targetSize * targetSize];
        byte[] imgData = new byte[(int) (rgb.total() * rgb.elemSize())];
        rgb.get(0, 0, imgData);
        
        // Convert from HWC to CHW format and normalize
        int idx = 0;
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < targetSize; h++) {
                for (int w = 0; w < targetSize; w++) {
                    int pixelIdx = (h * targetSize + w) * 3 + c;
                    data[idx++] = (imgData[pixelIdx] & 0xFF) / 255.0f;
                }
            }
        }
        
        // Clean up
        resized.release();
        padded.release();
        roiMat.release();
        rgb.release();
        
        return new Input(data, scale, padX, padY, origWidth, origHeight);
    }
}