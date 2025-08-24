package com.example.droneguard.yolo;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

public class Postprocessor {

    // Single-class model
    private static final String[] CLASS_NAMES = { "uav" };

    public static class Detection {
        public final float x, y, w, h; // Bounding box (center x, y, width, height)
        public final float confidence;
        public final int classId;
        public final String className;

        public Detection(float x, float y, float w, float h, float confidence, int classId) {
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
            this.confidence = confidence;
            this.classId = classId;
            this.className = (classId >= 0 && classId < CLASS_NAMES.length) ? CLASS_NAMES[classId] : "unknown";
        }

        // Convert to corner coordinates
        public float getLeft() { return x - w / 2; }
        public float getTop() { return y - h / 2; }
        public float getRight() { return x + w / 2; }
        public float getBottom() { return y + h / 2; }
    }

    public static class Detections {
        public final List<Detection> detections;

        public Detections(List<Detection> detections) {
            this.detections = detections;
        }

        /**
         * Draw bounding boxes and labels on the image
         */
        public void drawOn(Mat image) {
            System.out.printf("🖼️ Drawing %d detections on image size: %dx%d%n", 
                detections.size(), image.cols(), image.rows());
            
            for (Detection det : detections) {
                System.out.printf("📦 Final Detection: center(%.1f,%.1f) size(%.1f,%.1f) conf=%.3f%n", 
                    det.x, det.y, det.w, det.h, det.confidence);
                
                int left = (int) det.getLeft();
                int top = (int) det.getTop();
                int right = (int) det.getRight();
                int bottom = (int) det.getBottom();

                // Clamp coordinates to image bounds
                left = Math.max(0, Math.min(left, image.cols() - 1));
                top = Math.max(0, Math.min(top, image.rows() - 1));
                right = Math.max(left + 1, Math.min(right, image.cols()));
                bottom = Math.max(top + 1, Math.min(bottom, image.rows()));

                // Draw bounding box
                Point topLeft = new Point(left, top);
                Point bottomRight = new Point(right, bottom);
                Scalar boxColor = new Scalar(0, 255, 0); // Green
                Imgproc.rectangle(image, topLeft, bottomRight, boxColor, 3);

                // Draw label background
                String label = String.format("%s %.2f", det.className, det.confidence);
                Size labelSize = Imgproc.getTextSize(label, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, 2, null);
                Point labelPos = new Point(left, Math.max(top - 10, labelSize.height + 5));
                Point labelBg1 = new Point(left - 2, labelPos.y - labelSize.height - 5);
                Point labelBg2 = new Point(left + labelSize.width + 2, labelPos.y + 5);

                Imgproc.rectangle(image, labelBg1, labelBg2, boxColor, -1);

                // Draw text
                Scalar textColor = new Scalar(255, 255, 255); // White
                Imgproc.putText(image, label, labelPos, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
            }
        }
    }

    /**
     * Process YOLO v8 output for single-class UAV model
     * YOLOv8 output format: [batch_size, 4+num_classes, num_anchors]
     * For single class: [1, 5, 336] -> 336 detections with [x, y, w, h, confidence] each
     * BUT: YOLOv8 outputs in TRANSPOSED format: [x1,x2,...,x336, y1,y2,...,y336, w1,w2,...,w336, h1,h2,...,h336, c1,c2,...,c336]
     */
    public static Detections process(float[] output, int numDetections, int numClasses,
                                     float confThreshold, float nmsThreshold,
                                     Preprocessor.Input input) {

        DateTimeFormatter TS_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss.SSS")
                .withZone(ZoneId.systemDefault());
        String ts = "⏰ " + TS_FORMAT.format(Instant.now()) + " "; 

        System.out.print(ts);

        System.out.printf("🔍 Processing YOLO v8 output: Array length=%d, Expected shape=[1,5,336], conf_threshold=%.3f%n", 
            output.length, confThreshold);
        
        System.out.printf("📐 Image info: original=%dx%d, scale=%.3f, pad=(%.1f,%.1f)%n", 
            input.origWidth, input.origHeight, input.scale, input.padX, input.padY);

        List<Detection> allDetections = new ArrayList<>();
        
        // YOLOv8 TRANSPOSED format: output is [1680] flattened from [1, 5, 336]
        // Structure: [x1,x2,...,x336, y1,y2,...,y336, w1,w2,...,w336, h1,h2,...,h336, c1,c2,...,c336]
        int totalDetections = 336;
        int featuresPerDetection = 5; // x, y, w, h, confidence
        
        if (output.length != totalDetections * featuresPerDetection) {
            System.err.printf("❌ Unexpected output array length: %d (expected %d)%n", 
                output.length, totalDetections * featuresPerDetection);
        }

        int validDetections = 0;
        int processedDetections = 0;
        
        // Parse TRANSPOSED format
        for (int i = 0; i < totalDetections && i < output.length / featuresPerDetection; i++) {
            // In transposed format:
            // x values: output[0] to output[335]
            // y values: output[336] to output[671] 
            // w values: output[672] to output[1007]
            // h values: output[1008] to output[1343]
            // confidence values: output[1344] to output[1679]
            
            float x = output[i];                              // x coordinates: indices 0-335
            float y = output[i + totalDetections];            // y coordinates: indices 336-671
            float w = output[i + totalDetections * 2];        // w coordinates: indices 672-1007
            float h = output[i + totalDetections * 3];        // h coordinates: indices 1008-1343
            float confidence = output[i + totalDetections * 4]; // confidence: indices 1344-1679
            
            processedDetections++;
            
            // Debug first few detections to verify format
            if (processedDetections <= 3) {
                System.out.printf("🔍 Raw detection #%d: x=%.3f, y=%.3f, w=%.3f, h=%.3f, conf=%.3f%n", 
                    processedDetections, x, y, w, h, confidence);
            }
            
            // YOLOv8 confidence is typically already normalized (0-1), but check your model's output range
            float normalizedConfidence = confidence;
            if (confidence > 1.0f) {
                // If confidence is in 0-100 range, normalize it
                normalizedConfidence = confidence / 100.0f;
            }
            
            // Early filter - only process high-confidence detections
            if (normalizedConfidence > confThreshold) {
                validDetections++;
                
                // Debug first few valid detections
                if (validDetections <= 5) {
                    System.out.printf("🎯 Valid detection #%d: x=%.3f, y=%.3f, w=%.3f, h=%.3f, conf=%.3f%n", 
                        validDetections, x, y, w, h, normalizedConfidence);
                }
                
                // COORDINATE TRANSFORMATION
                // YOLOv8 typically outputs normalized coordinates (0-1), but your model might output pixel coordinates (0-128)
                // First, check if coordinates are normalized or in pixel space
                boolean isNormalized = (x <= 1.0f && y <= 1.0f && w <= 1.0f && h <= 1.0f);
                
                if (!isNormalized) {
                    // Coordinates are in letterbox pixel space (0-128)
                    if (validDetections <= 3) {
                        System.out.printf("📏 Letterbox pixel coordinates: x=%.1f, y=%.1f, w=%.1f, h=%.1f%n", 
                            x, y, w, h);
                    }
                } else {
                    // Coordinates are normalized (0-1), convert to letterbox pixel space
                    x *= 128;
                    y *= 128;
                    w *= 128;
                    h *= 128;
                    
                    if (validDetections <= 3) {
                        System.out.printf("📏 Converted to letterbox pixels: x=%.1f, y=%.1f, w=%.1f, h=%.1f%n", 
                            x, y, w, h);
                    }
                }
                
                // Transform from letterbox space back to original image space
                // Remove padding first
                x = x - (float) input.padX;
                y = y - (float) input.padY;
                
                // Scale back to original dimensions
                x = x / (float) input.scale;
                y = y / (float) input.scale;
                w = w / (float) input.scale;
                h = h / (float) input.scale;
                
                if (validDetections <= 3) {
                    System.out.printf("🔄 Final coordinates: x=%.1f, y=%.1f, w=%.1f, h=%.1f%n", 
                        x, y, w, h);
                }
                
                // Use normalized confidence for the detection object
                confidence = normalizedConfidence;
                
                // Sanity checks
                if (w <= 0 || h <= 0 || w > input.origWidth * 2 || h > input.origHeight * 2) {
                    if (validDetections <= 3) {
                        System.out.printf("⚠️ Invalid box dimensions, skipping%n");
                    }
                    continue;
                }
                
                // Ensure coordinates are within reasonable bounds (allow some margin for edge cases)
                x = Math.max(0, Math.min(x, input.origWidth));
                y = Math.max(0, Math.min(y, input.origHeight));
                
                allDetections.add(new Detection(x, y, w, h, confidence, 0));
            }
        }

        System.out.printf("📊 Processed %d detections, found %d above threshold (%.3f)%n", 
            processedDetections, validDetections, confThreshold);

        // Apply Non-Maximum Suppression
        List<Detection> finalDetections = applyNMS(allDetections, nmsThreshold);
        
        System.out.printf("📊 After NMS: %d final detections%n", finalDetections.size());

        return new Detections(finalDetections);
    }

    /**
     * Non-Maximum Suppression
     */
    private static List<Detection> applyNMS(List<Detection> detections, float nmsThreshold) {
        if (detections.isEmpty()) return detections;

        // Sort by confidence (highest first)
        detections.sort((a, b) -> Float.compare(b.confidence, a.confidence));
        List<Detection> result = new ArrayList<>();
        boolean[] suppressed = new boolean[detections.size()];

        for (int i = 0; i < detections.size(); i++) {
            if (suppressed[i]) continue;
            
            Detection det1 = detections.get(i);
            result.add(det1);
            
            System.out.printf("✅ Kept detection: conf=%.3f, box=(%.1f,%.1f,%.1f,%.1f)%n", 
                det1.confidence, det1.x, det1.y, det1.w, det1.h);

            // Suppress overlapping detections
            for (int j = i + 1; j < detections.size(); j++) {
                if (suppressed[j]) continue;
                
                Detection det2 = detections.get(j);
                float iou = calculateIoU(det1, det2);
                
                if (iou > nmsThreshold) {
                    suppressed[j] = true;
                    System.out.printf("🚫 Suppressed detection: conf=%.3f, IoU=%.3f%n", det2.confidence, iou);
                }
            }
        }

        return result;
    }

    /**
     * Intersection over Union
     */
    private static float calculateIoU(Detection a, Detection b) {
        float left = Math.max(a.getLeft(), b.getLeft());
        float top = Math.max(a.getTop(), b.getTop());
        float right = Math.min(a.getRight(), b.getRight());
        float bottom = Math.min(a.getBottom(), b.getBottom());

        if (left >= right || top >= bottom) return 0.0f;

        float intersection = (right - left) * (bottom - top);
        float areaA = a.w * a.h;
        float areaB = b.w * b.h;
        float union = areaA + areaB - intersection;

        return union > 0 ? intersection / union : 0.0f;
    }
}