package com.example.droneguard.video;

import com.example.droneguard.yolo.*;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.videoio.VideoCapture;
import org.opencv.videoio.Videoio;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import jakarta.annotation.PostConstruct;
import java.io.File;
import java.util.concurrent.atomic.AtomicReference;

@Component
public class VideoCaptureLoop {
    private final YOLOOnnxService yolo;
    private final String source;
    private final int imgSize;
    
    // Single atomic reference for thread-safe frame access
    private final AtomicReference<byte[]> currentFrame = new AtomicReference<>();
    private final byte[] placeholder;
    
    private volatile boolean running = true;
    private volatile long frameCounter = 0;
    private volatile long lastLogTime = System.currentTimeMillis();

    static {
        nu.pattern.OpenCV.loadShared();
    }

    public VideoCaptureLoop(YOLOOnnxService yolo,
                           @Value("${droneguard.source}") String source,
                           @Value("${droneguard.imgsz}") int imgSize) {
        this.yolo = yolo;
        this.source = source;
        this.imgSize = imgSize;
        
        // Create simple placeholder
        Mat greenMat = new Mat(240, 320, CvType.CV_8UC3, new Scalar(0, 255, 0));
        MatOfByte matOfByte = new MatOfByte();
        Imgcodecs.imencode(".jpg", greenMat, matOfByte);
        placeholder = matOfByte.toArray();
        
        // Clean up immediately
        greenMat.release();
        matOfByte.release();
        
        // Set initial frame
        currentFrame.set(placeholder);
        
        System.out.println("✅ VideoCaptureLoop initialized - placeholder size: " + placeholder.length);
    }

    public byte[] getLatestJpeg() {
        byte[] frame = currentFrame.get();
        return (frame != null) ? frame : placeholder;
    }

    @PostConstruct
    public void start() {
        System.out.println("🚀 Starting single-threaded video processing...");
        
        Thread captureThread = new Thread(this::captureAndProcess, "video-main");
        captureThread.setDaemon(true);
        captureThread.start();
    }

    private void captureAndProcess() {
        VideoCapture cap = null;
        Mat frame = new Mat();
        boolean isCamera = source.matches("\\d+");
        
        try {
            // Initialize capture
            cap = initializeCapture();
            if (cap == null || !cap.isOpened()) {
                System.err.println("❌ Failed to initialize video capture");
                return;
            }
            
            System.out.println("✅ Video capture started successfully");
            
            // Main processing loop - everything in one thread to avoid synchronization issues
            while (running && !Thread.currentThread().isInterrupted()) {
                
                // 1. Capture frame
                if (!cap.read(frame) || frame.empty()) {
                    if (isCamera) {
                        System.out.println("⚠️ Camera frame read failed, retrying...");
                        Thread.sleep(50);
                        continue;
                    } else {
                        // Restart video file
                        cap.set(Videoio.CAP_PROP_POS_FRAMES, 0);
                        continue;
                    }
                }
                
                frameCounter++;
                
                // 2. Create working copy for processing
                Mat workingFrame = null;
                byte[] jpegBytes = null;
                
                try {
                    workingFrame = frame.clone();
                    
                    // 3. YOLO processing (with error handling)
                    try {
                        Preprocessor.Input input = Preprocessor.letterbox(frame, imgSize);
                        Postprocessor.Detections detections = yolo.infer(input);
                        detections.drawOn(workingFrame);
                    } catch (Exception e) {
                        // If YOLO fails, continue with raw frame
                        System.err.println("⚠️ YOLO failed: " + e.getMessage());
                    }
                    
                    // 4. Encode to JPEG
                    MatOfByte matOfByte = new MatOfByte();
                    if (Imgcodecs.imencode(".jpg", workingFrame, matOfByte)) {
                        jpegBytes = matOfByte.toArray();
                    } else {
                        System.err.println("⚠️ JPEG encoding failed");
                    }
                    matOfByte.release();
                    
                } finally {
                    // Always clean up working frame
                    if (workingFrame != null) {
                        workingFrame.release();
                    }
                }
                
                // 5. Update current frame atomically
                if (jpegBytes != null && jpegBytes.length > 0) {
                    currentFrame.set(jpegBytes);
                }
                
                // 6. Logging (every 5 seconds)
                long currentTime = System.currentTimeMillis();
                if (currentTime - lastLogTime > 5000) {
                    System.out.printf("📽️ Processed %d frames, latest: %d bytes%n", 
                                    frameCounter, jpegBytes != null ? jpegBytes.length : 0);
                    lastLogTime = currentTime;
                }
                
                // 7. Frame rate control (~30 FPS)
                Thread.sleep(33);
            }
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            System.out.println("🛑 Video processing interrupted");
        } catch (Exception e) {
            System.err.println("❌ Fatal error in video processing: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Cleanup
            running = false;
            if (cap != null && cap.isOpened()) {
                cap.release();
                System.out.println("✅ Video capture released");
            }
            frame.release();
            System.out.printf("✅ Video processing ended after %d frames%n", frameCounter);
        }
    }
    
    private VideoCapture initializeCapture() {
        try {
            // Try as camera index
            int cameraIndex = Integer.parseInt(source);
            System.out.println("📹 Opening camera: " + cameraIndex);
            
            VideoCapture cap = new VideoCapture(cameraIndex);
            if (cap.isOpened()) {
                // Get camera properties
                double width = cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
                double height = cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
                System.out.printf("✅ Camera opened: %.0fx%.0f%n", width, height);
                return cap;
            } else {
                cap.release();
                System.err.println("❌ Could not open camera: " + cameraIndex);
                return null;
            }
            
        } catch (NumberFormatException e) {
            // Try as file path
            System.out.println("📁 Opening video file: " + source);
            
            File file = new File(source);
            if (!file.exists()) {
                System.err.println("❌ File not found: " + source);
                return null;
            }
            
            VideoCapture cap = new VideoCapture(source);
            if (cap.isOpened()) {
                double fps = cap.get(Videoio.CAP_PROP_FPS);
                double width = cap.get(Videoio.CAP_PROP_FRAME_WIDTH);
                double height = cap.get(Videoio.CAP_PROP_FRAME_HEIGHT);
                System.out.printf("✅ Video opened: %.0fx%.0f @ %.1f FPS%n", width, height, fps);
                return cap;
            } else {
                cap.release();
                System.err.println("❌ Could not open video: " + source);
                return null;
            }
        }
    }

    public void stop() {
        System.out.println("🛑 Stopping video capture...");
        running = false;
    }
    
    public String getStats() {
        long timeSinceLastLog = System.currentTimeMillis() - lastLogTime;
        return String.format("Running: %s, Frames: %d, Last activity: %dms ago", 
                           running, frameCounter, timeSinceLastLog);
    }
}