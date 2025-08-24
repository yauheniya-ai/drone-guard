package com.example.droneguard.yolo;

import ai.onnxruntime.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import jakarta.annotation.PostConstruct;

import java.io.IOException;
import java.util.Collections;
import java.util.Map;

@Service
public class YOLOOnnxService {

    private final String modelPath;
    private final int imgSize;
    private final float confThreshold;
    private final float nmsThreshold;

    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;
    private long[] inputShape;

    public YOLOOnnxService(@Value("${droneguard.model}") String modelPath,
                           @Value("${droneguard.imgsz}") int imgSize,
                           @Value("${droneguard.conf:0.5}") float confThreshold,
                           @Value("${droneguard.nms:0.4}") float nmsThreshold) {
        this.modelPath = modelPath;
        this.imgSize = imgSize;
        this.confThreshold = confThreshold;
        this.nmsThreshold = nmsThreshold;
    }

    @PostConstruct
    public void init() throws IOException, OrtException {
        System.out.println("🚀 Initializing YOLO model: " + modelPath);

        env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        session = env.createSession(modelPath, opts);

        inputName = session.getInputNames().iterator().next();
        inputShape = new long[]{1, 3, imgSize, imgSize};

        System.out.printf("✅ Model loaded successfully:%n");
        System.out.printf("   Input name: %s%n", inputName);
        System.out.printf("   Input shape (assumed): [%s]%n", java.util.Arrays.toString(inputShape));
        System.out.printf("   Image size: %d%n", imgSize);
        System.out.printf("   Confidence threshold: %.2f%n", confThreshold);
        System.out.printf("   NMS threshold: %.2f%n", nmsThreshold);
    }

    public Postprocessor.Detections infer(Preprocessor.Input input) throws OrtException {
        // Prepare input tensor [1,3,H,W]
        float[][][][] tensorData = new float[1][3][imgSize][imgSize];
        int idx = 0;
        for (int c = 0; c < 3; c++)
            for (int h = 0; h < imgSize; h++)
                for (int w = 0; w < imgSize; w++)
                    tensorData[0][c][h][w] = input.data[idx++];

        OnnxTensor inputTensor = OnnxTensor.createTensor(env, tensorData);
        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, inputTensor);
        OrtSession.Result result = session.run(inputs);

        OnnxTensor outputTensor = (OnnxTensor) result.get(0);
        Object outputValue = outputTensor.getValue();
        float[] flatOutput;
        int numDetections;

        if (outputValue instanceof float[][][])
            flatOutput = flatten3D((float[][][]) outputValue);
        else if (outputValue instanceof float[][])
            flatOutput = flatten2D((float[][]) outputValue);
        else
            throw new RuntimeException("Unexpected output type: " + outputValue.getClass());

        // Use 80 classes (COCO) or adjust as needed
        int numClasses = 80;
        numDetections = (outputValue instanceof float[][][]) ? ((float[][][]) outputValue)[0].length : ((float[][]) outputValue).length;

        Postprocessor.Detections detections = Postprocessor.process(flatOutput, numDetections, numClasses, confThreshold, nmsThreshold, input);

        inputTensor.close();
        result.close();
        return detections;
    }

    private float[] flatten2D(float[][] array) {
        int dim1 = array.length;
        int dim2 = array[0].length;
        float[] result = new float[dim1 * dim2];
        int idx = 0;
        for (int i = 0; i < dim1; i++)
            for (int j = 0; j < dim2; j++)
                result[idx++] = array[i][j];
        return result;
    }

    private float[] flatten3D(float[][][] array) {
        int batch = array.length;
        int dim1 = array[0].length;
        int dim2 = array[0][0].length;
        float[] result = new float[batch * dim1 * dim2];
        int idx = 0;
        for (int b = 0; b < batch; b++)
            for (int i = 0; i < dim1; i++)
                for (int j = 0; j < dim2; j++)
                    result[idx++] = array[b][i][j];
        return result;
    }

    public void cleanup() throws OrtException {
        if (session != null) session.close();
        if (env != null) env.close();
    }
}
