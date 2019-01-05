package com.example.laptopuser.digitconizer;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import static android.graphics.Color.blue;
import static android.graphics.Color.red;

public class ImageClassifier implements Classifier {

    private static final int CLASSES = 10;
    private static final int MAX_RESULTS = 6;
    private static final float THRESHOLD = 0.1f;

    private Interpreter interpreter;
    private int inputSize;

    private ImageClassifier(){

    }

    static Classifier create(AssetManager assetManager,
                             String modelPath,
                             int inputSize) throws IOException {
        ImageClassifier classifier = new ImageClassifier();
        classifier.inputSize = inputSize;
        classifier.interpreter = new Interpreter(classifier.loadModelFile(assetManager, modelPath));

        return classifier;
    }


    @Override
    public List<Recognition> recognizeImage(Bitmap bitmap) {
        int[] intPix = convertBitmapToPixels(bitmap);
        float[][][][] inp = new float[1][inputSize][inputSize][1];
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                inp[0][i][j][0] = intPix[i*inputSize+j];
            }
        }
        float[][] result = new float[1][CLASSES];
        interpreter.run(inp, result);
        return getSortedResult(result);
    }

    @Override
    public int[] convertBitmapToPixels(Bitmap bitmap) {
        int[] intValues = new int[inputSize * inputSize];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int[] pixels = new int[inputSize*inputSize];
        for (int i = 0; i < inputSize*inputSize; ++i) {
            int pix = red(intValues[i]);
            if(pix>100){
                pixels[i] = 0;
            }else{
                pixels[i] = 255;
            }
        }
        return pixels;
    }

    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    @Override
    public void close() {
        interpreter.close();
        interpreter = null;
    }

    @SuppressLint("DefaultLocale")
    private List<Recognition> getSortedResult(float[][] labelProbArray) {

        PriorityQueue<Recognition> pq =
                new PriorityQueue<>(
                        MAX_RESULTS,
                        new Comparator<Recognition>() {
                            @Override
                            public int compare(Recognition lhs, Recognition rhs) {
                                return Float.compare(rhs.getConfidence(), lhs.getConfidence());
                            }
                        });

        for (int i = 0; i < CLASSES; ++i) {
            float confidence = (labelProbArray[0][i]);
            if (confidence > THRESHOLD) {
                pq.add(new Recognition("" + i,
                         Integer.toString(i),
                        confidence));
            }
        }

        final ArrayList<Recognition> recognitions = new ArrayList<>();
        int recognitionsSize = Math.min(pq.size(), MAX_RESULTS);
        for (int i = 0; i < recognitionsSize; ++i) {
            recognitions.add(pq.poll());
        }

        return recognitions;
    }
}
