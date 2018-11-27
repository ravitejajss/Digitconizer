package com.example.laptopuser.digitconizer;

import android.graphics.Bitmap;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView resultTextView;
    private ImageView resultImageView;
    private CameraView cameraView;

    private static final String MODEL_PATH = "model.tflite";
    private static final String LABEL_PATH = "labels.txt";
    private static final int INPUT_SIZE = 28;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button btn = findViewById(R.id.btn);
        cameraView = findViewById(R.id.cameraView);
        resultImageView = findViewById(R.id.resultImageView);
        resultTextView = findViewById(R.id.resultTextView);

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bmp = cameraKitImage.getBitmap();
                bmp = Bitmap.createScaledBitmap(bmp, INPUT_SIZE, INPUT_SIZE, false);
                resultImageView.setImageBitmap(bmp);
                final List<Classifier.Recognition> results = classifier.recognizeImage(bmp);
                resultTextView.setText(results.toString());
            }

            @Override
            public void onVideo(CameraKitVideo cameraKitVideo) {

            }

        });

        btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                cameraView.captureImage();
            }
        });

        initTensorflow();
    }

    @Override
    protected void onResume() {
        super.onResume();
        cameraView.start();
    }

    @Override
    protected void onPause() {
        cameraView.stop();
        super.onPause();
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.execute(new Runnable() {
            @Override
            public void run() {
                classifier.close();
            }
        });
    }

    private void initTensorflow(){
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = ImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            LABEL_PATH,
                            INPUT_SIZE);
                }catch (final Exception e) {
                    throw new RuntimeException("Error Initializing TensorFlow!", e);
                }
            }
        });
    }
}