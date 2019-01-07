package com.example.laptopuser.digitconizer;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.wonderkiln.camerakit.CameraKitError;
import com.wonderkiln.camerakit.CameraKitEvent;
import com.wonderkiln.camerakit.CameraKitEventListener;
import com.wonderkiln.camerakit.CameraKitImage;
import com.wonderkiln.camerakit.CameraKitVideo;
import com.wonderkiln.camerakit.CameraView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.IntBuffer;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView resultTextView;
    private TextView textView;
    private ImageView resultImageView;
    private CameraView cameraView;

    private static final String MODEL_PATH = "0.89055.tflite";
    private static final int INPUT_SIZE_1 = 28;
    private static final int INPUT_SIZE_2 = 56;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button btn = findViewById(R.id.btn);
        cameraView = findViewById(R.id.cameraView);
        resultImageView = findViewById(R.id.resultImageView);
        resultTextView = findViewById(R.id.resultTextView);
        textView = findViewById(R.id.textView);

        cameraView.addCameraKitListener(new CameraKitEventListener() {
            @Override
            public void onEvent(CameraKitEvent cameraKitEvent) {

            }

            @Override
            public void onError(CameraKitError cameraKitError) {

            }

            @Override
            public void onImage(CameraKitImage cameraKitImage) {

                Bitmap bmp1 = cameraKitImage.getBitmap();
                Bitmap bmp = Bitmap.createScaledBitmap(bmp1, INPUT_SIZE_2, INPUT_SIZE_2, false);
                int[] pix = classifier.convertBitmapToPixels(bmp);
                String fileContents = "";
                for (int i = 0; i < INPUT_SIZE_1; ++i) {
                    for (int j = 0; j < INPUT_SIZE_2; ++j) {
                        fileContents = fileContents + pix[i*INPUT_SIZE_2+j] + " ";
                    }
                    fileContents = fileContents + "\n";
                }

                File path = getApplicationContext().getExternalFilesDir(null);
                File file = new File(path, "image.txt");

                try {
                    FileOutputStream stream = new FileOutputStream(file);
                    stream.write(fileContents.getBytes());
                    stream.close();
                }
                catch (IOException e) {
                    Log.e("Exception", "File write failed: " + e.toString());
                }

                Bitmap bitmap = Bitmap.createBitmap(pix,INPUT_SIZE_2, INPUT_SIZE_1, Bitmap.Config.RGB_565);
                resultImageView.setImageBitmap(bitmap);

                long time_init = System.nanoTime();
                final List<Classifier.Recognition> results = classifier.recognizeImage(bmp);
                long time = System.nanoTime();
                float time_elapsed = (time-time_init)/1000000;
                String result = "Time elapsed: " + String.valueOf(time_elapsed) + " ms";
                textView.setText(result);
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

        initTensorFlow();

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

    private void initTensorFlow(){
        executor.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    classifier = ImageClassifier.create(
                            getAssets(),
                            MODEL_PATH,
                            INPUT_SIZE_1,
                            INPUT_SIZE_2);
                }catch (final Exception e) {
                    throw new RuntimeException("Error Initializing TensorFlow!", e);
                }
            }
        });
    }
}
