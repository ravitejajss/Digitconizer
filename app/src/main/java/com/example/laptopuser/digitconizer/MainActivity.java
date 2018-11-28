package com.example.laptopuser.digitconizer;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
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

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

import static android.graphics.Color.red;

public class MainActivity extends AppCompatActivity {

    private Classifier classifier;

    private Executor executor = Executors.newSingleThreadExecutor();
    private TextView resultTextView;
    private ImageView resultImageView;
    private CameraView cameraView;

    private static final String MODEL_PATH = "converted_model1.tflite";
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
                float[][][][] pix = classifier.convertBitmapToByteBuffer(bmp);
                String s = "";
                for (int i = 0; i < INPUT_SIZE; ++i) {
                    for (int j = 0; j < INPUT_SIZE; ++j) {
                        s = s + pix[0][i][j][0] + " ";
                    }
                }
                File path = getApplicationContext().getExternalFilesDir(null);
                File file = new File(path, "imgData.txt");
                try {
                    FileOutputStream stream = new FileOutputStream(file);
                    stream.write(s.getBytes());
                    stream.close();
                } catch (IOException e){
                    e.printStackTrace();
                }
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
                            LABEL_PATH,
                            INPUT_SIZE);
                }catch (final Exception e) {
                    throw new RuntimeException("Error Initializing TensorFlow!", e);
                }
            }
        });
    }

    public Bitmap toGrayscale(Bitmap bmpOriginal)
    {
        int width, height;
        height = bmpOriginal.getHeight();
        width = bmpOriginal.getWidth();

        Bitmap bmpGrayscale = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        Canvas c = new Canvas(bmpGrayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        ColorMatrixColorFilter f = new ColorMatrixColorFilter(cm);
        paint.setColorFilter(f);
        c.drawBitmap(bmpOriginal, 0, 0, paint);
        return bmpGrayscale;
    }
}
