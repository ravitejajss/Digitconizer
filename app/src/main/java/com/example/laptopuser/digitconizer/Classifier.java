package com.example.laptopuser.digitconizer;

import android.graphics.Bitmap;

import java.util.List;

public interface Classifier {

    class Recognition{

        private final String id;
        private final String title;
        private final Float confidence;

        public Recognition(
                final String id, final String title, final Float confidence) {
            this.id = id;
            this.title = title;
            this.confidence = confidence;
        }

        public Float getConfidence() {
            return confidence;
        }
    }

    List<Recognition> recognizeImage(Bitmap bitmap);


    void close();
}
