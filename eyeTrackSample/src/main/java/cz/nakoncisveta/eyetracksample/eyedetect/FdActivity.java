package cz.nakoncisveta.eyetracksample.eyedetect;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.view.WindowManager;
import android.widget.SeekBar;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;


public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";

    private static final Scalar GREEN_COLOR = new Scalar(0, 255, 0, 255);
    private static final Scalar RED_COLOR = new Scalar(255, 0, 0, 255);
    private static final Scalar WHITE_COLOR = new Scalar(255, 255, 255, 255);
    private static final Scalar YELLOW_COLOR = new Scalar(255, 255, 0, 255);

    private static final int TM_SQDIFF = 0;
    private static final int TM_SQDIFF_NORMED = 1;
    private static final int TM_CCOEFF = 2;
    private static final int TM_CCOEFF_NORMED = 3;
    private static final int TM_CCORR = 4;
    private static final int TM_CCORR_NORMED = 5;

    private static final float RELATIVE_FACE_SIZE = 0.2f;

    private int learn_frames = 0;
    private Mat templateR;
    private Mat templateL;
    int method = 0;

    private Mat mRgba;
    private Mat mGray;
    private CascadeClassifier faceClassifier;
    private CascadeClassifier eyeClassifier;

    private CameraBridgeViewBase mOpenCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i(TAG, "OpenCV loaded successfully");
                try {
                    InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                    File faceClassifierFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                    is.close();

                    InputStream ise = getResources().openRawResource(R.raw.haarcascade_lefteye_2splits);
                    File cascadeDirEye = getDir("cascade", Context.MODE_PRIVATE);
                    File eyeClassifierFile = new File(cascadeDirEye, "haarcascade_lefteye_2splits.xml");
                    ise.close();

                    faceClassifier = new CascadeClassifier(faceClassifierFile.getAbsolutePath());
                    if (faceClassifier.empty()) {
                        Log.e(TAG, "Failed to load cascade classifier");
                        faceClassifier = null;
                    } else
                        Log.i(TAG, "Loaded cascade classifier from " + faceClassifierFile.getAbsolutePath());

                    eyeClassifier = new CascadeClassifier(eyeClassifierFile.getAbsolutePath());
                    if (eyeClassifier.empty()) {
                        Log.e(TAG, "Failed to load cascade classifier for eye");
                        eyeClassifier = null;
                    } else
                        Log.i(TAG, "Loaded cascade classifier from " + eyeClassifierFile.getAbsolutePath());

                    cascadeDir.delete();
                    cascadeDirEye.delete();

                } catch (IOException e) {
                    Log.e(TAG, "Failed to load cascade classifier !");
                    e.printStackTrace();
                }
                mOpenCvCameraView.enableFpsMeter();
                mOpenCvCameraView.setCameraIndex(1);
                mOpenCvCameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        mOpenCvCameraView = findViewById(R.id.fd_activity_surface_view);
        mOpenCvCameraView.setCvCameraViewListener(this);

        SeekBar mMethodSeekbar = findViewById(R.id.methodSeekBar);
        final TextView methodValue = findViewById(R.id.method);

        mMethodSeekbar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {

            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }

            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {
                // TODO Auto-generated method stub
            }

            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                method = progress;
                switch (method) {
                    case 0:
                        methodValue.setText("TM_SQDIFF");
                        break;
                    case 1:
                        methodValue.setText("TM_SQDIFF_NORMED");
                        break;
                    case 2:
                        methodValue.setText("TM_CCOEFF");
                        break;
                    case 3:
                        methodValue.setText("TM_CCOEFF_NORMED");
                        break;
                    case 4:
                        methodValue.setText("TM_CCORR");
                        break;
                    case 5:
                        methodValue.setText("TM_CCORR_NORMED");
                        break;
                }
            }
        });
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    public void onDestroy() {
        super.onDestroy();
        mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mGray = new Mat();
        mRgba = new Mat();
    }

    public void onCameraViewStopped() {
        mGray.release();
        mRgba.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();

        int frameHeight = mGray.rows();
        int mAbsoluteFaceSize = Math.round(frameHeight * RELATIVE_FACE_SIZE);

        MatOfRect faces = new MatOfRect();

        if (faceClassifier != null)
            faceClassifier.detectMultiScale(mGray, faces, 1.1, 2, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                    new Size(mAbsoluteFaceSize, mAbsoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();
        for (Rect rect : facesArray) {
            Imgproc.rectangle(mRgba, rect.tl(), rect.br(), GREEN_COLOR, 3);

            int x = rect.x + rect.width / 16;
            int y = (int) (rect.y + (rect.height / 4.5));
            int width = rect.width / 2 - rect.width / 16;
            int height = rect.height / 3;

            // split eye areas
            Rect eyeAreaL = new Rect(x + width, y, width, height);
            Rect eyeAreaR = new Rect(x, y, width, height);

            // draw the area - preview types: mGray or mRgba
            Imgproc.rectangle(mRgba, eyeAreaL.tl(), eyeAreaL.br(), RED_COLOR, 2);
            Imgproc.rectangle(mRgba, eyeAreaR.tl(), eyeAreaR.br(), RED_COLOR, 2);

            if (learn_frames < 10) {
                templateL = getTemplate(eyeClassifier, eyeAreaL, 24);
                templateR = getTemplate(eyeClassifier, eyeAreaR, 24);
                learn_frames++;
            } else {
                // Learning finished, use the new templates for template matching
                matchEye(eyeAreaL, templateL, method);
                matchEye(eyeAreaR, templateR, method);
            }
        }

        return mRgba;
    }

    private void matchEye(Rect area, Mat mTemplate, int method) {
        Point matchLoc;
        Mat mROI = mGray.submat(area);

        int result_cols = mROI.cols() - mTemplate.cols() + 1;
        int result_rows = mROI.rows() - mTemplate.rows() + 1;

        // Check for bad template size
        if (mTemplate.cols() == 0 || mTemplate.rows() == 0)
            return;

        Mat mResult = new Mat(result_cols, result_rows, CvType.CV_8U);

        switch (method) {
            case TM_SQDIFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF);
                break;
            case TM_SQDIFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_SQDIFF_NORMED);
                break;
            case TM_CCOEFF:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF);
                break;
            case TM_CCOEFF_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCOEFF_NORMED);
                break;
            case TM_CCORR:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR);
                break;
            case TM_CCORR_NORMED:
                Imgproc.matchTemplate(mROI, mTemplate, mResult, Imgproc.TM_CCORR_NORMED);
                break;
        }

        Core.MinMaxLocResult mmres = Core.minMaxLoc(mResult);
        // there is difference in matching methods - best match is max/min value
        if (method == TM_SQDIFF || method == TM_SQDIFF_NORMED)
            matchLoc = mmres.minLoc;
        else
            matchLoc = mmres.maxLoc;

        Point matchLoc_tx = new Point(matchLoc.x + area.x, matchLoc.y + area.y);
        Point matchLoc_ty = new Point(matchLoc.x + mTemplate.cols() + area.x, matchLoc.y + mTemplate.rows() + area.y);

        Imgproc.rectangle(mRgba, matchLoc_tx, matchLoc_ty, YELLOW_COLOR);
    }

    private Mat getTemplate(CascadeClassifier classifier, Rect area, int size) {
        Mat template = new Mat();
        Mat mROI = mGray.submat(area);
        MatOfRect eyes = new MatOfRect();
        Point iris = new Point();

        classifier.detectMultiScale(mROI, eyes, 1.15, 2,
                Objdetect.CASCADE_FIND_BIGGEST_OBJECT | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(30, 30), new Size());

        Rect[] eyesArray = eyes.toArray();
        for (Rect eye : eyesArray) {
            eye.x += area.x;
            eye.y += area.y;
            Rect eye_only_rectangle = new Rect(
                    (int) eye.tl().x,
                    (int) (eye.tl().y + eye.height * 0.4),
                    eye.width,
                    (int) (eye.height * 0.6));
            mROI = mGray.submat(eye_only_rectangle);
            Mat vyrez = mRgba.submat(eye_only_rectangle);

            Core.MinMaxLocResult mmG = Core.minMaxLoc(mROI);

            Imgproc.circle(vyrez, mmG.minLoc, 2, WHITE_COLOR, 2);
            iris.x = mmG.minLoc.x + eye_only_rectangle.x;
            iris.y = mmG.minLoc.y + eye_only_rectangle.y;

            Rect eye_template = new Rect((int) iris.x - size / 2, (int) iris.y - size / 2, size, size);
            Imgproc.rectangle(mRgba, eye_template.tl(), eye_template.br(), RED_COLOR, 2);

            template = (mGray.submat(eye_template)).clone();
            return template;
        }
        return template;
    }

    public void onRecreateClick(View v) {
        learn_frames = 0;
    }

}
