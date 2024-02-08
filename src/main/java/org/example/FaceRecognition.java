package org.example;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;

public class FaceRecognition {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        CascadeClassifier faceCascade = new CascadeClassifier();
        faceCascade.load("E:\\Face_regec\\src\\main\\java\\org\\example\\haarcascade_frontalface_default (1).xml"); // Change the path as necessary

        String knownFaceImagePath = "E:\\Face_regec\\src\\main\\resources\\WIN_20240206_17_30_27_Pro.jpg"; // Change the path as necessary
        String knownFaceName = "Ashen";

        Mat knownFaceImage = Imgcodecs.imread(knownFaceImagePath);
        if (knownFaceImage.empty()) {
            System.out.println("Error: Failed to load known face image.");
            return;
        }

        Mat knownFaceGray = new Mat();
        Imgproc.cvtColor(knownFaceImage, knownFaceGray, Imgproc.COLOR_BGR2GRAY);

        MatOfRect faces = new MatOfRect();
        faceCascade.detectMultiScale(knownFaceGray, faces);
        if (faces.toArray().length == 0) {
            System.out.println("Error: No face detected in the known face image.");
            return;
        }
        Rect face = faces.toArray()[0];
        Mat roi = new Mat(knownFaceGray, face);
        Mat faceEncoding = new Mat();
        Imgproc.resize(roi, faceEncoding, new Size(150, 150));

        VideoCapture capture = new VideoCapture(0);

        if (!capture.isOpened()) {
            System.out.println("Error: Camera not detected");
            return;
        }

        Mat frame = new Mat();
        while (true) {
            capture.read(frame);
            if (frame.empty()) {
                break;
            }

            Mat grayFrame = new Mat();
            Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY); // Convert frame to grayscale

            MatOfRect detectedFaces = new MatOfRect();
            faceCascade.detectMultiScale(grayFrame, detectedFaces);

            for (Rect rect : detectedFaces.toArray()) {
                Mat faceROI = new Mat(grayFrame, rect);
                Mat resizedFace = new Mat();
                Imgproc.resize(faceROI, resizedFace, new Size(150, 150));

                double result = Core.norm(faceEncoding, resizedFace, Core.NORM_L2);
                String name = (result < 3000) ? knownFaceName : "Unknown";

                if (name.equals(knownFaceName)) System.out.println("True");
                Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 0, 0), 2);
                Imgproc.putText(frame, name, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 0.9, new Scalar(36, 255, 12), 2);
            }

            HighGui.imshow("Face Recognition", frame);

            if (HighGui.waitKey(1) == 'q') {
                break;
            }
        }

        capture.release();
        HighGui.destroyAllWindows();
    }
}
