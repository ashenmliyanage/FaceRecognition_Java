package org.example;

import org.opencv.core.Core;

public class load {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
}
