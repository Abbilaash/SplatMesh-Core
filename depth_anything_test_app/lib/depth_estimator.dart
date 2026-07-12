import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:onnxruntime/onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:image/image.dart' as img;

class DepthEstimator {
  OrtSession? _session;
  bool _isInitialized = false;

  bool get isInitialized => _isInitialized;

  /// Initialize and load the model.
  /// Copies the files from assets to application documents directory first
  /// so that ONNX Runtime can resolve the external weights file `model.data`.
  Future<void> initialize(Function(String) logCallback) async {
    try {
      logCallback("Locating app documents directory...");
      final docDir = await getApplicationDocumentsDirectory();
      
      final modelPath = '${docDir.path}/model.onnx';
      final modelFile = File(modelPath);

      logCallback("Checking if model file exists locally...");
      bool needsCopy = false;
      if (!await modelFile.exists()) {
        needsCopy = true;
      } else {
        // Double check file size to ensure previous copy wasn't interrupted.
        // The self-contained model with embedded weights is ~101MB.
        final modelSize = await modelFile.length();
        if (modelSize < 90000000) {
          needsCopy = true;
          logCallback("Local file exists but appears incomplete (size: $modelSize bytes). Recopying...");
        }
      }

      if (needsCopy) {
        logCallback("Copying self-contained model.onnx from assets to device storage (this might take a few seconds)...");
        final modelData = await rootBundle.load('assets/model.onnx');
        await modelFile.writeAsBytes(modelData.buffer.asUint8List(
          modelData.offsetInBytes,
          modelData.lengthInBytes,
        ));
        logCallback("Model assets successfully copied!");
      }

      logCallback("Initializing ONNX Runtime environment...");
      OrtEnv.instance.init();

      logCallback("Configuring ONNX session options...");
      final sessionOptions = OrtSessionOptions();
      bool isNnapiAdded = false;
      
      // Try to enable NNAPI for NPU/GPU hardware acceleration on Android
      try {
        sessionOptions.appendNnapiProvider(NnapiFlags.useNone);
        logCallback("Android NNAPI hardware acceleration delegate added!");
        isNnapiAdded = true;
      } catch (e) {
        logCallback("NNAPI delegate init failed: $e. Falling back to CPU.");
      }

      logCallback("Loading ONNX model session from device storage...");
      try {
        _session = OrtSession.fromFile(modelFile, sessionOptions);
        _isInitialized = true;
        logCallback("Depth model successfully loaded${isNnapiAdded ? ' with NNAPI acceleration' : ''}!");
      } catch (sessionError) {
        if (isNnapiAdded) {
          logCallback("Failed to load session with NNAPI: $sessionError. Retrying on CPU...");
          final cpuOptions = OrtSessionOptions();
          _session = OrtSession.fromFile(modelFile, cpuOptions);
          _isInitialized = true;
          logCallback("Depth model successfully loaded on CPU!");
        } else {
          rethrow;
        }
      }
    } catch (e) {
      logCallback("Error during initialization: $e");
      _isInitialized = false;
      rethrow;
    }
  }

  /// Run depth estimation on an image file path.
  /// Returns a map with performance metrics and the file path of the resulting grayscale PNG.
  Future<Map<String, dynamic>> estimateDepth(String imagePath) async {
    if (!_isInitialized || _session == null) {
      throw StateError("DepthEstimator is not initialized.");
    }

    final stopwatch = Stopwatch()..start();

    // 1. Decode original image
    final bytes = await File(imagePath).readAsBytes();
    final originalImage = img.decodeImage(bytes);
    if (originalImage == null) {
      throw ArgumentError("Failed to decode image at $imagePath");
    }

    final int preprocessStart = stopwatch.elapsedMilliseconds;

    // 2. Preprocessing: Resize to 518x518
    final resizedImage = img.copyResize(
      originalImage,
      width: 518,
      height: 518,
      interpolation: img.Interpolation.linear,
    );

    // 3. Normalization: Planar format [1, 3, 518, 518]
    // ImageNet stats: mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    final int size = 518 * 518;
    final Float32List inputData = Float32List(3 * size);

    const List<double> mean = [0.485, 0.456, 0.406];
    const List<double> std = [0.229, 0.224, 0.225];

    final int rOffset = 0;
    final int gOffset = size;
    final int bOffset = 2 * size;

    for (int y = 0; y < 518; y++) {
      for (int x = 0; x < 518; x++) {
        final pixel = resizedImage.getPixel(x, y);
        
        // Scale values to [0.0, 1.0]
        final double r = pixel.r / 255.0;
        final double g = pixel.g / 255.0;
        final double b = pixel.b / 255.0;

        final int index = y * 518 + x;
        inputData[rOffset + index] = (r - mean[0]) / std[0];
        inputData[gOffset + index] = (g - mean[1]) / std[1];
        inputData[bOffset + index] = (b - mean[2]) / std[2];
      }
    }

    final int preprocessTime = stopwatch.elapsedMilliseconds - preprocessStart;
    final int inferenceStart = stopwatch.elapsedMilliseconds;

    // 4. Create ONNX Tensor
    final inputShape = [1, 3, 518, 518];
    final ortValueInput = OrtValueTensor.createTensorWithDataList(
      inputData,
      inputShape,
    );

    final inputName = _session!.inputNames.first;
    final inputs = {inputName: ortValueInput};

    // Run Model Session
    final runOptions = OrtRunOptions();
    final outputs = _session!.run(runOptions, inputs);
    
    // Release input tensor memory in FFI
    ortValueInput.release();

    final int inferenceTime = stopwatch.elapsedMilliseconds - inferenceStart;
    final int postprocessStart = stopwatch.elapsedMilliseconds;

    // 5. Postprocessing output
    if (outputs.isEmpty || outputs.first == null) {
      throw StateError("Model inference returned no outputs.");
    }

    final OrtValueTensor outputTensor = outputs.first as OrtValueTensor;
    final List<dynamic> outputDataList = outputTensor.value;
    outputTensor.release();

    // Flatten multi-dimensional output list
    final List<double> flattenedOutput = _flattenList(outputDataList);

    if (flattenedOutput.length != 518 * 518) {
      throw StateError("Expected output size ${518 * 518}, but got ${flattenedOutput.length}");
    }

    // Min-Max Normalization to scale output to [0.0, 1.0]
    double minVal = flattenedOutput[0];
    double maxVal = flattenedOutput[0];
    for (int i = 1; i < flattenedOutput.length; i++) {
      final double val = flattenedOutput[i];
      if (val < minVal) minVal = val;
      if (val > maxVal) maxVal = val;
    }

    final double range = maxVal - minVal + 1e-8;

    // 6. Generate Grayscale Image
    final depthImage = img.Image(width: 518, height: 518);
    for (int y = 0; y < 518; y++) {
      for (int x = 0; x < 518; x++) {
        final int index = y * 518 + x;
        final double normalizedDepth = (flattenedOutput[index] - minVal) / range;
        
        // Map depth [0.0, 1.0] to grayscale pixel [0, 255]
        final int gray = (normalizedDepth * 255.0).clamp(0, 255).toInt();
        depthImage.setPixelRgb(x, y, gray, gray, gray);
      }
    }

    // 7. Save depth map to a local temporary file
    final tempDir = await getTemporaryDirectory();
    final String depthMapPath = '${tempDir.path}/depth_${DateTime.now().millisecondsSinceEpoch}.png';
    
    final pngBytes = img.encodePng(depthImage);
    await File(depthMapPath).writeAsBytes(pngBytes);

    final int postprocessTime = stopwatch.elapsedMilliseconds - postprocessStart;
    final int totalTime = stopwatch.elapsedMilliseconds;

    stopwatch.stop();

    return {
      'depthMapPath': depthMapPath,
      'preprocessTimeMs': preprocessTime,
      'inferenceTimeMs': inferenceTime,
      'postprocessTimeMs': postprocessTime,
      'totalTimeMs': totalTime,
    };
  }

  /// Helper utility to flatten nested Lists generated by OrtValueTensor.value
  List<double> _flattenList(List<dynamic> list) {
    final List<double> result = [];
    _flattenHelper(list, result);
    return result;
  }

  void _flattenHelper(List<dynamic> list, List<double> result) {
    for (var item in list) {
      if (item is List) {
        _flattenHelper(item, result);
      } else if (item is num) {
        result.add(item.toDouble());
      }
    }
  }

  void dispose() {
    _session?.release();
    OrtEnv.instance.release();
  }
}
