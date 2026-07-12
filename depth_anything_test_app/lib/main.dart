import 'dart:async';
import 'dart:io';
import 'dart:ui' as ui;
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:camera/camera.dart';
import 'depth_estimator.dart';
import 'live_stream_manager.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const DepthTestApp());
}

class DepthTestApp extends StatelessWidget {
  const DepthTestApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SplatMesh Scanner',
      theme: ThemeData.dark().copyWith(
        primaryColor: Colors.indigo,
        scaffoldBackgroundColor: const Color(0xFF0B0E14),
        colorScheme: const ColorScheme.dark(
          primary: Colors.indigoAccent,
          secondary: Colors.cyanAccent,
          surface: Color(0xFF161A23),
          onSurface: Colors.white,
        ),
        cardTheme: const CardThemeData(
          color: Color(0xFF161A23),
          margin: EdgeInsets.symmetric(vertical: 8),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.all(Radius.circular(16)),
          ),
        ),
      ),
      home: const MainDashboardScreen(),
    );
  }
}

class MainDashboardScreen extends StatefulWidget {
  const MainDashboardScreen({super.key});

  @override
  State<MainDashboardScreen> createState() => _MainDashboardScreenState();
}

class _MainDashboardScreenState extends State<MainDashboardScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  // --- Offline Tab State ---
  final DepthEstimator _estimator = DepthEstimator();
  final ImagePicker _picker = ImagePicker();
  String? _pickedImagePath;
  String? _depthMapPath;
  bool _isProcessing = false;
  bool _isModelLoading = false;
  int? _preprocessTime;
  int? _inferenceTime;
  int? _postprocessTime;
  int? _totalTime;
  final List<String> _logsOffline = [];

  // --- Live Stream Tab State ---
  final LiveStreamManager _streamManager = LiveStreamManager();
  CameraController? _cameraController;
  List<CameraDescription> _cameras = [];
  bool _isCameraInitialized = false;
  final List<String> _logsStream = [];
  final GlobalKey _boundaryKey = GlobalKey();
  Timer? _streamTimer;

  // Controllers for IP/Port Configuration
  late TextEditingController _ipController;
  late TextEditingController _portController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _ipController = TextEditingController(text: _streamManager.ip);
    _portController = TextEditingController(text: _streamManager.port.toString());

    // Setup logging and callbacks for Stream Manager
    _streamManager.onLogReceived = (msg) {
      _logStream(msg);
    };
    _streamManager.onConnectionStateChanged = (state) {
      if (mounted) setState(() {});
    };
    _streamManager.onTelemetryUpdated = (p, r, y, frames) {
      if (mounted) setState(() {});
    };

    // Tab switcher listener to handle battery/sensor usage
    _tabController.addListener(() {
      if (_tabController.index == 1) {
        // Switch to streaming tab: start sensors & camera
        _streamManager.startSensors();
        _initCamera();
      } else {
        // Switch to offline tab: stop sensors, camera, and streaming
        _stopStreaming();
        _streamManager.stopSensors();
        _disposeCamera();
      }
    });

    _requestPermissions();
    _loadModel();
  }

  @override
  void dispose() {
    _tabController.dispose();
    _ipController.dispose();
    _portController.dispose();
    _estimator.dispose();
    _stopStreaming();
    _streamManager.stopSensors();
    _disposeCamera();
    super.dispose();
  }

  // --- Helpers ---
  void _logOffline(String message) {
    if (mounted) {
      setState(() {
        _logsOffline.insert(
            0, "[${DateTime.now().toString().substring(11, 19)}] $message");
      });
    }
  }

  void _logStream(String message) {
    if (mounted) {
      setState(() {
        _logsStream.insert(0, message);
      });
    }
  }

  Future<void> _requestPermissions() async {
    await Permission.camera.request();
    await Permission.photos.request();
  }

  // --- Offline Tab Code ---
  Future<void> _loadModel() async {
    setState(() {
      _isModelLoading = true;
    });
    _logOffline("Initializing DepthEstimator...");
    try {
      await _estimator.initialize((msg) {
        _logOffline(msg);
      });
      _logOffline("Ready to estimate depth.");
    } catch (e) {
      _logOffline("Error loading model: $e");
    } finally {
      setState(() {
        _isModelLoading = false;
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    if (!_estimator.isInitialized) {
      _logOffline("Cannot pick image: DepthEstimator is not initialized yet.");
      return;
    }

    try {
      final XFile? image = await _picker.pickImage(
        source: source,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (image != null) {
        setState(() {
          _pickedImagePath = image.path;
          _depthMapPath = null;
          _preprocessTime = null;
          _inferenceTime = null;
          _postprocessTime = null;
          _totalTime = null;
        });
        _logOffline("Loaded source image: ${image.name}");
      }
    } catch (e) {
      _logOffline("Failed to pick image: $e");
    }
  }

  Future<void> _runInference() async {
    if (_pickedImagePath == null) return;

    setState(() {
      _isProcessing = true;
    });
    _logOffline("Preprocessing image...");

    try {
      final results = await _estimator.estimateDepth(_pickedImagePath!);

      setState(() {
        _depthMapPath = results['depthMapPath'] as String;
        _preprocessTime = results['preprocessTimeMs'] as int;
        _inferenceTime = results['inferenceTimeMs'] as int;
        _postprocessTime = results['postprocessTimeMs'] as int;
        _totalTime = results['totalTimeMs'] as int;
      });

      _logOffline(
          "Inference complete! Saved depth map: ${_depthMapPath!.split('/').last}");
      _logOffline("NPU/CPU Inference Latency: $_inferenceTime ms");
    } catch (e) {
      _logOffline("Inference error: $e");
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Inference failed: $e')),
      );
    } finally {
      setState(() {
        _isProcessing = false;
      });
    }
  }

  // --- Live Stream Tab Code ---
  Future<void> _initCamera() async {
    _logStream("Initializing camera...");
    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        _logStream("No camera sensors found.");
        return;
      }

      CameraDescription? backCamera;
      for (var c in _cameras) {
        if (c.lensDirection == CameraLensDirection.back) {
          backCamera = c;
          break;
        }
      }
      backCamera ??= _cameras.first;

      _cameraController = CameraController(
        backCamera,
        ResolutionPreset.medium, // Resolves to ~720x480 / 640x480 which is perfect for streaming
        enableAudio: false,
      );

      await _cameraController!.initialize();
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = true;
      });
      _logStream("Camera feed active.");
    } catch (e) {
      _logStream("Camera failed to initialize: $e");
    }
  }

  void _disposeCamera() {
    _cameraController?.dispose();
    _cameraController = null;
    if (mounted) {
      setState(() {
        _isCameraInitialized = false;
      });
    }
  }

  void _toggleConnect() {
    if (_streamManager.connectionState == StreamConnectionState.connected) {
      _streamManager.disconnect();
    } else if (_streamManager.connectionState ==
        StreamConnectionState.disconnected) {
      final ip = _ipController.text.trim();
      final port = int.tryParse(_portController.text.trim()) ?? 3000;
      _streamManager.connect(ip, port);
    }
  }

  void _toggleRecord() {
    HapticFeedback.lightImpact();
    if (_streamManager.isRecording) {
      _stopStreaming();
    } else {
      _startStreaming();
    }
  }

  void _startStreaming() {
    _streamManager.toggleRecording();
    if (_streamManager.isRecording) {
      // Periodic frame grabber (approx 15 FPS -> every 66 milliseconds)
      _streamTimer = Timer.periodic(const Duration(milliseconds: 66), (timer) {
        _captureAndSendFrame();
      });
    }
  }

  void _stopStreaming() {
    _streamTimer?.cancel();
    _streamTimer = null;
    if (_streamManager.isRecording) {
      _streamManager.toggleRecording();
    }
    if (mounted) setState(() {});
  }

  Future<void> _captureAndSendFrame() async {
    if (!_streamManager.isRecording ||
        _streamManager.connectionState != StreamConnectionState.connected ||
        _cameraController == null ||
        !_cameraController!.value.isInitialized) {
      return;
    }

    try {
      final RenderRepaintBoundary? boundary =
          _boundaryKey.currentContext?.findRenderObject()
              as RenderRepaintBoundary?;
      if (boundary != null) {
        // Downscale capture to match stream suggested width natively (improves bandwidth usage)
        final ui.Image image = await boundary.toImage(pixelRatio: 0.5);
        final ByteData? byteData =
            await image.toByteData(format: ui.ImageByteFormat.png);
        image.dispose(); // Release graphic handle immediately to prevent GPU memory leaks
        
        if (byteData != null) {
          final Uint8List pngBytes = byteData.buffer.asUint8List();
          _streamManager.sendFrame(pngBytes);
        }
      }
    } catch (e) {
      _logStream("Frame capture skipped: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SplatMesh Scanner',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 0.5),
        ),
        centerTitle: true,
        backgroundColor: const Color(0xFF121620),
        elevation: 0,
        bottom: TabBar(
          controller: _tabController,
          indicatorColor: Colors.cyanAccent,
          tabs: const [
            Tab(icon: Icon(Icons.psychology), text: 'Offline Bench'),
            Tab(icon: Icon(Icons.sensors), text: 'Live Scanner'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        physics: const NeverScrollableScrollPhysics(), // Prevent camera swipe conflict
        children: [
          _buildOfflineTab(),
          _buildLiveTab(),
        ],
      ),
    );
  }

  // --- Tab 1 UI: Offline Depth Inference ---
  Widget _buildOfflineTab() {
    final theme = Theme.of(context);
    return SafeArea(
      child: Column(
        children: [
          if (_isModelLoading)
            LinearProgressIndicator(
              backgroundColor: theme.colorScheme.surface,
              valueColor: AlwaysStoppedAnimation<Color>(theme.colorScheme.secondary),
            ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.indigoAccent,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          icon: const Icon(Icons.photo_library),
                          label: const Text('Gallery'),
                          onPressed: _isProcessing || _isModelLoading
                              ? null
                              : () => _pickImage(ImageSource.gallery),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: Colors.cyan.shade700,
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                          icon: const Icon(Icons.camera_alt),
                          label: const Text('Camera'),
                          onPressed: _isProcessing || _isModelLoading
                              ? null
                              : () => _pickImage(ImageSource.camera),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  _buildComparisonView(theme),
                  const SizedBox(height: 16),
                  if (_pickedImagePath != null && _depthMapPath == null)
                    ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: const Color(0xFF059669),
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 16),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      onPressed: _isProcessing ? null : _runInference,
                      child: _isProcessing
                          ? const SizedBox(
                              height: 20,
                              width: 20,
                              child: CircularProgressIndicator(
                                strokeWidth: 2,
                                valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                              ),
                            )
                          : const Text('ESTIMATE DEPTH (RUN ON NPU)',
                              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15)),
                    ),
                  const SizedBox(height: 16),
                  _buildLatencyMetrics(theme),
                  const SizedBox(height: 16),
                  _buildLogsConsole(_logsOffline, _estimator.isInitialized),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildComparisonView(ThemeData theme) {
    if (_pickedImagePath == null) {
      return Container(
        height: 220,
        decoration: BoxDecoration(
          color: theme.colorScheme.surface,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white12, width: 1.5),
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.image_search, size: 48, color: Colors.indigo.shade200),
            const SizedBox(height: 12),
            const Text(
              'No Image Selected',
              style: TextStyle(fontWeight: FontWeight.w600, fontSize: 16),
            ),
            const SizedBox(height: 4),
            const Text(
              'Capture or select an image from gallery to test',
              style: TextStyle(color: Colors.white54, fontSize: 13),
            ),
          ],
        ),
      );
    }

    return Column(
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Original Image',
                        style: TextStyle(
                            fontWeight: FontWeight.bold, color: Colors.indigoAccent)),
                    if (_depthMapPath != null)
                      const Text('Depth Estimation Output',
                          style: TextStyle(
                              fontWeight: FontWeight.bold, color: Colors.cyanAccent)),
                  ],
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: Container(
                        height: 200,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          image: DecorationImage(
                            image: FileImage(File(_pickedImagePath!)),
                            fit: BoxFit.cover,
                          ),
                        ),
                      ),
                    ),
                    if (_depthMapPath != null) ...[
                      const SizedBox(width: 8),
                      Expanded(
                        child: Container(
                          height: 200,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(12),
                            image: DecorationImage(
                              image: FileImage(File(_depthMapPath!)),
                              fit: BoxFit.cover,
                            ),
                          ),
                        ),
                      ),
                    ]
                  ],
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLatencyMetrics(ThemeData theme) {
    if (_totalTime == null) return const SizedBox.shrink();

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Row(
              children: [
                Icon(Icons.speed, color: Colors.cyanAccent, size: 20),
                SizedBox(width: 8),
                Text(
                  'Latency Benchmarks (On-Device)',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const Divider(height: 20, color: Colors.white10),
            _buildMetricRow('Image Preprocessing', '$_preprocessTime ms'),
            _buildMetricRow('ONNX Inference (NPU/CPU)', '$_inferenceTime ms', highlight: true),
            _buildMetricRow('Output Postprocessing', '$_postprocessTime ms'),
            const Divider(height: 20, color: Colors.white10),
            _buildMetricRow('Total Frame Latency', '$_totalTime ms', bold: true),
          ],
        ),
      ),
    );
  }

  Widget _buildMetricRow(String label, String value, {bool highlight = false, bool bold = false}) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 13,
              fontWeight: bold ? FontWeight.bold : FontWeight.normal,
              color: bold ? Colors.white : Colors.white70,
            ),
          ),
          Text(
            value,
            style: TextStyle(
              fontSize: 14,
              fontFamily: 'monospace',
              fontWeight: FontWeight.bold,
              color: highlight ? Colors.cyanAccent : (bold ? Colors.white : Colors.greenAccent),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLogsConsole(List<String> logs, bool isOnline) {
    return Card(
      color: const Color(0xFF0F111A),
      child: Padding(
        padding: const EdgeInsets.all(12.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Diagnostics Console',
                  style: TextStyle(
                      fontFamily: 'monospace',
                      fontSize: 13,
                      color: Colors.grey,
                      fontWeight: FontWeight.bold),
                ),
                Text(
                  isOnline ? 'ONLINE' : 'OFFLINE',
                  style: TextStyle(
                    fontFamily: 'monospace',
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: isOnline ? Colors.green : Colors.red,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Container(
              height: 120,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(8),
                color: Colors.black45,
              ),
              child: ListView.builder(
                padding: const EdgeInsets.all(8),
                itemCount: logs.length,
                itemBuilder: (context, index) {
                  return Padding(
                    padding: const EdgeInsets.symmetric(vertical: 2.0),
                    child: Text(
                      logs[index],
                      style: const TextStyle(
                        fontFamily: 'monospace',
                        fontSize: 11,
                        color: Colors.greenAccent,
                      ),
                    ),
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  // --- Tab 2 UI: Live Streaming & IMU Sensors ---
  Widget _buildLiveTab() {
    final statusColor = _streamManager.connectionState == StreamConnectionState.connected
        ? const Color(0x3334C759)
        : (_streamManager.connectionState == StreamConnectionState.connecting
            ? const Color(0x33FFCC00)
            : const Color(0x33FF3B30));
    final statusText = _streamManager.connectionState == StreamConnectionState.connected
        ? 'Server Ready'
        : (_streamManager.connectionState == StreamConnectionState.connecting
            ? 'Connecting...'
            : 'Disconnected');

    return SafeArea(
      child: Column(
        children: [
          // 1. Connection Status Banner
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 16),
            color: statusColor,
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Row(
                  children: [
                    Container(
                      width: 8,
                      height: 8,
                      decoration: BoxDecoration(
                        color: _streamManager.connectionState == StreamConnectionState.connected
                            ? Colors.greenAccent
                            : (_streamManager.connectionState == StreamConnectionState.connecting
                                ? Colors.amber
                                : Colors.redAccent),
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(
                      statusText,
                      style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                Text(
                  'Width: ${_streamManager.streamWidth}px',
                  style: const TextStyle(fontSize: 12, color: Colors.white70, fontFamily: 'monospace'),
                ),
              ],
            ),
          ),
          
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // 2. Server Configuration Card
                  Card(
                    child: Padding(
                      padding: const EdgeInsets.all(12.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text('PC Relay Server Settings',
                              style: TextStyle(fontWeight: FontWeight.bold, color: Colors.indigoAccent)),
                          const SizedBox(height: 8),
                          Row(
                            children: [
                              Expanded(
                                flex: 3,
                                child: TextField(
                                  controller: _ipController,
                                  enabled: _streamManager.connectionState == StreamConnectionState.disconnected,
                                  decoration: const InputDecoration(
                                    labelText: 'Server IP',
                                    border: OutlineInputBorder(),
                                    isDense: true,
                                  ),
                                  style: const TextStyle(fontSize: 14, fontFamily: 'monospace'),
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                flex: 2,
                                child: TextField(
                                  controller: _portController,
                                  enabled: _streamManager.connectionState == StreamConnectionState.disconnected,
                                  keyboardType: TextInputType.number,
                                  decoration: const InputDecoration(
                                    labelText: 'Port',
                                    border: OutlineInputBorder(),
                                    isDense: true,
                                  ),
                                  style: const TextStyle(fontSize: 14, fontFamily: 'monospace'),
                                ),
                              ),
                              const SizedBox(width: 8),
                              ElevatedButton(
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: _streamManager.connectionState == StreamConnectionState.connected
                                      ? Colors.red.shade800
                                      : Colors.indigoAccent,
                                  padding: const EdgeInsets.all(14),
                                  shape: RoundedRectangleBorder(
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                ),
                                onPressed: _toggleConnect,
                                child: Icon(
                                  _streamManager.connectionState == StreamConnectionState.connected
                                      ? Icons.link_off
                                      : Icons.link,
                                  color: Colors.white,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ),

                  // 3. Camera Viewport & Telemetry Stack
                  ClipRRect(
                    borderRadius: BorderRadius.circular(16),
                    child: AspectRatio(
                      aspectRatio: 4 / 3,
                      child: Stack(
                        fit: StackFit.expand,
                        children: [
                          // A. Camera Preview
                          RepaintBoundary(
                            key: _boundaryKey,
                            child: _isCameraInitialized && _cameraController != null
                                ? CameraPreview(_cameraController!)
                                : Container(
                                    color: Colors.black87,
                                    child: const Column(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: [
                                        Icon(Icons.camera, size: 40, color: Colors.white30),
                                        SizedBox(height: 8),
                                        Text('Camera Standby', style: TextStyle(color: Colors.white30)),
                                      ],
                                    ),
                                  ),
                          ),

                          // B. 3x3 Alignment Grid (Drawn when recording/streaming is active)
                          if (_streamManager.isRecording)
                            Positioned.fill(
                              child: IgnorePointer(
                                child: CustomPaint(
                                  painter: AlignmentGridPainter(),
                                ),
                              ),
                            ),

                          // C. Telemetry HUD Overlay
                          if (_streamManager.isRecording)
                            Positioned(
                              top: 12,
                              right: 12,
                              child: Container(
                                  padding: const EdgeInsets.all(8),
                                  decoration: BoxDecoration(
                                    color: const Color(0xA6000000), // Black with 65% opacity
                                    borderRadius: BorderRadius.circular(8),
                                    border: Border.all(color: Colors.white24, width: 1),
                                  ),
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.end,
                                  children: [
                                    Row(
                                      mainAxisSize: MainAxisSize.min,
                                      children: [
                                        const PulseDotWidget(),
                                        const SizedBox(width: 6),
                                        Text(
                                          '${_streamManager.framesSent} frames',
                                          style: const TextStyle(
                                            fontFamily: 'monospace',
                                            fontWeight: FontWeight.bold,
                                            fontSize: 12,
                                          ),
                                        ),
                                      ],
                                    ),
                                    const SizedBox(height: 4),
                                    Text('P: ${_streamManager.pitch.toStringAsFixed(1)}°',
                                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: Colors.white70)),
                                    Text('R: ${_streamManager.roll.toStringAsFixed(1)}°',
                                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: Colors.white70)),
                                    Text('Y: ${_streamManager.yaw.toStringAsFixed(1)}°',
                                        style: const TextStyle(fontFamily: 'monospace', fontSize: 11, color: Colors.white70)),
                                  ],
                                ),
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // 4. Record/Stream Haptic Circular Trigger
                  Center(
                    child: GestureDetector(
                      onTap: _streamManager.connectionState == StreamConnectionState.connected
                          ? _toggleRecord
                          : null,
                      child: Container(
                        width: 76,
                        height: 76,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: _streamManager.connectionState == StreamConnectionState.connected
                              ? Colors.redAccent
                              : Colors.grey.shade800,
                          border: Border.all(color: Colors.white, width: 4),
                          boxShadow: [
                            BoxShadow(
                              color: const Color(0x66000000), // Black with 40% opacity
                              blurRadius: 16,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 150),
                          margin: EdgeInsets.all(_streamManager.isRecording ? 18 : 6),
                          decoration: BoxDecoration(
                            color: Colors.white,
                            borderRadius: BorderRadius.circular(_streamManager.isRecording ? 8 : 36),
                          ),
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),

                  // 5. Diagnostics Terminal Logs Console
                  _buildLogsConsole(_logsStream, _streamManager.connectionState == StreamConnectionState.connected),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

// Custom Painter to draw a clean 3x3 alignment grid over the camera view
class AlignmentGridPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = const Color(0x4DFFFFFF) // White with 30% opacity
      ..strokeWidth = 1.0
      ..style = PaintingStyle.stroke;

    // Horizontal Lines
    canvas.drawLine(Offset(0, size.height / 3), Offset(size.width, size.height / 3), paint);
    canvas.drawLine(Offset(0, 2 * size.height / 3), Offset(size.width, 2 * size.height / 3), paint);

    // Vertical Lines
    canvas.drawLine(Offset(size.width / 3, 0), Offset(size.width / 3, size.height), paint);
    canvas.drawLine(Offset(2 * size.width / 3, 0), Offset(2 * size.width / 3, size.height), paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

// Widget to render the pulsing red recording indicator
class PulseDotWidget extends StatefulWidget {
  const PulseDotWidget({super.key});

  @override
  State<PulseDotWidget> createState() => _PulseDotWidgetState();
}

class _PulseDotWidgetState extends State<PulseDotWidget> with SingleTickerProviderStateMixin {
  late AnimationController _animController;
  late Animation<double> _opacityAnim;

  @override
  void initState() {
    super.initState();
    _animController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 750),
    )..repeat(reverse: true);

    _opacityAnim = Tween<double>(begin: 0.3, end: 1.0).animate(
      CurvedAnimation(parent: _animController, curve: Curves.easeInOut),
    );
  }

  @override
  void dispose() {
    _animController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return FadeTransition(
      opacity: _opacityAnim,
      child: Container(
        width: 8,
        height: 8,
        decoration: const BoxDecoration(
          color: Colors.redAccent,
          shape: BoxShape.circle,
        ),
      ),
    );
  }
}
