import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:flutter_image_compress/flutter_image_compress.dart';

enum StreamConnectionState {
  disconnected,
  connecting,
  connected,
}

class LiveStreamManager {
  // Config
  String ip = '10.91.53.25';
  int port = 3000;
  int streamWidth = 640; // Updated dynamically by server handshake

  // Connection
  WebSocket? _ws;
  StreamConnectionState connectionState = StreamConnectionState.disconnected;
  bool isRecording = false;
  int framesSent = 0;

  // Sensor Telemetry
  double pitch = 0.0;
  double roll = 0.0;
  double yaw = 0.0;

  // Stream Subscriptions
  StreamSubscription<AccelerometerEvent>? _accelSub;
  StreamSubscription<MagnetometerEvent>? _magnetSub;
  int _accelEventCount = 0;
  int _magnetEventCount = 0;

  // Latest sensor readings
  AccelerometerEvent? _latestAccel;
  MagnetometerEvent? _latestMagnet;

  // Callbacks
  Function(StreamConnectionState)? onConnectionStateChanged;
  Function(String)? onLogReceived;
  Function(double p, double r, double y, int frames)? onTelemetryUpdated;

  // Low pass filter alpha for smoothing sensor noise
  static const double _filterAlpha = 0.15;

  LiveStreamManager();

  /// Starts listening to the Accelerometer and Magnetometer sensors.
  void startSensors() {
    _log("Starting device accelerometer and magnetometer sensors...");
    
    _accelSub = accelerometerEventStream().listen((AccelerometerEvent event) {
      _latestAccel = event;
      _accelEventCount++;
      if (_accelEventCount % 50 == 0) {
        _log("Sensor Accel tick: ax=${event.x.toStringAsFixed(1)}, ay=${event.y.toStringAsFixed(1)}");
      }
      _updateOrientation();
    });

    _magnetSub = magnetometerEventStream().listen((MagnetometerEvent event) {
      _latestMagnet = event;
      _magnetEventCount++;
      if (_magnetEventCount % 50 == 0) {
        _log("Sensor Magnet tick: mx=${event.x.toStringAsFixed(1)}, my=${event.y.toStringAsFixed(1)}");
      }
      _updateOrientation();
    });
  }

  /// Stops sensor streams.
  void stopSensors() {
    _log("Stopping sensors...");
    _accelSub?.cancel();
    _magnetSub?.cancel();
    _latestAccel = null;
    _latestMagnet = null;
  }

  /// Computes Pitch, Roll, and Yaw using tilt-compensated sensor fusion.
  void _updateOrientation() {
    if (_latestAccel == null) return;

    final double ax = _latestAccel!.x;
    final double ay = _latestAccel!.y;
    final double az = _latestAccel!.z;

    // 1. Calculate Pitch and Roll from Accelerometer
    // pitchRad (rotation around X-axis): -pi/2 to pi/2
    // rollRad (rotation around Y-axis): -pi to pi
    final double targetPitchRad = atan2(ay, sqrt(ax * ax + az * az));
    final double targetRollRad = atan2(-ax, az);

    // Apply low-pass filter to smooth out sensor jitter
    final double pitchRad = (pitch * pi / 180) * (1 - _filterAlpha) + targetPitchRad * _filterAlpha;
    final double rollRad = (roll * pi / 180) * (1 - _filterAlpha) + targetRollRad * _filterAlpha;

    pitch = pitchRad * 180 / pi;
    roll = rollRad * 180 / pi;

    // 2. Calculate Yaw (Heading) using Accelerometer + Magnetometer (Tilt-compensated, if available)
    if (_latestMagnet != null) {
      final double mx = _latestMagnet!.x;
      final double my = _latestMagnet!.y;
      final double mz = _latestMagnet!.z;

      final double cosPitch = cos(pitchRad);
      final double sinPitch = sin(pitchRad);
      final double cosRoll = cos(rollRad);
      final double sinRoll = sin(rollRad);

      final double xh = mx * cosRoll + mz * sinRoll;
      final double yh = mx * sinPitch * sinRoll + my * cosPitch - mz * sinPitch * cosRoll;

      double yawRad = atan2(-yh, xh);
      double targetYawDeg = yawRad * 180 / pi;
      if (targetYawDeg < 0) {
        targetYawDeg += 360;
      }

      // Apply smoothing to Yaw (handling boundary wrap-around cleanly)
      double diff = targetYawDeg - yaw;
      if (diff < -180) diff += 360;
      if (diff > 180) diff -= 360;
      yaw = (yaw + diff * _filterAlpha) % 360;
    }

    onTelemetryUpdated?.call(pitch, roll, yaw, framesSent);
  }

  /// Establishes the WebSocket connection to the server.
  Future<void> connect(String targetIp, int targetPort) async {
    ip = targetIp;
    port = targetPort;
    _setConnectionState(StreamConnectionState.connecting);
    _log("Connecting to WebSocket ws://$ip:$port/ws...");

    try {
      _ws = await WebSocket.connect('ws://$ip:$port/ws')
          .timeout(const Duration(seconds: 5));
      _setConnectionState(StreamConnectionState.connected);
      _log("Successfully connected to PC relay server!");

      _ws!.listen(
        (data) {
          if (data is String) {
            _handleServerMessage(data);
          }
        },
        onDone: () {
          _log("WebSocket closed by server.");
          disconnect();
        },
        onError: (err) {
          _log("WebSocket error: $err");
          disconnect();
        },
        cancelOnError: true,
      );
    } catch (e) {
      _log("Failed to connect: $e");
      _setConnectionState(StreamConnectionState.disconnected);
    }
  }

  /// Closes the connection and stops recording.
  void disconnect() {
    if (isRecording) {
      toggleRecording();
    }
    _ws?.close();
    _ws = null;
    _setConnectionState(StreamConnectionState.disconnected);
    _log("Disconnected from server.");
  }

  /// Sends the start/stop command to the server and updates state.
  void toggleRecording() {
    if (connectionState != StreamConnectionState.connected || _ws == null) {
      _log("Cannot record: Not connected.");
      return;
    }

    if (!isRecording) {
      _log("Sending START signal to server...");
      _ws!.add("START");
      isRecording = true;
      framesSent = 0;
    } else {
      _log("Sending STOP signal to server...");
      _ws!.add("STOP");
      isRecording = false;
    }
    onTelemetryUpdated?.call(pitch, roll, yaw, framesSent);
  }

  /// Handles incoming string messages from the server (e.g. handshake config).
  void _handleServerMessage(String message) {
    try {
      final config = jsonDecode(message);
      if (config is Map && config.containsKey('suggested_width')) {
        streamWidth = config['suggested_width'];
        final acc = config['hardware_acceleration'] ?? 'unknown';
        _log("Handshake received! Stream width: ${streamWidth}px. NPU/GPU Acc: $acc");
      }
    } catch (e) {
      _log("Received unparsed text message: $message");
    }
  }

  /// Natively encodes the frame as JPEG and streams it along with packed IMU header bytes.
  Future<void> sendFrame(Uint8List pngBytes) async {
    if (connectionState != StreamConnectionState.connected || _ws == null || !isRecording) {
      return;
    }

    try {
      // 1. Convert PNG bytes to JPEG using native platform compressor (extremely fast)
      final jpegBytes = await FlutterImageCompress.compressWithList(
        pngBytes,
        format: CompressFormat.jpeg,
        quality: 85,
      );

      // 2. Build little-endian binary header (32 bytes)
      final header = ByteData(32);
      // Offset 0-7: Timestamp (int64 milliseconds)
      header.setInt64(0, DateTime.now().millisecondsSinceEpoch, Endian.little);
      // Offset 8-11: Pitch (float32)
      header.setFloat32(8, pitch, Endian.little);
      // Offset 12-15: Roll (float32)
      header.setFloat32(12, roll, Endian.little);
      // Offset 16-19: Yaw (float32)
      header.setFloat32(16, yaw, Endian.little);
      // Offset 20-31: Unused padding bytes (zeroed out automatically)

      // 3. Concatenate header and JPEG payload
      final packet = Uint8List(32 + jpegBytes.length);
      packet.setRange(0, 32, header.buffer.asUint8List(header.offsetInBytes, header.lengthInBytes));
      packet.setRange(32, packet.length, jpegBytes);

      // 4. Send binary packet over WebSocket
      _ws!.add(packet);
      
      // Log telemetries once a second to verify packing values
      if (framesSent % 15 == 0) {
        _log("Packed binary frame #$framesSent: TS=${DateTime.now().millisecondsSinceEpoch}, P=${pitch.toStringAsFixed(1)}, R=${roll.toStringAsFixed(1)}, Y=${yaw.toStringAsFixed(1)}");
      }

      framesSent++;
      onTelemetryUpdated?.call(pitch, roll, yaw, framesSent);
    } catch (e) {
      _log("Error packing/sending frame: $e");
    }
  }

  void _setConnectionState(StreamConnectionState state) {
    connectionState = state;
    onConnectionStateChanged?.call(state);
  }

  void _log(String msg) {
    final timestamp = DateTime.now().toString().substring(11, 19);
    onLogReceived?.call("[$timestamp] $msg");
  }
}
