import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:onnxruntime/onnxruntime.dart';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  img.Image? _preprocessedImage;
  final ImagePicker _picker = ImagePicker();
  String? _prediction;
  OrtSession? _session;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    final modelBytes = await rootBundle.load('assets/model.onnx');

    final sessionOptions = OrtSessionOptions();

    final session = await OrtSession.fromBuffer(
      modelBytes.buffer.asUint8List(),
      sessionOptions,
    );
    setState(() {
      _session = session;
    });
  }

  Future<void> _getImageFromCamera() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      setState(() {
        _image = File(image.path);
        _prediction = null;
      });
      await _classifyImage(File(image.path));
    }
  }

  Future<void> _classifyImage(File imageFile) async {
    if (_session == null) return;
    // Load and decode image
    final imgBytes = await imageFile.readAsBytes();
    img.Image? oriImage = img.decodeImage(imgBytes);
    if (oriImage == null) return;
    // Resize to 518x518
    img.Image resized = img.copyResize(oriImage, width: 518, height: 518);
    // Save preprocessed image for visualization (denormalized)
    setState(() {
      _preprocessedImage = img.copyResize(resized, width: 518, height: 518);
    });
    // Convert to Float32List and normalize
    Float32List input = Float32List(1 * 3 * 518 * 518);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    int idx = 0;
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 518; y++) {
        for (int x = 0; x < 518; x++) {
          final pixel = resized.getPixel(x, y);
          double value = 0;
          if (c == 0) {
            value = pixel.r / 255.0;
          } else if (c == 1) {
            value = pixel.g / 255.0;
          } else if (c == 2) {
            value = pixel.b / 255.0;
          }
          input[idx++] = ((value - mean[c]) / std[c]).toDouble();
        }
      }
    }
    // Get input/output names (hardcoded for MobileNetV3 ONNX export)
    final inputName = 'l_x_'; // or check your model's input name
    // Wrap input in OrtValueTensor
    final inputTensor = OrtValueTensor.createTensorWithDataList(input, [
      1,
      3,
      518,
      518,
    ]);
    // Run inference
    final runOptions = OrtRunOptions();
    final outputs = await _session!.run(runOptions, {inputName: inputTensor});
    inputTensor.release();
    // Get output tensor (first output)
    final outputTensor = outputs[1]?.value as List<List<int>>;
    int predIdx = outputTensor[0][0];
    setState(() {
      _prediction = 'Predicted class index: $predIdx';
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Text(widget.title),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? const Text('No image selected.')
                : Image.file(_image!, height: 300),
            const SizedBox(height: 20),
            _preprocessedImage != null
                ? Image.memory(
                    Uint8List.fromList(img.encodeJpg(_preprocessedImage!)),
                    height: 224,
                  )
                : Container(),
            const SizedBox(height: 20),
            _prediction != null
                ? Text(_prediction!, style: const TextStyle(fontSize: 18))
                : Container(),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: _getImageFromCamera,
              icon: const Icon(Icons.camera_alt),
              label: const Text('Take a Picture'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(
                  horizontal: 24,
                  vertical: 12,
                ),
                textStyle: const TextStyle(fontSize: 18),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
